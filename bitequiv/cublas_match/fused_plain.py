"""The `plain` cuBLAS plan mode with a pointwise epilogue folded in before the store.

`plain` is one fp32 accumulator over the whole of K, fed by `tl.dot`s of BK real k-elements in
increasing k, scaled once and rounded once.  That is the whole bit contract, and none of it is
touched here: the k loop below is `kernels._plain_gemm`'s and `kernels._plain_gemm_tma_sm103`'s,
statement for statement.  The only difference is what happens to `acc` after the loop.

WHAT MAKES A FUSED KERNEL EQUAL TO THE UNFUSED PATH
---------------------------------------------------
The unfused path a user gets today is: cuBLAS writes the GEMM's output to memory *in the output
dtype*, and a second kernel reads it back and applies the epilogue.  So the accumulator has to be
rounded through the output dtype before the epilogue sees it, and every later op has to round
where eager's kernel boundary rounds:

    c = acc.to(out_dtype)            # the round cuBLAS did when it wrote to memory
    y = epilogue(c.to(tl.float32))   # the epilogue in fp32, as torch eager does it
    store(y.to(out_dtype))

Dropping any one of those rounds is what makes an ordinary fused epilogue faster and not
byte-identical.

WHY THE EPILOGUE ARRIVES AS TEXT
--------------------------------
The epilogue spellings are not written here.  They are verified elsewhere over their complete
input domain -- `artifact_eval/fusion_oracle/probe_pairs.txt` records the counts -- and a second
transcription of them in this file could only ever drift out of agreement with the verified one.
So `build()` takes the epilogue as the list of source lines the caller already trusts, splices it
into the templates below, and this file never learns which epilogues exist.

    from artifact_eval.fusion_oracle.inductor_kernel import EPI_SRC
    k = build(EPI_SRC["swiglu"], ("mn",), "fp16", cache_dir)
    c = launch_pre(k, a, b, torch.float16, extras, {"BM": 128, ...})

Two traps are recorded in that same file and both bite here.  `tl.minimum` / `tl.maximum` return
the non-NaN operand where `torch.clamp` keeps the NaN, so a clamp has to be the compare-and-select
pair (`triton_helpers.minimum`); and a `.to(tl.float16)` does NOT survive the compiler unless the
launch passes `enable_fp_fusion=False`, because LLVM contracts the chain into one `fma.rn.f16`
and skips the round the cast asked for.  Every launcher here passes it.

TWO KERNEL SHAPES, THE SAME ARITHMETIC
--------------------------------------
`_PRE_SRC`   `kernels._plain_gemm`: the kernel as it was before the sm_103 rewrites -- one
             program per output tile, row-major tile order, masked k loop, int64 addressing.
`_GB300_SRC` `kernels._plain_gemm_tma_sm103`: TMA operands, a persistent grid and warp
             specialization, which is what the package runs on a GB300.
`_GB300_NOTMA_SRC`
             `kernels._plain_gemm_sm103`: the same launch shape without TMA, for an operand that
             cannot carry a tensor descriptor.  The shipped launcher falls back to it too.

Three differences from the shipped GB300 kernel are deliberate and are the price of having an
epilogue at all:

  * `SUBTILE` is off.  It splits the store in two along N, and an epilogue that reads a second
    [BM, BN] operand would have to be split with it.
  * the extra operands are read with ordinary `tl.load`, not through a tensor descriptor.  A
    per-token [M, 1] column cannot carry one, so one path serves every operand kind.
  * `enable_fp_fusion=False`, as above.

None of the three can move a bit; they cost speed, and the arm they cost it to is ours.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import sys

import torch
import triton

from .ltapi import DEVICE

# --------------------------------------------------------------------------------------------
# operand plumbing
# --------------------------------------------------------------------------------------------
# An epilogue may read tensors besides the accumulator: the gate projection for a SwiGLU, the
# base layer's output for a LoRA merge, a per-token routing weight for an MoE down projection.
# Three kinds appear, and all three are indexed through ONE expression by giving a broadcast axis
# a stride of zero -- the same trick `fusion_oracle/inductor_kernel.launch_standalone` uses:
#
#   mn  a full [M, N] tensor          (sm, sn) = its own strides
#   m1  a per-token [M, 1] column     (sm, 0)
#   n   a [N] row                     (0, sn)
OPERAND_KINDS = ("mn", "m1", "n")


def operand_strides(kind, t):
    """The (row, column) stride pair to pass for one extra operand, zeros on broadcast axes."""
    if kind == "n":
        return 0, t.stride(0)
    if kind == "m1":
        return (t.stride(0) if t.shape[0] != 1 else 0), 0
    return (t.stride(0) if t.shape[0] != 1 else 0), (t.stride(1) if t.shape[1] != 1 else 0)


def _decl(n):
    """Parameter text for `n` extra operands: the pointers, and their stride pairs."""
    ptrs = "".join(f"E{i}, " for i in range(n))
    strides = "".join(f"e{i}m, e{i}n, " for i in range(n))
    return ptrs, strides


def _loads(n, row, col, mask, indent="    "):
    """The load of each extra operand, named as the epilogue text expects.

    `tmp_e<i>` is the name `EPI_SRC` uses, and `<i>` is the operand's position in the epilogue's
    declared list -- never the order the loads happen to appear in, which is not ours to choose.
    The raw value is widened to fp32 exactly once, so the epilogue text does not have to know
    what dtype it came out of memory as.
    """
    out = ""
    for i in range(n):
        out += (f"{indent}tmp_e{i}_raw = tl.load(E{i} + {row} * e{i}m + {col} * e{i}n, "
                f"mask={mask}, other=0.0)\n"
                f"{indent}tmp_e{i} = tmp_e{i}_raw.to(tl.float32)\n")
    return out


_HEAD = """
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers  # noqa: F401
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math  # noqa: F401
"""

# --------------------------------------------------------------------------------------------
# the pre-sm_103 kernel: `kernels._plain_gemm` with the epilogue folded in
# --------------------------------------------------------------------------------------------
_PRE_SRC = '''

@triton.jit
def fused_pre(A, B, C, {ptrs}M, N, K, am, ak, bk, bn, cm, cn, {strides}s,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
    on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
    ok = tl.arange(0, BK).to(tl.int64)
    ap = A + om[:, None] * am + ok[None, :] * ak
    bp = B + ok[:, None] * bk + on[None, :] * bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(ap, mask=ok[None, :] < K - k * BK, other=0.0),
                     tl.load(bp, mask=ok[:, None] < K - k * BK, other=0.0), acc)
        ap += BK * ak
        bp += BK * bk
    acc = acc * s
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    mask = (ocm[:, None] < M) & (ocn[None, :] < N)
{loads}{body}
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], tmp_out, mask=mask)
'''

# --------------------------------------------------------------------------------------------
# the GB300 kernel: `kernels._plain_gemm_tma_sm103` with the epilogue folded in
# --------------------------------------------------------------------------------------------
_GB300_SRC = '''

@triton.jit
def fused_gb300(a_desc, b_desc, c_desc, {ptrs}M, N, K, {strides}s,
                B_NK: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, WS: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    start_pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    num_tiles = npm * npn
    per_group = GROUP_M * npn

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=WS):
        gid = tile_id // per_group
        first = gid * GROUP_M
        rows = tl.minimum(npm - first, GROUP_M)
        om = (first + (tile_id % per_group) % rows) * BM
        on = ((tile_id % per_group) // rows) * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            ok = ki * BK
            a = a_desc.load([om, ok])
            if B_NK:
                acc = tl.dot(a, b_desc.load([on, ok]).T, acc)
            else:
                acc = tl.dot(a, b_desc.load([ok, on]), acc)
        acc = acc * s
        rm = (om + tl.arange(0, BM)).to(tl.int64)
        rn = (on + tl.arange(0, BN)).to(tl.int64)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
{loads}{body}
        c_desc.store([om, on], tmp_out.to(c_desc.dtype))
'''

# --------------------------------------------------------------------------------------------
# the GB300 kernel without TMA: `kernels._plain_gemm_sm103` with the epilogue folded in
# --------------------------------------------------------------------------------------------
_GB300_NOTMA_SRC = '''

@triton.jit
def fused_gb300_notma(A, B, C, {ptrs}M, N, K, am, ak, bk, bn, cm, cn, {strides}s,
                      GROUP_M: tl.constexpr, EVEN_K: tl.constexpr, I64: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    if GROUP_M == 1:
        pm = pid // npn
        pn = pid % npn
    else:
        per_group = GROUP_M * npn
        gid = pid // per_group
        first = gid * GROUP_M
        rows = tl.minimum(npm - first, GROUP_M)
        pm = first + (pid % per_group) % rows
        pn = (pid % per_group) // rows
    if I64:
        om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
        on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
        ok = tl.arange(0, BK).to(tl.int64)
    else:
        om = (pm * BM + tl.arange(0, BM)) % M
        on = (pn * BN + tl.arange(0, BN)) % N
        ok = tl.arange(0, BK)
    ap = A + om[:, None] * am + ok[None, :] * ak
    bp = B + ok[:, None] * bk + on[None, :] * bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        if EVEN_K:
            acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        else:
            acc = tl.dot(tl.load(ap, mask=ok[None, :] < K - k * BK, other=0.0),
                         tl.load(bp, mask=ok[:, None] < K - k * BK, other=0.0), acc)
        ap += BK * ak
        bp += BK * bk
    acc = acc * s
    if I64:
        ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
        ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    else:
        ocm = pm * BM + tl.arange(0, BM)
        ocn = pn * BN + tl.arange(0, BN)
    mask = (ocm[:, None] < M) & (ocn[None, :] < N)
{loads}{body}
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], tmp_out, mask=mask)
'''


def round_lines(lines, round_dtype):
    """`EPI_SRC` is written for an fp16 output; point its rounds at this output dtype instead.

    Every `.to(tl.float16)` in that text is one eager kernel boundary, so for a bf16 model the
    same boundary rounds to bf16.  The exhaustive probe evidence behind those spellings was taken
    at fp16 only, which is why the caller has to record which dtype a row ran at.

    THE UNFUSED BASELINE HAS TO GO THROUGH THIS TOO.  Its epilogue is the same text as one
    separate kernel, and if the two round through different dtypes the fused arm is not being
    compared with anything -- it comes out byte-different on every draw and every configuration,
    which reads like a broken kernel and is a broken reference.  This is the one definition;
    nothing else may transcribe it.

    Idempotent: after the substitution there is no `tl.float16` left to substitute again.
    """
    if round_dtype == "fp16":
        return list(lines)
    if round_dtype != "bf16":
        raise ValueError(f"no rounding dtype for {round_dtype!r}")
    return [ln.replace("tl.float16", "tl.bfloat16") for ln in lines]


class FusedPlain:
    """The three compiled kernels for one (epilogue, operand list, output dtype)."""

    def __init__(self, mod, path, n_extra):
        self.pre = mod.fused_pre
        self.gb300 = mod.fused_gb300
        self.gb300_notma = mod.fused_gb300_notma
        self.path = path
        self.n_extra = n_extra


_BUILT: dict[tuple, FusedPlain] = {}


def build(epi_lines, operands, round_dtype, cache_dir, tag="epi"):
    """Generate and import the three kernels for one epilogue.

    `epi_lines` is the epilogue body, already formatted -- the caller has filled any `{scale}` or
    `{lim}` -- and it must end by assigning `tmp_out`.  `operands` is the epilogue's declared
    operand kinds, in the order the text's `tmp_e0`, `tmp_e1`, ... expect them.

    The generated file is left on disk on purpose: it is the thing to read when checking that the
    mainloop really is the shipped one and only the epilogue differs.
    """
    for k in operands:
        if k not in OPERAND_KINDS:
            raise ValueError(f"unknown operand kind {k!r}")
    body_lines = round_lines(epi_lines, round_dtype)
    n = len(operands)
    ptrs, strides = _decl(n)
    src = _HEAD
    src += _PRE_SRC.format(ptrs=ptrs, strides=strides, loads=_loads(n, "ocm[:, None]", "ocn[None, :]", "mask"),
                           body="\n".join("    " + ln for ln in body_lines))
    src += _GB300_SRC.format(ptrs=ptrs, strides=strides, loads=_loads(n, "rm[:, None]", "rn[None, :]", "mask",
                                                                      indent="        "),
                             body="\n".join("        " + ln for ln in body_lines))
    src += _GB300_NOTMA_SRC.format(ptrs=ptrs, strides=strides, loads=_loads(n, "ocm[:, None]", "ocn[None, :]", "mask"),
                                   body="\n".join("    " + ln for ln in body_lines))
    key = (src, )
    hit = _BUILT.get(key)
    if hit is not None:
        return hit
    built = _BUILT[key] = FusedPlain(*_import_src(src, f"fused_plain_{tag}", cache_dir), n)
    return built


def _import_src(src, tag, cache_dir):
    """`triton.jit` reads its function back with `inspect.getsourcelines`, so the source has to
    live in a real file -- an `exec` of a string fails with "should be defined in a Python file".
    Returns (module, path)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{tag}_{hashlib.sha1(src.encode()).hexdigest()[:10]}.py")
    if not os.path.exists(path):
        tmp = path + f".{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(src)
        os.replace(tmp, path)
    name = os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, path


# --------------------------------------------------------------------------------------------
# launchers.  Each body is the corresponding launcher in `kernels.py` with the epilogue's
# operands threaded through and `enable_fp_fusion=False` added; nothing else moves.
# --------------------------------------------------------------------------------------------

_NO_FP_FUSION = False  # see the module docstring: a `.to(fp16)` does not survive without this


def _kcontig(b):
    return b if b.stride(1) == 1 else b.contiguous()


def _extra_args(kinds, extras):
    ptrs = list(extras)
    strides = []
    for k, t in zip(kinds, extras):
        strides += list(operand_strides(k, t))
    return ptrs, strides


def launch_pre(kern, a, b, out_dtype, kinds, extras, cfg, scale=1.0):
    """`kernels._triton_plain`'s pre-sm_103 body, with the epilogue folded in."""
    BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    ptrs, strides = _extra_args(kinds, extras)
    kern.pre[(triton.cdiv(M, BM) * triton.cdiv(N, BN), )](a, b, c, *ptrs, M, N,
                                                          K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                                          c.stride(0), c.stride(1), *strides, scale, BM=BM, BN=BN,
                                                          BK=BK, num_warps=cfg["num_warps"],
                                                          num_stages=cfg["num_stages"], enable_fp_fusion=_NO_FP_FUSION)
    return c


_I32_MAX = 2**31 - 1


def _fits_i32(*ts):
    return all(sum((d - 1) * s for d, s in zip(t.shape, t.stride())) <= _I32_MAX for t in ts)


def _tma_ok(t, block_inner_bytes):
    if t.stride(-1) != 1 or block_inner_bytes % 16:
        return False
    e = t.element_size()
    return all(s * e % 16 == 0 for s in t.stride()[:-1])


@contextlib.contextmanager
def _meta_ws():
    """Compile through Meta's warp-specialization passes, as `kernels._meta_ws` does: upstream
    refuses `warp_specialize=True` on sm_103 for a TMA-fed loop."""
    import triton.knobs
    prev = triton.knobs.nvidia.use_meta_ws
    triton.knobs.nvidia.use_meta_ws = True
    try:
        yield
    finally:
        triton.knobs.nvidia.use_meta_ws = prev


def _num_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def gb300_config(M, N, fp8=False):
    """The tile the shipped GB300 launcher picks for this shape -- its own fitted rule, unchanged.

    `kernels._sm103_config` returns (BM, BN, num_warps, num_stages); BK is the shipped 64 and
    GROUP_M the shipped 8.
    """
    from .kernels import _sm103_config
    BM, BN, nw, ns = _sm103_config(M, N, fp8)
    return {"BM": BM, "BN": BN, "BK": 64, "GROUP_M": 8, "num_warps": nw, "num_stages": ns}


def launch_gb300(kern, a, b, out_dtype, kinds, extras, cfg, scale=1.0, ws=True):
    """`kernels._triton_plain_tma_sm103`'s body, with the epilogue folded in.

    Returns (c, "tma") or (c, "notma"): the shipped launcher falls back the same way when an
    operand cannot carry a tensor descriptor, and the caller records which one ran.
    """
    from triton.tools.tensor_descriptor import TensorDescriptor
    BM, BN, BK, GROUP_M = cfg["BM"], cfg["BN"], cfg["BK"], cfg.get("GROUP_M", 8)
    M, K = a.shape
    N = b.shape[1]
    b_nk = b.stride(0) == 1 and b.stride(1) != 1
    bt = b.t() if b_nk else b
    bb = [BN, BK] if b_nk else [BK, BN]
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    ptrs, strides = _extra_args(kinds, extras)
    if not (_tma_ok(a, BK * a.element_size()) and _tma_ok(bt, bb[1] * bt.element_size())
            and _tma_ok(c, BN * c.element_size())):
        return launch_gb300_notma(kern, a, b, out_dtype, kinds, extras, cfg, scale), "notma"
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(bt, bb)
    c_desc = TensorDescriptor.from_tensor(c, [BM, BN])
    nprog = min(_num_sms(), triton.cdiv(M, BM) * triton.cdiv(N, BN))
    with _meta_ws():
        kern.gb300[(nprog, )](a_desc, b_desc, c_desc, *ptrs, M, N, K, *strides, scale, B_NK=b_nk, GROUP_M=GROUP_M,
                              NUM_SMS=nprog, WS=ws, BM=BM, BN=BN, BK=BK, num_warps=cfg["num_warps"],
                              num_stages=cfg["num_stages"], enable_fp_fusion=_NO_FP_FUSION)
    return c, "tma"


def launch_gb300_notma(kern, a, b, out_dtype, kinds, extras, cfg, scale=1.0):
    """`kernels._triton_plain_sm103`'s body, with the epilogue folded in."""
    BM, BN, BK, GROUP_M = cfg["BM"], cfg["BN"], cfg["BK"], cfg.get("GROUP_M", 8)
    if b.stride(0) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    ptrs, strides = _extra_args(kinds, extras)
    kern.gb300_notma[(triton.cdiv(M, BM) * triton.cdiv(N, BN), )](a, b, c, *ptrs, M, N, K, a.stride(0), a.stride(1),
                                                                  b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                                                                  *strides, scale, GROUP_M=GROUP_M,
                                                                  EVEN_K=(K % BK == 0), I64=not _fits_i32(a, b, c),
                                                                  BM=BM, BN=BN, BK=BK, num_warps=cfg["num_warps"],
                                                                  num_stages=cfg["num_stages"],
                                                                  enable_fp_fusion=_NO_FP_FUSION)
    return c


def fit_gb300(kern, a, b, out_dtype, kinds, extras, cfg):
    """The first configuration at or below the shipped one that this kernel actually fits into.

    The shipped GB300 tile was fitted for a kernel with no epilogue and with `SUBTILE` on, which
    halves the store staging buffer.  Fusing an epilogue takes `SUBTILE` away and adds the extra
    operand's registers, so at the deepest rungs of the shipped ladder the kernel asks for more
    shared memory than an SM has -- measured on a 1024x2048 MoE down projection, 233908 bytes
    against a 232448 limit, 0.6% over.  Rather than dropping the arm, step the pipeline down one
    stage at a time and then halve the tile, and return the configuration that ran so the row can
    record it.

    A tile and a pipeline depth cannot move a bit -- the k loop is the same one either way -- and
    the caller byte-checks the result regardless, so this trades speed for the arm existing.
    Raises the last error if nothing in the ladder fits.
    """
    last = None
    for trial in _gb300_ladder(cfg):
        try:
            c, how = launch_gb300(kern, a, b, out_dtype, kinds, extras, trial)
            torch.cuda.synchronize()
            return trial, how, c
        except Exception as e:  # OutOfResources, and anything else the tile could cause
            last = e
            torch.cuda.empty_cache()
    raise last if last is not None else RuntimeError("no GB300 configuration was tried")


def _gb300_ladder(cfg):
    """The shipped configuration first, then shallower, then smaller.  Best first, so a shape
    that fits at the shipped point pays nothing for this.

    Whether a stage or half the tile width is the cheaper thing to give up is a guess, and it is
    written here rather than hidden: pipeline depth goes first, all the way to 2, before the tile
    is halved.  Every row records the configuration it ended on.
    """
    seen = []
    for bn in (cfg["BN"], cfg["BN"] // 2, cfg["BN"] // 4):
        if bn < 32:
            continue
        for ns in range(cfg["num_stages"], 1, -1):
            t = dict(cfg, BN=bn, num_stages=ns)
            if t not in seen:
                seen.append(t)
    return seen


# --------------------------------------------------------------------------------------------
# the search space for the pre-sm_103 arm
# --------------------------------------------------------------------------------------------
# Deliberately the same space `steps/gemm_perf_random.config_space` uses for mode `plain`, so the
# unfused and the fused measurements of "what a search buys" are searches of the same size.  Only
# launch parameters are in it: the plan fixes the arithmetic and none of the plan is here.
_POW2 = (16, 32, 64, 128, 256)

DEFAULT_PRE_CONFIG = {"BM": 128, "BN": 128, "BK": 64, "num_warps": 8, "num_stages": 3}


def _np2(x):
    return 1 << max(0, int(x - 1)).bit_length()


def _smem_ok(BM, BN, BK, esz, stages):
    return (BM * BK + BK * BN) * esz * max(1, stages) <= 220000


def config_space(M, N, K, kind, extra_bytes_per_element=2):
    """Configurations worth trying for the pre-sm_103 fused kernel, best-first.

    The shipped launch is first, so a search that runs out of budget can never come out worse
    than no search at all.  After it, tiles that fill the machine, then the wider k step, then
    more warps, then more stages -- the order is a heuristic and it is written down here so that
    a truncated search can be read as "the front of THIS order".
    """
    import math
    esz = 1 if kind == "fp8" else 2
    min_bm = 64 if kind == "fp8" else 16
    bms = [b for b in _POW2 if min_bm <= b <= max(min_bm, 2 * _np2(M))] or [min_bm]
    bns = [b for b in _POW2 if b <= max(16, 2 * _np2(N))] or [16]
    out = []
    for BM in bms:
        for BN in bns:
            for BK in (16, 32, 64, 128):
                for nw in (4, 8):
                    for ns in (2, 3, 4):
                        if not _smem_ok(BM, BN, BK, esz, ns) or BM * BN < 32 * nw:
                            continue
                        out.append({"BM": BM, "BN": BN, "BK": BK, "num_warps": nw, "num_stages": ns})

    def order(c):
        tiles = triton.cdiv(M, c["BM"]) * triton.cdiv(N, c["BN"])
        return (abs(math.log(max(tiles, 1) / 148.0)), -c["BK"], -c["num_warps"], -c["num_stages"])

    d = DEFAULT_PRE_CONFIG
    return [d] + sorted((c for c in out if c != d), key=order)


def config_str(cfg):
    return " ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
