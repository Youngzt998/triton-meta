"""The kernels the `inner_tree.*` steps measure, and the code that compiles, runs and times them.

Read this before either step. `steps/inner_tree_layout.py` is about the *pass*: resolve it,
inject it, compare the arms, print the table. This file is about the *kernels*: what they are,
where they came from, how big they are, and what one measured row is made of. Nothing here knows
the pass exists.

TWO STEPS DEPEND ON THIS FILE. `inner_tree.layout` measures the 24 (kernel, dtype) pairs in
`catalogue()`; `inner_tree.bitmatch` measures the same 24 plus two order-invariant control pairs
of its own. So a change to the catalogue -- a kernel added or removed, a shape or a dtype
altered -- or to the input draws, the launch path or `bench_ms` **moves both steps' numbers**,
and neither step's committed table can be compared with a table taken after such a change. If
you edit this file, re-run both.

WHY THIS FILE EXISTS
--------------------
The kernels live in three different files, written at three different times, with three
different harness shapes -- a `KernelSpec` dataclass, a tuple-returning `_spec_*` builder, and a
`register()` benchmark record. An earlier version of this step drove one of those three shapes
and reported the other two as "not reachable", which is a hole a reader cannot see past. So the
kernel *definitions* are imported from where they live -- they are not retyped, and a reviewer
can diff them against their source -- while every harness detail (input data, launch, timing,
byte compare) is written once, here, and applies to all seventeen the same way.

THE THREE SUITES

  1. `eval_kernels`             synthetic reduction micro-kernels, written for this project to
                                isolate the layout question. Small tiles, a wide-dynamic-range
                                input so the reduction order decides the bits, and a dtype axis.
  2. `realistic_inductor_kernels`  verbatim TorchInductor output. Every body in that file is a
                                Triton kernel `torch.compile` actually emitted; the shapes are
                                the shapes the model ran at. These answer the question the
                                synthetic ones cannot: what the ordering constraint costs on a
                                kernel a user would really get.
  3. `benchmark_kernels`        the benchmark zoo; one weight-gradient reduction from the
                                LayerNorm tutorial.

WHAT IS DELIBERATELY LEFT OUT, AND WHY
--------------------------------------
`OUT_OF_SUITE` below names every reduction in those files this step does *not* measure, with the
reason. Two of the eight TorchInductor GROUP 1 kernels have no `reduction_ordering` parameter at
all -- a Welford combine and a scan are not ordered add-reductions, so there is nothing for the
pass to be bit-safe about. GROUP 2 and later of that file are an inspect-only reference corpus:
hard-coded shapes and references to Inductor runtime helpers, not runnable as they stand.

A kernel the pass declines to touch is still measured and still gets a row. "The pass correctly
did nothing here" is a result; a kernel missing from the table is a hole.

DTYPES
------
The synthetic suite carries a dtype axis and is measured at f16 and fp8, the two this project
reports (`col_bf16` pins bf16 in its own body, so the axis does not reach it). The TorchInductor
and zoo kernels are measured at f32: their bodies are verbatim generated code that computes in
f32, and narrowing them would mean editing the body -- which would cost exactly the property
that makes them worth having. It also makes them the only rows that are like-for-like with the
earlier H100 measurement, which was f32 throughout.

THE TWO TRAPS THIS FILE IS BUILT AROUND
---------------------------------------
* **The operand dtype must match the compile.** A benching helper in the path this step
  replaced built f32 operands for a kernel compiled at the configuration's dtype. That hands an
  f32 buffer to a kernel expecting f16 pointers: it reads a fraction of the bytes and times a
  kernel that is not the one under test, on every f16 and fp8 row, silently. `Build` carries the
  dtype it was compiled at and every launch asserts each operand matches it.
* **Output buffers are zeroed before every launch.** Every kernel here covers its whole output,
  so zeroing hides nothing; but a recycled allocator buffer can differ in bytes no kernel wrote,
  and that reads as BITS CHANGED -- the one verdict this step must never produce by accident.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

DEVICE = "cuda"

# Reductions in the three source files this step does NOT measure, and why. Printed in the
# report: a kernel that is out of scope is a result, a kernel that is absent is a hole.
OUT_OF_SUITE = (
    ("B_layernorm_welford_gather", "realistic_inductor_kernels.py GROUP 1",
     "no reduction_ordering parameter: a Welford combine is not an ordered add-reduction, so "
     "inner_tree does not apply and the pass has nothing to preserve"),
    ("F_cumsum_scan", "realistic_inductor_kernels.py GROUP 1",
     "no reduction_ordering parameter: tl.associative_scan is a scan, not a tt.reduce, and the "
     "pass walks tt.reduce only"),
    ("GROUP 2 and later", "realistic_inductor_kernels.py",
     "inspect-only reference corpus -- hard-coded shapes and torch._inductor runtime helpers, "
     "not runnable as they stand; that file's own docstring says so"),
)


# --------------------------------------------------------------------------------------------
# Inputs. Two kinds, and the difference between them is the whole experiment.
# --------------------------------------------------------------------------------------------
def _fp8(torch):
    return getattr(torch, "float8_e4m3fn", None)


def _clip_logspace(torch, lo, hi, dt):
    """Narrow the exponent spread for the dtypes that cannot hold it.

    f16 tops out near 65504 and its smallest normal is about 6e-5, so a twelve-decade spread
    overflows to inf and the reduction returns inf whatever the order -- which reads as "the bits
    never moved" and is a false pass. fp8 e4m3 tops out at 448 and has no inf at all, so anything
    above it becomes NaN. bf16 shares f32's exponent range and is left alone.

    Same numbers as `eval_kernels._clip_logspace`: a different spread would make these rows
    non-comparable with the earlier measurement for no gain.
    """
    if dt is torch.float16:
        return max(lo, -3), min(hi, 3)
    if _fp8(torch) is not None and dt is _fp8(torch):
        return max(lo, -2), min(hi, 2)
    return lo, hi


def torch_dtype(torch, name):
    table = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    if _fp8(torch) is not None:
        table["fp8"] = _fp8(torch)
    if name not in table:
        raise ValueError(f"dtype {name!r} is not available in this build of torch")
    return table[name]


def adv_nd(torch, numel, seed, dt):
    """Order-sensitive data: wide exponent spread, alternating signs.

    Adding these in a different order gives different bits, which is the only reason a
    `bit_changed = 0` on such a row means anything. On narrow, unit-scale data almost every
    regrouping rounds to the same answer and the check passes things it should not.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(numel, generator=g, dtype=torch.float32)
    lo, hi = _clip_logspace(torch, -6, 6, dt)
    scale = torch.logspace(lo, hi, numel, dtype=torch.float32)
    signs = torch.where(torch.arange(numel) % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0))
    return (base * scale * signs).to(DEVICE, dt)


def plain_nd(torch, numel, seed, dt):
    """Unit-scale data. Used for the timing arm, where only speed matters, and for the one kernel
    whose math (`exp`) overflows on the wide-range draw."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(numel, generator=g, dtype=torch.float32).to(DEVICE, dt)


def to_bytes(t):
    """The equality unit: the exact output bits."""
    return t.detach().cpu().contiguous().numpy().tobytes()


# --------------------------------------------------------------------------------------------
# One kernel, and one built launch of it.
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Launch:
    """Everything one launch needs: the runtime arguments, the output tensors whose bytes are the
    answer, the constexpr kwargs, and the grid."""
    args: list
    outs: list
    consts: dict
    grid: int

    def zero_outputs(self):
        for o in self.outs:
            o.zero_()


@dataclass(frozen=True)
class Kernel:
    name: str
    suite: str  # which of the three source files
    source: str  # file and symbol, so a reviewer can go read the body
    what: str  # one line: what it computes and why it is in the table
    jit: object  # the @triton.jit function, imported not retyped
    build: object  # (torch, dtype_name, seed, wide, size) -> Launch
    dtypes: tuple  # dtype names measured for this kernel
    prior_dtype: str = "f32"  # what the earlier H100 measurement ran this kernel at
    tags: tuple = field(default_factory=tuple)


# --------------------------------------------------------------------------------------------
# Suite 1 -- the synthetic reduction micro-kernels.
# --------------------------------------------------------------------------------------------
# Sizes come from `eval_kernels._layout_spec`: 64 programs for the bit check, 2048 for the
# timing. The tile shape is a constexpr, so one compile serves both; only the grid changes.
_SYNTH_GRID = {"precision": 64, "perf": 2048}


def _never(_dtype):
    return False


def _fp8_only(dtype):
    return dtype == "fp8"


def _not_f32(dtype):
    return dtype != "f32"


def _synth_build(consts, tile_numel, out_numel, n_inputs, pinned_dtype, plain_precision, widen):
    """A launch builder for the `eval_kernels` shape: N input buffers, one f32 output, the tile
    dimensions as constexprs, `grid` programs each handling one tile."""

    def build(torch, dtype_name, seed, wide, size):
        grid = _SYNTH_GRID[size]
        name = pinned_dtype or dtype_name
        dt = torch_dtype(torch, name)
        draw = plain_nd if (not wide or plain_precision) else adv_nd
        ins = [draw(torch, grid * tile_numel, seed + i, dt) for i in range(n_inputs)]
        out = torch.zeros(grid * out_numel, device=DEVICE, dtype=torch.float32)
        extra = dict(consts)
        if widen(name):
            # fp8 e4m3 has no fadd/fmul lowering at all and tl.exp only accepts f32, so those
            # kernels widen the loaded tile first. The flag is passed only when it is True, so
            # every other row compiles the body it always did.
            extra["CAST_F32"] = True
        return Launch(args=[*ins, out], outs=[out], consts=extra, grid=grid)

    return build


# --------------------------------------------------------------------------------------------
# Suite 2 -- the TorchInductor kernels.
# --------------------------------------------------------------------------------------------
def _inductor_build(spec_name):
    """A launch builder that defers to `realistic_inductor_kernels.SPECS[spec_name]`.

    Those builders are what the earlier measurement used, and they carry the shapes the model
    actually ran at (`XBLOCK`, `R0_BLOCK`, `xnumel`, `r0_numel`, and the grid that follows). They
    are called rather than retyped so the shapes cannot drift from the corpus. They return
    `(jit, pointer_args, outs, scalars, constexprs, grid)`; the runtime argument list is the
    pointers followed by the scalars, which is what both warmup and launch must be handed.

    `size` is ignored: an Inductor kernel's grid is fixed by the tensor it was generated for, so
    the bit check and the timing run the same launch. Its data is always the order-sensitive
    draw, again matching the earlier measurement.
    """

    def build(torch, dtype_name, seed, wide, size):
        del torch, size, wide
        if dtype_name != "f32":
            raise ValueError(f"{spec_name} is verbatim generated f32 code; {dtype_name} would need the "
                             "body edited, which is the one thing that must not happen to it")
        from bitequiv.evaluation.realistic_inductor_kernels import SPECS
        _fn, args, outs, scalars, consts, grid = SPECS[spec_name](None, seed)
        return Launch(args=[*args, *scalars], outs=list(outs), consts=dict(consts), grid=grid)

    return build


# --------------------------------------------------------------------------------------------
# Suite 3 -- the benchmark zoo weight-gradient reduction.
# --------------------------------------------------------------------------------------------
_DWDB_M, _DWDB_N, _DWDB_BLOCK_N = 64, 2048, 256


def _dwdb_build(torch, dtype_name, seed, wide, size):
    """`layernorm_bwd_dwdb`: two column sums over the 64 partial rows the LayerNorm backward
    leaves behind, one for dw and one for db. Shape and block size are the zoo's own
    (`benchmark_kernels.register(name="layernorm_bwd_dwdb", ...)`); the grid follows from them,
    so the bit check and the timing run the same launch."""
    del size
    dt = torch_dtype(torch, dtype_name)
    draw = adv_nd if wide else plain_nd
    dwp = draw(torch, _DWDB_M * _DWDB_N, 112 + seed, dt).view(_DWDB_M, _DWDB_N)
    dbp = draw(torch, _DWDB_M * _DWDB_N, 113 + seed, dt).view(_DWDB_M, _DWDB_N)
    dw = torch.zeros(_DWDB_N, device=DEVICE, dtype=torch.float32)
    db = torch.zeros(_DWDB_N, device=DEVICE, dtype=torch.float32)
    grid = (_DWDB_N + _DWDB_BLOCK_N - 1) // _DWDB_BLOCK_N
    return Launch(args=[dwp, dbp, dw, db, _DWDB_M, _DWDB_N], outs=[dw, db],
                  consts=dict(BLOCK_M=_DWDB_M, BLOCK_N=_DWDB_BLOCK_N), grid=grid)


# --------------------------------------------------------------------------------------------
# The catalogue.
# --------------------------------------------------------------------------------------------
def catalogue():
    """The seventeen kernels, in report order. Imports happen here, not at module import, so
    `artifact.py --list` still works on a machine with no GPU and no torch."""
    from bitequiv.evaluation import eval_kernels as ek
    from bitequiv.evaluation import realistic_inductor_kernels as rk
    from bitequiv.evaluation.benchmark_kernels import _norm_bwd_dwdb_kernel

    synth = ("f16", "fp8")
    ek_src = "bitequiv/evaluation/eval_kernels.py"
    rk_src = "bitequiv/evaluation/realistic_inductor_kernels.py"
    bk_src = "bitequiv/evaluation/benchmark_kernels.py"

    def synthetic(name, jit, symbol, what, consts, tile, out_numel, n_inputs=1, pinned=None, plain_precision=False,
                  widen=_fp8_only, dtypes=synth, prior_dtype="f32", tags=()):
        return Kernel(name=name, suite="synthetic", source=f"{ek_src} :: {symbol}", what=what, jit=jit,
                      build=_synth_build(consts, tile, out_numel, n_inputs, pinned, plain_precision,
                                         widen), dtypes=dtypes, prior_dtype=prior_dtype, tags=tags)

    kernels = [
        synthetic("sum_3d_outer", ek._k_sum3d_outer, "_k_sum3d_outer",
                  "[B=64,M=8,N=16] sum over the OUTER axis: the most strided reduce axis in the table",
                  dict(B=64, M=8, N=16), 64 * 8 * 16, 8 * 16),
        synthetic("sum_2d_col", ek._k_sum2d_axis0,
                  "_k_sum2d_axis0", "[M=256,C=32] column sum over axis 0: reduce the non-contiguous axis",
                  dict(M=256, C=32), 256 * 32, 32),
        synthetic("sum_2d_col_big", ek._k_sum2d_axis0,
                  "_k_sum2d_axis0", "[M=1024,C=32] the same at four times the reduce extent, a 32K-element tile",
                  dict(M=1024, C=32), 1024 * 32, 32),
        synthetic("sum_2d_axis0", ek._k_sum2d_axis0, "_k_sum2d_axis0", "[M=128,C=32] the same at a quarter of it",
                  dict(M=128, C=32), 128 * 32, 32),
        synthetic("col_exp_sum", ek._k_col_exp_sum, "_k_col_exp_sum",
                  "[M=256,C=32] sum(exp(x)) over axis 0: exp widens the leaves to f32 whatever the input was",
                  dict(M=256, C=32), 256 * 32, 32, plain_precision=True, widen=_not_f32, tags=("fp8-sensitive", )),
        synthetic("col_bf16", ek._k_col_bf16, "_k_col_bf16",
                  "[M=256,C=32] bf16 input, f32 accumulate: the one kernel whose dtype is pinned in its own body",
                  dict(M=256, C=32), 256 * 32, 32, pinned="bf16", widen=_never, dtypes=("bf16", ), prior_dtype="bf16"),
        synthetic("col_sum_loop", ek._k_col_sum_loop, "_k_col_sum_loop",
                  "[M=512,C=32] column sum as a for-loop over 128-row chunks: loop structure around the reduce",
                  dict(M=512, C=32, CHUNK=128), 512 * 32, 32),
        synthetic("col_dot", ek._k_col_dot, "_k_col_dot",
                  "[M=256,C=32] sum(a*b) over axis 0: a mul-fed reduce, which the pass declines on purpose",
                  dict(M=256, C=32), 256 * 32, 32, n_inputs=2, tags=("mul-fed", "fp8-sensitive")),
    ]

    inductor = [
        ("A_rms_norm_fwd", "A_rms_norm_fwd",
         "RMS-norm forward: one sum of x*x over the contiguous 256-wide feature axis, then rsqrt and affine"),
        ("C_rms_norm_bwd_2reduce", "C_rms_norm_bwd_2reduce",
         "RMS-norm backward: two sums where the second one's input depends on the first one's result"),
        ("D_masked_global_sum", "D_masked_global_sum",
         "MSE-loss tail: 608 live elements zero-padded into a 1024 tile, summed down to one scalar"),
        ("E_triu_masked_rowsum", "E_triu_masked_rowsum", "masked per-row sum at the odd extent 235, padded into 256"),
        ("G_plain_sum_looped", "G_plain_sum_looped",
         "looped sum: a running accumulate across 128-wide chunks, then one tree-reduce at the end"),
        ("H_mean_permute", "H_mean_permute", "mean over a permuted, contiguous 256-wide axis"),
        ("I_bias_grad_dim0", "RED_colsum_dim0",
         "dbias = grad.sum(dim=0), [512,256] -> [256]: the reduce axis is strided by xnumel while the kept "
         "axis is contiguous, which is the shape the pass exists for"),
        ("J_epilogue_colsum_dim0", "RED_colsum_dim0",
         "matmul-epilogue column sum, [512,64] -> [64]: the same shape with a narrow kept axis, one tile"),
    ]
    for name, symbol, what in inductor:
        kernels.append(
            Kernel(name=name, suite="inductor", source=f"{rk_src} :: {symbol}", what=what, jit=getattr(rk, symbol),
                   build=_inductor_build(name), dtypes=("f32", ), prior_dtype="f32", tags=("real-inductor-output", )))

    kernels.append(
        Kernel(
            name="layernorm_bwd_dwdb", suite="zoo", source=f"{bk_src} :: _norm_bwd_dwdb_kernel",
            what="LayerNorm backward weight gradient: two column sums over the 64 partial rows, "
            "[64,2048] -> two [2048] vectors", jit=_norm_bwd_dwdb_kernel, build=_dwdb_build, dtypes=("f32", ),
            prior_dtype="f32", tags=("two-outputs", )))
    return kernels


def jit_functions(kernels):
    """The distinct JITFunction objects behind the catalogue -- what has to be cache-cleared.

    Clearing exactly the functions in use, rather than sweeping a module's globals, is what makes
    the cache defeat complete by construction: a kernel that is measured is a kernel that is
    cleared, with no second list to keep in step.
    """
    seen, out = set(), []
    for k in kernels:
        if id(k.jit) not in seen:
            seen.add(id(k.jit))
            out.append(k.jit)
    return out


def clear_caches(jits):
    """Drop the in-memory compile cache of each JITFunction.

    This is not belt-and-braces, it is the experiment. Triton's in-memory kernel cache is keyed
    on the specialization and the launch options only (`jit.py`, `compute_cache_key`), and an
    injected pass is part of neither, so the second build of a configuration is served the first
    build's kernel. Without this the pass-on arm silently reuses the pass-off one and the whole
    table comes back a perfect, meaningless 1.00x with 0 bits changed. `TRITON_ALWAYS_COMPILE`
    covers the on-disk cache and does nothing for this one.
    """
    for fn in jits:
        for attr in ("device_caches", "cache"):
            cache = getattr(fn, attr, None)
            if hasattr(cache, "clear"):
                try:
                    cache.clear()
                except Exception:  # noqa: BLE001
                    pass


# --------------------------------------------------------------------------------------------
# Launch reuse. Inputs depend on (kernel, dtype, size, seed) and NOT on the configuration, so
# building them once per kernel instead of once per configuration removes most of the host work
# from the sweep -- the largest draw here is 67 million elements. Outputs are zeroed before every
# launch, so a reused buffer cannot carry an answer over from the previous arm.
# --------------------------------------------------------------------------------------------
_LAUNCHES: dict = {}


def reset_launches():
    """Drop the cached launches. Called when the sweep moves to a new (kernel, dtype) so the
    device memory of the previous one is released rather than accumulated."""
    _LAUNCHES.clear()


def get_launch(kernel, dtype, seed, wide, size):
    import torch

    key = (kernel.name, dtype, seed, wide, size)
    launch = _LAUNCHES.get(key)
    if launch is None:
        launch = kernel.build(torch, dtype, seed, wide, size)
        _LAUNCHES[key] = launch
    return launch


# --------------------------------------------------------------------------------------------
# Compile, run, time.
# --------------------------------------------------------------------------------------------
@dataclass
class Build:
    """A compiled kernel plus the two things a caller must not get wrong about it: which dtype
    its pointers expect, and which ordering it was compiled with."""
    kernel: Kernel
    ck: object
    dtype: str
    ordering: str
    config: object

    @property
    def ttgir(self):
        asm = getattr(self.ck, "asm", None) or {}
        return asm.get("ttgir") or ""


def _ordering_enum(tl, name):
    return {"unordered": tl.ReductionOrdering.UNORDERED, "inner_tree": tl.ReductionOrdering.INNER_TREE}[name]


def compile_kernel(kernel, dtype, ordering, config, size="precision"):
    """Compile one (kernel, dtype, ordering, configuration). Does not launch anything.

    The caller clears the caches and installs the pipeline variant first; this only builds.
    `warmup` does not run the kernel, so only the argument *types* matter here, which is why the
    cheap precision-sized launch is used even for a build that will later be timed at perf size.
    """
    import triton.language as tl

    launch = get_launch(kernel, dtype, 0, False, size)
    ck = kernel.jit.warmup(*launch.args, grid=(launch.grid, ), ORD=_ordering_enum(tl,
                                                                                  ordering), num_warps=config.num_warps,
                           num_stages=config.num_stages, enable_fp_fusion=config.enable_fp_fusion, **launch.consts)
    return Build(kernel=kernel, ck=ck, dtype=dtype, ordering=ordering, config=config)


def _assert_operand_dtype(build, launch):
    """The trap this replaces: hand a kernel compiled for f16 pointers an f32 buffer and it reads
    a fraction of the bytes and times something else, with no error. Checked on every launch, not
    only in the timing path -- a bit row measured on the wrong operands is just as wrong."""
    import torch

    want = torch_dtype(torch, build.dtype)
    out_ids = {id(o) for o in launch.outs}
    for a in launch.args:
        if not torch.is_tensor(a) or id(a) in out_ids:
            continue
        if a.dtype is not want:
            raise AssertionError(f"{build.kernel.name}: an operand is {a.dtype} but the kernel was compiled for "
                                 f"{want}; the kernel measured would not be the one under test")


def run_bytes(build, seed, size="precision"):
    """Launch on the order-sensitive draw and return the exact output bytes."""
    import torch

    launch = get_launch(build.kernel, build.dtype, seed, True, size)
    _assert_operand_dtype(build, launch)
    launch.zero_outputs()
    build.ck[(launch.grid, 1, 1)](*launch.args)
    torch.cuda.synchronize()
    return b"".join(to_bytes(o) for o in launch.outs)


def bench_ms(build, reps=3, warmup=50, rep=200):
    """Device time, min of `reps` `do_bench` medians -- the earlier measurement's protocol, kept
    so the numbers are comparable to it. Returns (ms, output_bytes).

    The operands are built outside the timed thunk, at the dtype the kernel was compiled for. The
    bytes come back so the timing arm doubles as a second, independent bit check on a larger and
    differently shaped input than the correctness arm used.
    """
    import torch
    from triton.testing import do_bench

    launch = get_launch(build.kernel, build.dtype, 0, False, "perf")
    _assert_operand_dtype(build, launch)
    launch.zero_outputs()
    ck, grid, args = build.ck, launch.grid, launch.args

    def thunk():
        ck[(grid, 1, 1)](*args)

    thunk()
    torch.cuda.synchronize()
    ms = min(do_bench(thunk, warmup=warmup, rep=rep) for _ in range(reps))
    torch.cuda.synchronize()
    return ms, b"".join(to_bytes(o) for o in launch.outs)


def nan_fraction(raw):
    """How much of the output is NaN or Inf. A saturated output compares equal to itself, so a
    high value makes `bit_changed = 0` say nothing."""
    import numpy as np

    a = np.frombuffer(raw, dtype=np.float32)
    return float((~np.isfinite(a)).mean()) if a.size else 0.0


# --------------------------------------------------------------------------------------------
# The fp8 order-invariance probe.
# --------------------------------------------------------------------------------------------
def fp8_order_invariance_probe(draws=200, leaves=256):
    """Measure, rather than assert, why the fp8 pure-sum rows are no evidence.

    An fp8 e4m3 value carries a four-bit significand. The exact sum of a few hundred of them
    still fits inside an f32 mantissa, so every association order gives the same answer -- the
    reduction is order-invariant by arithmetic, not because the pass preserved anything. A
    `bit_changed = 0` on such a row is not weak evidence, it is none, and the report says so
    rather than counting it as a pass.

    Runs on the CPU in a second or two on the same draw the kernels use, and reports, for fp8 and
    for f16 as the contrast: how many draws sum exactly in f32, and on how many a sequential sum
    differs from a pairwise tree. Returns {dtype: (exact, differ, draws)}.
    """
    import torch

    out = {}
    for name in ("fp8", "f16"):
        dt = _fp8(torch) if name == "fp8" else torch.float16
        if dt is None:
            continue
        exact = differ = 0
        lo, hi = _clip_logspace(torch, -6, 6, dt)
        scale = torch.logspace(lo, hi, leaves, dtype=torch.float32)
        sign = torch.where(torch.arange(leaves) % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0))
        for seed in range(draws):
            g = torch.Generator().manual_seed(seed)
            base = torch.randn(leaves, generator=g, dtype=torch.float32)
            leaf32 = (base * scale * sign).to(dt).to(torch.float32)
            seq = torch.zeros((), dtype=torch.float32)
            for v in leaf32:
                seq = seq + v
            tree = leaf32.clone()
            while tree.numel() > 1:
                tree = tree[0::2] + tree[1::2]
            if float(seq) == float(leaf32.to(torch.float64).sum()):
                exact += 1
            if float(seq) != float(tree[0]):
                differ += 1
        out[name] = (exact, differ, draws)
    return out


def median(xs):
    return statistics.median(xs) if xs else None
