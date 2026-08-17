"""A bit-exact GEMM with the epilogue folded in, for the `mode=plain` shapes on sm_103.

The bit contract copied from `bitequiv/cublas_match/kernels.py::_plain_gemm_tma_sm103`: BK real
k-elements per `tl.dot`, in increasing k, into one fp32 accumulator, scaled once and rounded
once.  Only the store is different -- the accumulator is rounded to the GEMM's output dtype
first (that is where the unfused path rounds), read back as fp32, and the epilogue is applied to
that.  So the fused result is the unfused result by construction, and the bit gate in `run.py`
checks it against cuBLAS + a separate Triton epilogue kernel on real inputs anyway.

Everything here is a copy, not an edit: `bitequiv/` is untouched.
"""
from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl

from bitequiv.cublas_match.kernels import _num_sms, _sm103_config, _tma_ok

# Epilogue codes. Every one is something cuBLASLt cannot express -- its whole list is bias,
# relu, gelu and their backward / bias-gradient variants.
EPI_NONE = 0  # store the GEMM result; the control arm
EPI_SILU = 1  # x * sigmoid(x)
EPI_GATE = 2  # x * sigmoid(g),  g an extra M x N tensor       (two-tensor gate)
EPI_SILU_Q = 3  # (x * sigmoid(x)) * scale, stored as fp8       (the output SHRINKS)
EPI_CHAIN = 4  # t = (x + r) * scale; t * sigmoid(t)            (four ops, one extra tensor)
EPI_GATE_Q = 5  # (x * sigmoid(g)) * scale, stored as fp8       (extra read AND smaller output)

EPI_NAMES = {
    EPI_NONE: "none", EPI_SILU: "silu", EPI_GATE: "gate", EPI_SILU_Q: "silu_fp8", EPI_CHAIN: "chain4",
    EPI_GATE_Q: "gate_fp8"
}
# Which epilogues read a second M x N tensor, and which shrink the output.
EPI_NEEDS_G = {EPI_GATE, EPI_GATE_Q}
EPI_NEEDS_R = {EPI_CHAIN}
EPI_FP8_OUT = {EPI_SILU_Q, EPI_GATE_Q}


@triton.jit
def apply_epi(x, g, r, s, EPI: tl.constexpr):
    """The epilogue itself, in fp32. Used by BOTH the fused kernel and the standalone one, so the
    two arms cannot drift in the arithmetic -- only in where the value came from."""
    if EPI == 0:
        y = x
    elif EPI == 1:
        y = x * tl.sigmoid(x)
    elif EPI == 2:
        y = x * tl.sigmoid(g)
    elif EPI == 3:
        y = (x * tl.sigmoid(x)) * s
    elif EPI == 4:
        t = (x + r) * s
        y = t * tl.sigmoid(t)
    else:
        y = (x * tl.sigmoid(g)) * s
    return y


# --------------------------------------------------------------------------------------------
# arm 1's second kernel: the epilogue on its own, reading the GEMM's output back from HBM.
# One kernel even for the four-op chain, which is what Inductor produces.
# --------------------------------------------------------------------------------------------


@triton.jit
def epi_kernel(C, G, R, O, NEL, s, EPI: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < NEL
    x = tl.load(C + off, mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + off, mask=m, other=0.0).to(tl.float32) if EPI == 2 or EPI == 5 else x
    r = tl.load(R + off, mask=m, other=0.0).to(tl.float32) if EPI == 4 else x
    tl.store(O + off, apply_epi(x, g, r, s, EPI).to(O.dtype.element_ty), mask=m)


def run_epi(c, g, r, scale, epi, out=None, cfg=(4096, 8)):
    """The standalone epilogue. `out` lets the caller reuse a buffer so the timed call allocates
    nothing -- an allocation inside a CUDA graph capture is not what we want to measure."""
    nel = c.numel()
    dt = torch.float8_e4m3fn if epi in EPI_FP8_OUT else c.dtype
    if out is None:
        out = torch.empty_like(c, dtype=dt)
    BLOCK, nw = cfg
    gg = g if g is not None else c
    rr = r if r is not None else c
    epi_kernel[(triton.cdiv(nel, BLOCK), )](c, gg, rr, out, nel, scale, EPI=epi, BLOCK=BLOCK, num_warps=nw)
    return out


# --------------------------------------------------------------------------------------------
# arm 2: the bit-exact GEMM with the epilogue folded in.
# --------------------------------------------------------------------------------------------


@triton.jit
def _plain_gemm_tma_fused(a_desc, b_desc, c_desc, g_desc, r_desc, M, N, K, s, es, B_NK: tl.constexpr,
                          GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, SUBTILE: tl.constexpr, WS: tl.constexpr,
                          FLATTEN: tl.constexpr, EPI: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                          BK: tl.constexpr, ODT: tl.constexpr):
    """`_plain_gemm_tma_sm103` with the epilogue folded in.

    The k loop is copied character for character. The only change is between the accumulator and
    the store: `acc.to(ODT)` is the GEMM's own rounding -- exactly the bytes the unfused path
    would have written to HBM -- and the epilogue then runs on that, in fp32, the same way the
    standalone kernel runs it on the bytes it reads back."""
    start_pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    num_tiles = npm * npn
    per_group = GROUP_M * npn
    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WS):
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

        tile_id_c += NUM_SMS
        gidc = tile_id_c // per_group
        firstc = gidc * GROUP_M
        rowsc = tl.minimum(npm - firstc, GROUP_M)
        omc = (firstc + (tile_id_c % per_group) % rowsc) * BM
        onc = ((tile_id_c % per_group) // rowsc) * BN
        if SUBTILE:
            half = tl.permute(tl.reshape(acc, (BM, 2, BN // 2)), (0, 2, 1))
            acc0, acc1 = tl.split(half)
            n1 = onc + BN // 2
            x0 = acc0.to(ODT).to(tl.float32)
            g0 = g_desc.load([omc, onc]).to(tl.float32) if EPI == 2 or EPI == 5 else x0
            r0 = r_desc.load([omc, onc]).to(tl.float32) if EPI == 4 else x0
            c_desc.store([omc, onc], apply_epi(x0, g0, r0, es, EPI).to(c_desc.dtype))
            x1 = acc1.to(ODT).to(tl.float32)
            g1 = g_desc.load([omc, n1]).to(tl.float32) if EPI == 2 or EPI == 5 else x1
            r1 = r_desc.load([omc, n1]).to(tl.float32) if EPI == 4 else x1
            c_desc.store([omc, n1], apply_epi(x1, g1, r1, es, EPI).to(c_desc.dtype))
        else:
            x = acc.to(ODT).to(tl.float32)
            g = g_desc.load([omc, onc]).to(tl.float32) if EPI == 2 or EPI == 5 else x
            r = r_desc.load([omc, onc]).to(tl.float32) if EPI == 4 else x
            c_desc.store([omc, onc], apply_epi(x, g, r, es, EPI).to(c_desc.dtype))


@contextlib.contextmanager
def _meta_ws():
    import triton.knobs
    prev = triton.knobs.nvidia.use_meta_ws
    triton.knobs.nvidia.use_meta_ws = True
    try:
        yield
    finally:
        triton.knobs.nvidia.use_meta_ws = prev


_TL = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


def fused_plain(a, b, gemm_dtype, epi, g=None, r=None, escale=1.0, scale=1.0, out=None, BK=64, GROUP_M=8,
                SUBTILE=True, WS=True, META_WS=True, cfg=None):
    """The `mode=plain` bit-exact GEMM with `epi` folded in. Returns None when TMA cannot carry
    one of the tensors -- there is no point measuring a fallback path here."""
    from triton.tools.tensor_descriptor import TensorDescriptor
    M, K = a.shape
    N = b.shape[1]
    BM, BN, nw, ns = cfg if cfg is not None else _sm103_config(M, N, a.dtype == torch.float8_e4m3fn)
    b_nk = b.stride(0) == 1 and b.stride(1) != 1
    bt = b.t() if b_nk else b
    bb = [BN, BK] if b_nk else [BK, BN]
    cbn = BN // 2 if SUBTILE else BN
    odt = torch.float8_e4m3fn if epi in EPI_FP8_OUT else gemm_dtype
    if out is None:
        out = torch.empty(M, N, device="cuda", dtype=odt)
    tensors = [(a, BK * a.element_size()), (bt, bb[1] * bt.element_size()), (out, cbn * out.element_size())]
    if epi in EPI_NEEDS_G:
        tensors.append((g, cbn * g.element_size()))
    if epi in EPI_NEEDS_R:
        tensors.append((r, cbn * r.element_size()))
    if not all(_tma_ok(t, nb) for t, nb in tensors):
        return None
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(bt, bb)
    c_desc = TensorDescriptor.from_tensor(out, [BM, cbn])
    dummy = a_desc
    g_desc = TensorDescriptor.from_tensor(g, [BM, cbn]) if epi in EPI_NEEDS_G else dummy
    r_desc = TensorDescriptor.from_tensor(r, [BM, cbn]) if epi in EPI_NEEDS_R else dummy
    nprog = min(_num_sms(), triton.cdiv(M, BM) * triton.cdiv(N, BN))
    with (_meta_ws() if META_WS else contextlib.nullcontext()):
        _plain_gemm_tma_fused[(nprog, )](a_desc, b_desc, c_desc, g_desc, r_desc, M, N, K, scale, escale, B_NK=b_nk,
                                         GROUP_M=GROUP_M, NUM_SMS=nprog, SUBTILE=SUBTILE, WS=WS, FLATTEN=False,
                                         EPI=epi, BM=BM, BN=BN, BK=BK, ODT=_TL[gemm_dtype], num_warps=nw,
                                         num_stages=ns)
    return out


# --------------------------------------------------------------------------------------------
# The same thing without TMA. `_triton_plain_tma_sm103` falls back to `_plain_gemm_sm103`
# whenever a tensor cannot carry a descriptor, which is every N that is not a whole number of
# 16-byte lines -- exactly the non-aligned shapes cuBLAS declines to optimise. Same bit contract.
# --------------------------------------------------------------------------------------------


@triton.jit
def _plain_gemm_fused_notma(A, B, C, G, R, M, N, K, am, ak, bk, bn, cm, cn, gm, gn, s, es, GROUP_M: tl.constexpr,
                            EVEN_K: tl.constexpr, I64: tl.constexpr, EPI: tl.constexpr, BM: tl.constexpr,
                            BN: tl.constexpr, BK: tl.constexpr, ODT: tl.constexpr):
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
    msk = (ocm[:, None] < M) & (ocn[None, :] < N)
    off = gm * ocm[:, None] + gn * ocn[None, :]
    x = acc.to(ODT).to(tl.float32)
    g = tl.load(G + off, mask=msk, other=0.0).to(tl.float32) if EPI == 2 or EPI == 5 else x
    r = tl.load(R + off, mask=msk, other=0.0).to(tl.float32) if EPI == 4 else x
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], apply_epi(x, g, r, es, EPI).to(C.dtype.element_ty), mask=msk)


def fused_plain_notma(a, b, gemm_dtype, epi, g=None, r=None, escale=1.0, scale=1.0, out=None, BK=64, GROUP_M=8,
                      cfg=(128, 128, 8, 3)):
    from bitequiv.cublas_match.kernels import _fits_i32
    BM, BN, nw, ns = cfg
    if b.stride(0) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    M, K = a.shape
    N = b.shape[1]
    odt = torch.float8_e4m3fn if epi in EPI_FP8_OUT else gemm_dtype
    if out is None:
        out = torch.empty(M, N, device="cuda", dtype=odt)
    ref = g if epi in EPI_NEEDS_G else (r if epi in EPI_NEEDS_R else out)
    _plain_gemm_fused_notma[(triton.cdiv(M, BM) * triton.cdiv(N, BN), )](
        a, b, out, g if g is not None else out, r if r is not None else out, M, N, K, a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), out.stride(0), out.stride(1), ref.stride(0), ref.stride(1), scale, escale,
        GROUP_M=GROUP_M, EVEN_K=(K % BK == 0), I64=not _fits_i32(a, b, out), EPI=epi, BM=BM, BN=BN, BK=BK,
        ODT=_TL[gemm_dtype], num_warps=nw, num_stages=ns)
    return out


# --------------------------------------------------------------------------------------------
# arm 3: the same fusion on an ordinary Triton matmul with no numerics requirement. The ceiling,
# and what torch.compile with max-autotune would pick.
# --------------------------------------------------------------------------------------------


@triton.jit
def _free_gemm_fused(A, B, C, G, R, M, N, K, am, ak, bk, bn, cm, cn, gm, gn, es, EPI: tl.constexpr, BM: tl.constexpr,
                     BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    ng = GM * npn
    gid = pid // ng
    first = gid * GM
    gs = tl.minimum(npm - first, GM)
    pm = first + ((pid % ng) % gs)
    pn = (pid % ng) // gs
    om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
    on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
    ok = tl.arange(0, BK).to(tl.int64)
    ap = A + om[:, None] * am + ok[None, :] * ak
    bp = B + ok[:, None] * bk + on[None, :] * bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        m = ok < K - k * BK
        acc = tl.dot(tl.load(ap, mask=m[None, :], other=0.0), tl.load(bp, mask=m[:, None], other=0.0), acc)
        ap += BK * ak
        bp += BK * bk
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    msk = (ocm[:, None] < M) & (ocn[None, :] < N)
    goff = G + gm * ocm[:, None] + gn * ocn[None, :]
    g = tl.load(goff, mask=msk, other=0.0).to(tl.float32) if EPI == 2 or EPI == 5 else acc
    r = tl.load(R + gm * ocm[:, None] + gn * ocn[None, :], mask=msk, other=0.0).to(tl.float32) if EPI == 4 else acc
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :],
             apply_epi(acc, g, r, es, EPI).to(C.dtype.element_ty), mask=msk)


FREE_SPACE = [(bm, bn, bk, w, s) for bm in (64, 128, 256) for bn in (64, 128, 256) for bk in (16, 32, 64)
              for w in (4, 8) for s in (2, 3, 4)]


def free_fused(a, b, cfg, epi, g=None, r=None, escale=1.0, out=None):
    """The unconstrained arm. No rounding to the GEMM's output dtype before the epilogue -- that
    is the whole point: it keeps the accumulator in fp32 straight into the epilogue, which is
    what an ordinary fused matmul does and why its answer is not the unfused one."""
    BM, BN, BK, W, S = cfg
    M, K = a.shape
    N = b.shape[1]
    odt = torch.float8_e4m3fn if epi in EPI_FP8_OUT else a.dtype
    if out is None:
        out = torch.empty(M, N, device="cuda", dtype=odt)
    gg = g if g is not None else out
    rr = r if r is not None else out
    ref = g if g is not None else (r if r is not None else out)
    _free_gemm_fused[(triton.cdiv(M, BM) * triton.cdiv(N, BN), )](a, b, out, gg, rr, M, N, K, a.stride(0), a.stride(1),
                                                                  b.stride(0), b.stride(1), out.stride(0),
                                                                  out.stride(1), ref.stride(0), ref.stride(1), escale,
                                                                  EPI=epi, BM=BM, BN=BN, BK=BK, GM=8, num_warps=W,
                                                                  num_stages=S)
    return out
