"""Bit-exact GEMMs with a REAL model epilogue folded in, for the `mode=plain` shapes on sm_103.

Copied from `artifact_eval/fusion/fused.py` (which is left untouched) and cut down to the
epilogues that actually follow a GEMM in a transformer:

    lora    y = base + x * (alpha / r)      after `lora_B`            second M x N read
    fp8q    y = (x * s) as fp8              at a quantization boundary  output SHRINKS 2x
    swiglu  y = x * silu(gate)              after `up_proj`           second M x N read
    resid   y = x + residual                after `o_proj`            second M x N read
    argmax  row-wise argmax over N          after `lm_head`           output collapses to 1

The bit contract is copied from `bitequiv/cublas_match/kernels.py::_plain_gemm_tma_sm103`: BK
real k-elements per `tl.dot`, in increasing k, into one fp32 accumulator, scaled once and
rounded once.  Only the store differs -- the accumulator is rounded to the GEMM's output dtype
first (that is where the unfused path rounds), read back as fp32, and the epilogue is applied to
that.  So the fused result is the unfused result by construction, and the byte gate in `run.py`
checks it against cuBLAS plus a separate Triton epilogue kernel on real inputs anyway.

`apply_epi` is shared by the fused kernel and the standalone one, so the two arms cannot drift
in the arithmetic -- only in where the value came from.
"""
from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl

from bitequiv.cublas_match.kernels import _num_sms, _sm103_config, _tma_ok

# Epilogue codes.  None of these is something cuBLASLt can express -- its whole list is bias,
# relu, gelu and their backward / bias-gradient variants.
EPI_NONE = 0
EPI_LORA = 1  # y = r + x * s          r = the frozen layer's output, s = alpha / rank
EPI_FP8Q = 2  # y = (x * s) -> fp8     static-scale cast for the next layer
EPI_SWIGLU = 3  # y = x * silu(g)      g = gate_proj's output, x = up_proj's
EPI_RESID = 4  # y = x + r             r = the residual stream
EPI_ARGMAX = 5  # row-wise argmax over N; handled by its own kernels, not `apply_epi`

EPI_NAMES = {
    EPI_NONE: "none", EPI_LORA: "lora", EPI_FP8Q: "fp8q", EPI_SWIGLU: "swiglu", EPI_RESID: "resid", EPI_ARGMAX: "argmax"
}
EPI_CODES = {v: k for k, v in EPI_NAMES.items()}

EPI_NEEDS_G = {EPI_SWIGLU}  # reads a second M x N tensor as `g`
EPI_NEEDS_R = {EPI_LORA, EPI_RESID}  # reads a second M x N tensor as `r`
EPI_FP8_OUT = {EPI_FP8Q}
NEG_INF = float("-inf")
BIG_IDX = 2147483647


def out_dtype_for(epi, gemm_dtype):
    return torch.float8_e4m3fn if epi in EPI_FP8_OUT else gemm_dtype


@triton.jit
def apply_epi(x, g, r, s, EPI: tl.constexpr):
    """The epilogue itself, in fp32, on the value the unfused GEMM would have left in HBM."""
    if EPI == 1:
        y = r + x * s
    elif EPI == 2:
        y = x * s
    elif EPI == 3:
        y = x * (g * tl.sigmoid(g))
    elif EPI == 4:
        y = x + r
    else:
        y = x
    return y


# ------------------------------------------------------------------------------------------- #
# arm 1's second kernel: the epilogue on its own, reading the GEMM's output back from HBM.
# One kernel even for the two-op chains, which is what Inductor produces.
# ------------------------------------------------------------------------------------------- #


@triton.jit
def epi_kernel(C, G, R, O, NEL, s, EPI: tl.constexpr, NEED_G: tl.constexpr, NEED_R: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < NEL
    x = tl.load(C + off, mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + off, mask=m, other=0.0).to(tl.float32) if NEED_G else x
    r = tl.load(R + off, mask=m, other=0.0).to(tl.float32) if NEED_R else x
    tl.store(O + off, apply_epi(x, g, r, s, EPI).to(O.dtype.element_ty), mask=m)


def run_epi(c, g, r, scale, epi, out=None, cfg=(4096, 8)):
    """The standalone epilogue.  `out` lets the caller reuse a buffer so the timed call allocates
    nothing -- an allocation inside a CUDA graph capture is not what we want to measure."""
    nel = c.numel()
    if out is None:
        out = torch.empty_like(c, dtype=out_dtype_for(epi, c.dtype))
    BLOCK, nw = cfg
    epi_kernel[(triton.cdiv(nel, BLOCK), )](c, g if g is not None else c, r if r is not None else c, out, nel, scale,
                                            EPI=epi, NEED_G=epi in EPI_NEEDS_G, NEED_R=epi in EPI_NEEDS_R, BLOCK=BLOCK,
                                            num_warps=nw)
    return out


# ------------------------------------------------------------------------------------------- #
# argmax.  Both arms use the same rule -- the largest value, and among equal largest values the
# LOWEST column index.  fp16 has few distinct values, so over 128256 columns a tie for the
# maximum is common; without a fixed rule the two arms would disagree for a reason that has
# nothing to do with the GEMM.  The rule is a monoid, so reducing per tile and then merging is
# exactly the same answer as one left-to-right scan.
# ------------------------------------------------------------------------------------------- #


@triton.jit
def _merge(bv, bi, x, n):
    """One step of the monoid, elementwise: the larger value wins, and on a tie the lower index."""
    ni = tl.where(x > bv, n, tl.where(x == bv, tl.minimum(bi, n), bi))
    return tl.maximum(bv, x), ni


@triton.jit
def argmax_rows_kernel(C, OV, OI, M, N, sm, BLOCK: tl.constexpr, NINF: tl.constexpr = NEG_INF,
                       BIG: tl.constexpr = BIG_IDX):
    """arm 1's epilogue: the row-wise argmax of the M x N matrix cuBLAS just wrote to HBM."""
    m = tl.program_id(0)
    bv = tl.full([BLOCK], NINF, tl.float32)
    bi = tl.full([BLOCK], BIG, tl.int32)
    for n0 in range(0, N, BLOCK):
        n = n0 + tl.arange(0, BLOCK)
        x = tl.load(C + m.to(tl.int64) * sm + n, mask=n < N, other=NINF).to(tl.float32)
        bv, bi = _merge(bv, bi, x, n.to(tl.int32))
    best = tl.max(bv, axis=0)
    tl.store(OV + m, best)
    tl.store(OI + m, tl.min(tl.where(bv == best, bi, BIG), axis=0))


def run_argmax(c, ov=None, oi=None, cfg=(4096, 8)):
    M, N = c.shape
    if ov is None:
        ov = torch.empty(M, device="cuda", dtype=torch.float32)
    if oi is None:
        oi = torch.empty(M, device="cuda", dtype=torch.int32)
    BLOCK, nw = cfg
    argmax_rows_kernel[(M, )](c, ov, oi, M, N, c.stride(0), BLOCK=BLOCK, num_warps=nw)
    return ov, oi


@triton.jit
def argmax_merge_kernel(PV, PI, OV, OI, M, T, BT: tl.constexpr, NINF: tl.constexpr = NEG_INF,
                        BIG: tl.constexpr = BIG_IDX):
    """Stage 2 of the fused arm: merge the per-tile winners.  T is at most a few thousand and M
    at most a few hundred, so this reads under a megabyte."""
    m = tl.program_id(0)
    bv = tl.full([BT], NINF, tl.float32)
    bi = tl.full([BT], BIG, tl.int32)
    for t0 in range(0, T, BT):
        t = t0 + tl.arange(0, BT)
        msk = t < T
        v = tl.load(PV + t * M + m, mask=msk, other=NINF)
        i = tl.load(PI + t * M + m, mask=msk, other=BIG)
        bv, bi = _merge(bv, bi, v, i)
    best = tl.max(bv, axis=0)
    tl.store(OV + m, best)
    tl.store(OI + m, tl.min(tl.where(bv == best, bi, BIG), axis=0))


# ------------------------------------------------------------------------------------------- #
# arm 2: the bit-exact GEMM with the epilogue folded in.
# ------------------------------------------------------------------------------------------- #


@triton.jit
def _plain_gemm_tma_fused(a_desc, b_desc, c_desc, g_desc, r_desc, M, N, K, s, es, B_NK: tl.constexpr,
                          GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, SUBTILE: tl.constexpr, WS: tl.constexpr,
                          FLATTEN: tl.constexpr, EPI: tl.constexpr, NEED_G: tl.constexpr, NEED_R: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ODT: tl.constexpr, ROUND: tl.constexpr):
    """`_plain_gemm_tma_sm103` with the epilogue folded in.  The k loop is copied character for
    character; `acc.to(ODT)` is the GEMM's own rounding, i.e. exactly the bytes the unfused path
    would have written to HBM, and the epilogue runs on that."""
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
            x0 = acc0.to(ODT).to(tl.float32) if ROUND else acc0
            g0 = g_desc.load([omc, onc]).to(tl.float32) if NEED_G else x0
            r0 = r_desc.load([omc, onc]).to(tl.float32) if NEED_R else x0
            c_desc.store([omc, onc], apply_epi(x0, g0, r0, es, EPI).to(c_desc.dtype))
            x1 = acc1.to(ODT).to(tl.float32) if ROUND else acc1
            g1 = g_desc.load([omc, n1]).to(tl.float32) if NEED_G else x1
            r1 = r_desc.load([omc, n1]).to(tl.float32) if NEED_R else x1
            c_desc.store([omc, n1], apply_epi(x1, g1, r1, es, EPI).to(c_desc.dtype))
        else:
            x = acc.to(ODT).to(tl.float32) if ROUND else acc
            g = g_desc.load([omc, onc]).to(tl.float32) if NEED_G else x
            r = r_desc.load([omc, onc]).to(tl.float32) if NEED_R else x
            c_desc.store([omc, onc], apply_epi(x, g, r, es, EPI).to(c_desc.dtype))


@triton.jit
def _plain_gemm_tma_argmax(a_desc, b_desc, PV, PI, M, N, K, s, B_NK: tl.constexpr, GROUP_M: tl.constexpr,
                           NUM_SMS: tl.constexpr, WS: tl.constexpr, FLATTEN: tl.constexpr, BM: tl.constexpr,
                           BN: tl.constexpr, BK: tl.constexpr, ODT: tl.constexpr, NINF: tl.constexpr = NEG_INF,
                           BIG: tl.constexpr = BIG_IDX):
    """The same GEMM, but each tile reduces its own BM x BN block to one (value, index) pair per
    row and writes that.  The M x N intermediate is never written to HBM at all."""
    start_pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    num_tiles = npm * npn
    per_group = GROUP_M * npn

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
        x = (acc * s).to(ODT).to(tl.float32)
        cols = on + tl.arange(0, BN)
        x = tl.where(cols[None, :] < N, x, NINF)
        mx = tl.max(x, axis=1)
        mi = tl.min(tl.where(x == mx[:, None], cols[None, :], BIG), axis=1)
        r = om + tl.arange(0, BM)
        off = (on // BN) * M + r
        tl.store(PV + off, mx, mask=r < M)
        tl.store(PI + off, mi, mask=r < M)


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


def fused_plain(a, b, gemm_dtype, epi, g=None, r=None, escale=1.0, scale=1.0, out=None, BK=64, GROUP_M=8, SUBTILE=True,
                WS=True, META_WS=True, cfg=None, ROUND=True):
    """The `mode=plain` bit-exact GEMM with `epi` folded in.  Returns None when TMA cannot carry
    one of the tensors -- there is no point measuring a fallback path here."""
    from triton.tools.tensor_descriptor import TensorDescriptor
    M, K = a.shape
    N = b.shape[1]
    BM, BN, nw, ns = cfg if cfg is not None else _sm103_config(M, N, a.dtype == torch.float8_e4m3fn)
    b_nk = b.stride(0) == 1 and b.stride(1) != 1
    bt = b.t() if b_nk else b
    bb = [BN, BK] if b_nk else [BK, BN]
    cbn = BN // 2 if SUBTILE else BN
    if out is None:
        out = torch.empty(M, N, device="cuda", dtype=out_dtype_for(epi, gemm_dtype))
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
    g_desc = TensorDescriptor.from_tensor(g, [BM, cbn]) if epi in EPI_NEEDS_G else a_desc
    r_desc = TensorDescriptor.from_tensor(r, [BM, cbn]) if epi in EPI_NEEDS_R else a_desc
    nprog = min(_num_sms(), triton.cdiv(M, BM) * triton.cdiv(N, BN))
    with (_meta_ws() if META_WS else contextlib.nullcontext()):
        _plain_gemm_tma_fused[(nprog, )](a_desc, b_desc, c_desc, g_desc, r_desc, M, N, K, scale, escale, B_NK=b_nk,
                                         GROUP_M=GROUP_M, NUM_SMS=nprog, SUBTILE=SUBTILE, WS=WS, FLATTEN=False, EPI=epi,
                                         NEED_G=epi in EPI_NEEDS_G, NEED_R=epi in EPI_NEEDS_R, BM=BM, BN=BN, BK=BK,
                                         ODT=_TL[gemm_dtype], ROUND=ROUND, num_warps=nw, num_stages=ns)
    return out


def fused_argmax(a, b, gemm_dtype, scale=1.0, bufs=None, BK=64, GROUP_M=8, WS=True, META_WS=True, cfg=None):
    """lm_head + greedy decode: the fused GEMM writes one (value, index) pair per row per column
    tile, and a second tiny kernel merges them.  Returns (values, indices) or None if TMA is
    refused."""
    from triton.tools.tensor_descriptor import TensorDescriptor
    M, K = a.shape
    N = b.shape[1]
    BM, BN, nw, ns = cfg if cfg is not None else _sm103_config(M, N, False)
    b_nk = b.stride(0) == 1 and b.stride(1) != 1
    bt = b.t() if b_nk else b
    bb = [BN, BK] if b_nk else [BK, BN]
    if not (_tma_ok(a, BK * a.element_size()) and _tma_ok(bt, bb[1] * bt.element_size())):
        return None
    T = triton.cdiv(N, BN)
    if bufs is None or bufs[0].numel() < T * M:
        bufs = (torch.empty(T * M, device="cuda",
                            dtype=torch.float32), torch.empty(T * M, device="cuda", dtype=torch.int32),
                torch.empty(M, device="cuda", dtype=torch.float32), torch.empty(M, device="cuda", dtype=torch.int32))
    pv, pi, ov, oi = bufs
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(bt, bb)
    nprog = min(_num_sms(), triton.cdiv(M, BM) * T)
    with (_meta_ws() if META_WS else contextlib.nullcontext()):
        _plain_gemm_tma_argmax[(nprog, )](a_desc, b_desc, pv, pi, M, N, K, scale, B_NK=b_nk, GROUP_M=GROUP_M,
                                          NUM_SMS=nprog, WS=WS, FLATTEN=False, BM=BM, BN=BN, BK=BK, ODT=_TL[gemm_dtype],
                                          num_warps=nw, num_stages=ns)
    bt_merge = max(16, min(1024, triton.next_power_of_2(T)))
    argmax_merge_kernel[(M, )](pv, pi, ov, oi, M, T, BT=bt_merge, num_warps=4)
    return ov, oi


# ------------------------------------------------------------------------------------------- #
# The same thing without TMA, for a shape whose row stride is not a whole number of 16-byte
# lines.  `_triton_plain_tma_sm103` falls back the same way, so the fused arm falls back with it.
# ------------------------------------------------------------------------------------------- #


@triton.jit
def _plain_gemm_fused_notma(A, B, C, G, R, M, N, K, am, ak, bk, bn, cm, cn, gm, gn, s, es, GROUP_M: tl.constexpr,
                            EVEN_K: tl.constexpr, I64: tl.constexpr, EPI: tl.constexpr, NEED_G: tl.constexpr,
                            NEED_R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                            ODT: tl.constexpr, ROUND: tl.constexpr):
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
    x = acc.to(ODT).to(tl.float32) if ROUND else acc
    g = tl.load(G + off, mask=msk, other=0.0).to(tl.float32) if NEED_G else x
    r = tl.load(R + off, mask=msk, other=0.0).to(tl.float32) if NEED_R else x
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], apply_epi(x, g, r, es, EPI).to(C.dtype.element_ty), mask=msk)


def fused_plain_notma(a, b, gemm_dtype, epi, g=None, r=None, escale=1.0, scale=1.0, out=None, BK=64, GROUP_M=8,
                      cfg=(128, 128, 8, 3), ROUND=True):
    from bitequiv.cublas_match.kernels import _fits_i32
    BM, BN, nw, ns = cfg
    if b.stride(0) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    M, K = a.shape
    N = b.shape[1]
    if out is None:
        out = torch.empty(M, N, device="cuda", dtype=out_dtype_for(epi, gemm_dtype))
    ref = g if epi in EPI_NEEDS_G else (r if epi in EPI_NEEDS_R else out)
    _plain_gemm_fused_notma[(triton.cdiv(M, BM) * triton.cdiv(N, BN), )](
        a, b, out, g if g is not None else out, r if r is not None else out, M, N, K, a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), out.stride(0), out.stride(1), ref.stride(0), ref.stride(1), scale, escale,
        GROUP_M=GROUP_M, EVEN_K=(K % BK == 0), I64=not _fits_i32(a, b, out), EPI=epi, NEED_G=epi in EPI_NEEDS_G,
        NEED_R=epi in EPI_NEEDS_R, BM=BM, BN=BN, BK=BK, ODT=_TL[gemm_dtype], ROUND=ROUND, num_warps=nw, num_stages=ns)
    return out


# ------------------------------------------------------------------------------------------- #
# arm 3: the same fusion on an ordinary Triton matmul with no numerics requirement.  The ceiling,
# and what torch.compile with max-autotune would pick.  It keeps the accumulator in fp32 straight
# into the epilogue, which is why its answer is not the unfused one.
# ------------------------------------------------------------------------------------------- #


@triton.jit
def _free_gemm_fused(A, B, C, G, R, M, N, K, am, ak, bk, bn, cm, cn, gm, gn, es, EPI: tl.constexpr,
                     NEED_G: tl.constexpr, NEED_R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     GM: tl.constexpr):
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
    off = gm * ocm[:, None] + gn * ocn[None, :]
    g = tl.load(G + off, mask=msk, other=0.0).to(tl.float32) if NEED_G else acc
    r = tl.load(R + off, mask=msk, other=0.0).to(tl.float32) if NEED_R else acc
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], apply_epi(acc, g, r, es, EPI).to(C.dtype.element_ty), mask=msk)


@triton.jit
def _free_gemm_argmax(A, B, PV, PI, M, N, K, am, ak, bk, bn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GM: tl.constexpr, NINF: tl.constexpr = NEG_INF, BIG: tl.constexpr = BIG_IDX):
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
    cols = (pn * BN + tl.arange(0, BN))
    rws = pm * BM + tl.arange(0, BM)
    x = tl.where(cols[None, :] < N, acc, NINF)
    mx = tl.max(x, axis=1)
    mi = tl.min(tl.where(x == mx[:, None], cols[None, :], BIG), axis=1)
    off = pn * M + rws
    tl.store(PV + off, mx, mask=rws < M)
    tl.store(PI + off, mi, mask=rws < M)


def free_fused(a, b, cfg, epi, g=None, r=None, escale=1.0, out=None):
    BM, BN, BK, W, S = cfg
    M, K = a.shape
    N = b.shape[1]
    if out is None:
        out = torch.empty(M, N, device="cuda", dtype=out_dtype_for(epi, a.dtype))
    ref = g if g is not None else (r if r is not None else out)
    _free_gemm_fused[(triton.cdiv(M, BM) * triton.cdiv(N, BN), )](a, b, out, g if g is not None else out,
                                                                  r if r is not None else out, M, N, K, a.stride(0),
                                                                  a.stride(1), b.stride(0), b.stride(1), out.stride(0),
                                                                  out.stride(1), ref.stride(0), ref.stride(1), escale,
                                                                  EPI=epi, NEED_G=epi in EPI_NEEDS_G, NEED_R=epi
                                                                  in EPI_NEEDS_R, BM=BM, BN=BN, BK=BK, GM=8,
                                                                  num_warps=W, num_stages=S)
    return out


def free_argmax(a, b, cfg, bufs=None):
    BM, BN, BK, W, S = cfg
    M, K = a.shape
    N = b.shape[1]
    T = triton.cdiv(N, BN)
    if bufs is None or bufs[0].numel() < T * M:
        bufs = (torch.empty(T * M, device="cuda",
                            dtype=torch.float32), torch.empty(T * M, device="cuda", dtype=torch.int32),
                torch.empty(M, device="cuda", dtype=torch.float32), torch.empty(M, device="cuda", dtype=torch.int32))
    pv, pi, ov, oi = bufs
    _free_gemm_argmax[(triton.cdiv(M, BM) * T, )](a, b, pv, pi, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                                                  b.stride(1), BM=BM, BN=BN, BK=BK, GM=8, num_warps=W, num_stages=S)
    argmax_merge_kernel[(M, )](pv, pi, ov, oi, M, T, BT=max(16, min(1024, triton.next_power_of_2(T))), num_warps=4)
    return ov, oi
