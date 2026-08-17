"""Step 2 -- one kernel that is the bit-exact GEMM with Inductor's epilogue folded in.

Two halves, and only one of them is ours.

  the mainloop  is the recipe `bitequiv.cublas_match` derives for the shape. This file only
                serves plan modes `plain` and `k_per_dot`; the split-K and gemv modes need a
                second kernel or a workspace and cannot live inside one fused template, so a
                shape whose plan is one of those is refused rather than approximated.
  the epilogue  is copied from what Inductor emits under `emulate_precision_casts=1`, which is
                the knob that makes its fused template round the accumulator through the output
                dtype first and then round after every epilogue op -- i.e. reproduce the
                op-by-op rounding of the unfused eager path. Without that knob Inductor keeps
                the whole chain in fp32 and is NOT bit-faithful; that is the default, and it is
                what arm 2 of the three-way table measures.

Nothing here is trusted: `best_fused` only ever returns a config whose bytes matched the eager
reference, and the caller checks the bytes again on fresh draws.
"""
from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# epilogue selector, shared by the kernels and the torch-side reference
EPI_NONE = 0
EPI_SILU = 1
EPI_FP8CAST = 2
EPI_SOFTCAP = 3
EPI_GATE = 4
EPI_RESSCALE = 5
EPI_SILU_FAST = 6  # silu with Triton's own sigmoid: NOT bit-exact, only for costing the exact one
EPI_SILU_RN = 7  # silu with the correctly rounded exp and divide: bit-exact but 2x the PTX

EPI_ID = {
    "none": EPI_NONE, "silu": EPI_SILU, "fp8cast": EPI_FP8CAST, "softcap": EPI_SOFTCAP, "gate": EPI_GATE, "resscale":
    EPI_RESSCALE, "silu_fast": EPI_SILU_FAST, "silu_rn": EPI_SILU_RN
}
EPI_NEEDS_R = {EPI_GATE, EPI_RESSCALE}  # takes one extra [M,N] tensor
EPI_OUT_FP8 = {EPI_FP8CAST}

FP8 = torch.float8_e4m3fn


# --------------------------------------------------------------------------------------------
# the epilogue, in Triton, with the rounding points the unfused eager path has
# --------------------------------------------------------------------------------------------
@triton.jit
def _sigmoid(x):
    """A spelling of sigmoid that returns torch eager's fp16 bytes on every fp16 input.

    torch's CUDA sigmoid for a low-precision dtype is `1.f / (1.f + ::expf(-x))` evaluated in
    fp32, i.e. the accurate `__nv_expf` and the correctly rounded `div.rn.f32`. Writing that
    literally is `_sigmoid_rn` below, and it costs about 0.0020 ms per 16.8M elements over the
    approximate divide -- which is most of what bit-exactness would cost.

    It is not needed. The claim to defend is that the OUTPUT BYTES match, not that the
    instructions match, and the input here is an fp16 value, so the whole input domain is 63488
    values and can be checked exhaustively instead of sampled. `probe_sigmoid.py` does that: over
    all 63488, for sigmoid and for the composed silu alike, this spelling gives torch's bytes on
    63488 of 63488 and costs nothing over a plain copy. `tl.sigmoid`, which is what Inductor
    emits, differs on 2 -- exact fp16 ties where a sub-ulp fp32 error decides the rounding, which
    is why it survives ordinary random testing and still is not bit-exact.

    Why it works: `div.approx.f32` is MUFU.RCP plus a multiply, about 22 good bits; one Newton
    step takes that past what fp32 carries, and the fp16 store keeps only 11 bits. The `tl.where`
    is not cosmetic -- for x below about -88 the exp overflows to inf, so q is exactly 0 and the
    correction computes inf*0 = NaN. Unguarded, that alone is 9844 of the 63488 wrong.
    """
    d = 1.0 + libdevice.fast_expf(-x)
    q = libdevice.fast_dividef(1.0, d)
    return tl.where(q > 0.0, tl.fma(tl.fma(-d, q, 1.0), q, q), q)


@triton.jit
def _sigmoid_rn(x):
    """The literal transcription of torch's C++: correctly rounded exp and divide. Kept only to
    price `_sigmoid` against it -- both give the same bytes."""
    return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))


@triton.jit
def _epilogue(acc, r, EPI: tl.constexpr, HAS_R: tl.constexpr, RDT: tl.constexpr):
    """`acc` is the fp32 accumulator, `r` the extra operand already widened to fp32.

    Every `.to(RDT).to(tl.float32)` is one eager kernel boundary: eager materialises a RDT tensor
    there, so the value has to be rounded to RDT and read back. Dropping any one of them is what
    makes an ordinary fused epilogue faster and not bit-exact.

    The one exception is the LAST op of each chain, which is left unrounded here. The caller
    stores the result with `.to(<output dtype>)`, and for every epilogue below that store performs
    the same rounding the dropped `.to(RDT)` would have: for the fp16 outputs because
    `v.to(f16).to(f32).to(f16) == v.to(f16)`, and for the fp8 output because its last op is a
    clamp to +-448, which is exact in fp16, so the dropped rounding was already a no-op.
    """
    x = acc.to(RDT).to(tl.float32)  # what `torch.mm` wrote to memory
    if EPI == 1:  # silu:  x * sigmoid(x)
        s = _sigmoid(x).to(RDT).to(tl.float32)
        y = x * s
    elif EPI == 2:  # fp8cast: (x * 0.375).clamp(-448, 448) -> e4m3
        t = (x * 0.375).to(RDT).to(tl.float32)
        y = tl.minimum(tl.maximum(t, -448.0), 448.0)
    elif EPI == 3:  # softcap: tanh(x * 0.0625) * 16
        t = (x * 0.0625).to(RDT).to(tl.float32)
        t = libdevice.tanh(t).to(RDT).to(tl.float32)
        y = t * 16.0
    elif EPI == 4:  # gate: x * sigmoid(g)
        s = _sigmoid(r).to(RDT).to(tl.float32)
        y = x * s
    elif EPI == 5:  # resscale: (x + r) * 1.7
        t = (x + r).to(RDT).to(tl.float32)
        y = t * 1.7
    elif EPI == 6:  # silu with Triton's approximate sigmoid: NOT bit-exact, prices the exact one
        s = tl.sigmoid(x).to(RDT).to(tl.float32)
        y = x * s
    elif EPI == 7:  # silu with the correctly rounded exp and divide: bit-exact, and 2x the PTX
        s = _sigmoid_rn(x).to(RDT).to(tl.float32)
        y = x * s
    else:
        y = x
    if HAS_R:  # keep `r` live for the EPI values that ignore it
        y = y
    return y


# --------------------------------------------------------------------------------------------
# plain mode: one fp32 accumulator over the whole K, increasing k, scale once, round once
# --------------------------------------------------------------------------------------------
@triton.jit
def _fused_plain_tma(a_desc, b_desc, c_desc, r_desc, M, N, K, s, EPI: tl.constexpr, HAS_R: tl.constexpr,
                     RDT: tl.constexpr, B_NK: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
                     WS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """`_plain_gemm_tma_sm103` with the epilogue folded in before the store.

    The k loop is unchanged and that is the point: BK real k-elements per `tl.dot`, in
    increasing k, into one fp32 accumulator, scaled once. Only what happens to `acc` after the
    loop is different.
    """
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
        if HAS_R:
            r = r_desc.load([om, on]).to(tl.float32)
        else:
            r = acc
        y = _epilogue(acc, r, EPI, HAS_R, RDT)
        c_desc.store([om, on], y.to(c_desc.dtype))


@triton.jit
def _fused_plain(A, B, C, R, M, N, K, am, ak, bk, bn, cm, cn, rm_, rn_, s, EPI: tl.constexpr, HAS_R: tl.constexpr,
                 RDT: tl.constexpr, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr, I64: tl.constexpr, BM: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr):
    """`_plain_gemm_sm103` with the same epilogue, for shapes TMA cannot take."""
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
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
    if HAS_R:
        r = tl.load(R + rm_ * ocm[:, None] + rn_ * ocn[None, :], mask=mask, other=0.0).to(tl.float32)
    else:
        r = acc
    y = _epilogue(acc, r, EPI, HAS_R, RDT)
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], y.to(C.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------------------------
# k_per_dot mode: the accumulator rounds where CUTLASS rounds
# --------------------------------------------------------------------------------------------
@triton.jit
def _fused_kpd(A, B, C, R, M, N, K, am, ak, bk, bn, cm, cn, rm_, rn_, s, KPD, RES, EPI: tl.constexpr,
               HAS_R: tl.constexpr, RDT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """`_plain_gemm_k_per_dot` with the epilogue folded in; the k grouping is untouched."""
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
    on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
    ok = tl.arange(0, BK).to(tl.int64)
    pre = RES // KPD
    part = RES - pre * KPD
    has_part = tl.where(part > 0, 1, 0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, pre + has_part + (K - RES) // KPD):
        is_pre = g < pre
        is_part = (part > 0) & (g == pre)
        k0 = tl.where(is_pre, g * KPD, tl.where(is_part, pre * KPD, RES + (g - pre - has_part) * KPD))
        klen = tl.where(is_part, part, KPD)
        real = ok < klen
        a = tl.load(A + om[:, None] * am + (k0 + ok)[None, :] * ak, mask=real[None, :], other=0.0)
        b = tl.load(B + (k0 + ok)[:, None] * bk + on[None, :] * bn, mask=real[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
    acc = acc * s
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    mask = (ocm[:, None] < M) & (ocn[None, :] < N)
    if HAS_R:
        r = tl.load(R + rm_ * ocm[:, None] + rn_ * ocn[None, :], mask=mask, other=0.0).to(tl.float32)
    else:
        r = acc
    y = _epilogue(acc, r, EPI, HAS_R, RDT)
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], y.to(C.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------------------------
# launchers
# --------------------------------------------------------------------------------------------
_TL = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


def _tma_ok(t, block_inner_bytes: int) -> bool:
    if t.stride(-1) != 1 or block_inner_bytes % 16:
        return False
    e = t.element_size()
    return all(s * e % 16 == 0 for s in t.stride()[:-1])


@contextlib.contextmanager
def _meta_ws():
    import triton.knobs
    prev = triton.knobs.nvidia.use_meta_ws
    triton.knobs.nvidia.use_meta_ws = True
    try:
        yield
    finally:
        triton.knobs.nvidia.use_meta_ws = prev


_I32_MAX = 2**31 - 1


def _fits_i32(*ts) -> bool:
    return all(sum((d - 1) * st for d, st in zip(t.shape, t.stride())) <= _I32_MAX for t in ts)


def _num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def launch_plain_tma(a, b, out_dtype, epi, r, cfg, scale=1.0, mm_dtype=None):
    """Persistent + TMA + warp-specialized, the shape `bitequiv` uses on sm_103."""
    from triton.tools.tensor_descriptor import TensorDescriptor
    BM, BN, BK, GROUP_M, nw, ns, WS = cfg
    M, K = a.shape
    N = b.shape[1]
    mm_dtype = mm_dtype or a.dtype
    b_nk = b.stride(0) == 1 and b.stride(1) != 1
    bt = b.t() if b_nk else b
    bb = [BN, BK] if b_nk else [BK, BN]
    c = torch.empty(M, N, device=a.device, dtype=out_dtype)
    need = [(a, BK * a.element_size()), (bt, bb[1] * bt.element_size()), (c, BN * c.element_size())]
    if r is not None:
        need.append((r, BN * r.element_size()))
    if not all(_tma_ok(t, nb) for t, nb in need):
        return None
    if M % BM or N % BN:  # a TMA store past the edge traps; the plain launcher masks instead
        return None
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(bt, bb)
    c_desc = TensorDescriptor.from_tensor(c, [BM, BN])
    r_desc = TensorDescriptor.from_tensor(r if r is not None else a, [BM, BN] if r is not None else [BM, BK])
    nprog = min(_num_sms(), triton.cdiv(M, BM) * triton.cdiv(N, BN))
    with _meta_ws():
        _fused_plain_tma[(nprog, )](a_desc, b_desc, c_desc, r_desc, M, N, K, scale, EPI=epi, HAS_R=r is not None,
                                    RDT=_TL[mm_dtype], B_NK=b_nk, GROUP_M=GROUP_M, NUM_SMS=nprog, WS=WS, BM=BM, BN=BN,
                                    BK=BK, num_warps=nw, num_stages=ns)
    return c


def launch_plain(a, b, out_dtype, epi, r, cfg, scale=1.0, mm_dtype=None):
    BM, BN, BK, GROUP_M, nw, ns, _WS = cfg
    M, K = a.shape
    N = b.shape[1]
    mm_dtype = mm_dtype or a.dtype
    if b.stride(0) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    c = torch.empty(M, N, device=a.device, dtype=out_dtype)
    rm_, rn_ = (r.stride(0), r.stride(1)) if r is not None else (0, 0)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), )
    _fused_plain[grid](a, b, c, r if r is not None else a, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                       c.stride(0), c.stride(1), rm_, rn_, scale, EPI=epi, HAS_R=r is not None, RDT=_TL[mm_dtype],
                       GROUP_M=GROUP_M, EVEN_K=(K % BK == 0), I64=not _fits_i32(a, b, c), BM=BM, BN=BN, BK=BK,
                       num_warps=nw, num_stages=ns)
    return c


def launch_kpd(a, b, out_dtype, epi, r, cfg, k_per_dot, res, scale=1.0, mm_dtype=None):
    BM, BN, BK, _GROUP_M, nw, ns, _WS = cfg
    M, K = a.shape
    N = b.shape[1]
    mm_dtype = mm_dtype or a.dtype
    if b.stride(1) != 1:
        b = b.contiguous()
    c = torch.empty(M, N, device=a.device, dtype=out_dtype)
    rm_, rn_ = (r.stride(0), r.stride(1)) if r is not None else (0, 0)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), )
    _fused_kpd[grid](a, b, c, r if r is not None else a, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1), rm_, rn_, scale, k_per_dot, res, EPI=epi, HAS_R=r is not None,
                     RDT=_TL[mm_dtype], BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    return c


# --------------------------------------------------------------------------------------------
# the epilogue on its own, for the bit-exact two-kernel baseline
# --------------------------------------------------------------------------------------------
@triton.jit
def _epi_only(X, Y, R, n, EPI: tl.constexpr, HAS_R: tl.constexpr, RDT: tl.constexpr, BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < n
    x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
    r = tl.load(R + off, mask=m, other=0.0).to(tl.float32) if HAS_R else x
    tl.store(Y + off, _epilogue(x, r, EPI, HAS_R, RDT).to(Y.dtype.element_ty), mask=m)


def launch_epi(x, out_dtype, epi, r=None, BLK=2048, num_warps=8):
    """The same epilogue as one separate kernel. `x` is already the GEMM's output, so the
    leading `.to(RDT)` in `_epilogue` is a no-op here -- which is exactly the point: it is what
    makes the fused version agree with the unfused one."""
    xf = x.contiguous().view(-1)
    y = torch.empty(xf.numel(), device=x.device, dtype=out_dtype)
    rf = r.contiguous().view(-1) if r is not None else xf
    _epi_only[(triton.cdiv(xf.numel(), BLK), )](xf, y, rf, xf.numel(), EPI=epi, HAS_R=r is not None, RDT=_TL[x.dtype],
                                                BLK=BLK, num_warps=num_warps)
    return y.view(x.shape)


# (BM, BN, BK, GROUP_M, num_warps, num_stages, warp_specialize).
# BK is in the space because it is bit-free for f16/bf16 here -- the k loop still walks k
# upward into one fp32 accumulator whatever BK is -- and at small K it is the knob that
# matters most: a BK far above K wastes most of every `tl.dot`.
_BM_BN = ((128, 256), (128, 128), (256, 128), (64, 256), (128, 64), (64, 128), (64, 64))
_BK = (16, 32, 64, 128)
_WS = ((8, 3), (4, 4), (4, 3), (8, 4))

TMA_SPACE = [(bm, bn, bk, gm, nw, ns, ws)
             for bm, bn in _BM_BN
             for bk in _BK
             for gm in (8, )
             for nw, ns in _WS
             for ws in (True, False)]

PLAIN_SPACE = [(bm, bn, bk, gm, nw, ns, None) for bm, bn in _BM_BN for bk in _BK for gm in (8, 1) for nw, ns in _WS]

# At tiny K the kernel is a store with some math bolted on, not a GEMM, so the knobs that matter
# are the ones that set how many bytes are in flight: the tile area, the warp count and the stage
# count. The narrow space above was built for real GEMMs and does not reach them.
_WIDE_BM_BN = ((128, 256), (128, 128), (256, 128), (64, 256), (256, 256), (64, 128), (128, 64), (64, 64), (32, 256),
               (256, 64))
WIDE_SPACE = [(bm, bn, bk, gm, nw, ns, None)
              for bm, bn in _WIDE_BM_BN
              for bk in (16, 32, 64)
              for gm in (8, 1)
              for nw in (4, 8)
              for ns in (1, 2, 3, 4)]
