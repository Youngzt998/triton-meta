"""The `split` cuBLAS plan mode with a pointwise epilogue folded in before the store.

`split` is already two kernels.  Pass 1 sums each `k_chunk`-long slice of K into its own fp32
partial; pass 2 adds the S partials in forward order starting from partial 0, scales once and
casts once.  The epilogue goes at the END OF PASS 2 and nowhere else, so pass 1 is the shipped
kernel, imported rather than copied, and the sum is untouched.

What fusing buys here is one whole round trip of the [M, N] output: the unfused path has pass 2
write the result to memory in the output dtype and a third kernel read it back.  What it cannot
buy is the workspace -- the S fp32 partials still have to be written and read, because that is
what the plan says cuBLAS does.

Rounding.  Pass 2's `acc.to(out_dtype)` is the round the unfused path performs when it stores,
so the epilogue text's leading `.to(<out dtype>).to(tl.float32)` reproduces it exactly, the same
way it does in `fused_plain`.  See that file's docstring for the two traps -- the NaN-keeping
clamp, and `enable_fp_fusion=False` -- which apply here unchanged.

TWO KERNEL SHAPES
-----------------
`pre`    `kernels._splitk_partial` + `kernels._splitk_combine`: the pair as it was before the
         sm_103 rewrites.  int64 addressing, `_tile`-sized pass 1, a combine on the same BM x BN
         tile grid as the GEMM.
`gb300`  `kernels._splitk_partial_fast` (int32 addressing) sized by `kernels._split_tile`, and a
         combine sized by how many bytes have to move rather than by the GEMM's tile grid.

The GB300 combine here differs from the shipped `_splitk_combine_flat` in its GRID only: that one
walks the output as one flat vector, this one as (row, column block).  A flat program would have
to divide by N to find its row, and an epilogue with a per-token [M, 1] operand needs the row.
The program count is the same, the arithmetic per output element is the same -- partial 0 is the
starting value, 1..S-1 are added onto it forward, one scale, one cast -- and a grid cannot move a
bit.
"""
from __future__ import annotations

import torch
import triton

from .fused_plain import (OPERAND_KINDS, _decl, _HEAD, _import_src, _kcontig, _loads, _NO_FP_FUSION, round_lines,
                          config_str, operand_strides)
from .kernels import _split_tile, _splitk_partial, _splitk_partial_fast, _tile
from .ltapi import DEVICE

__all__ = [
    "build", "launch_pre", "launch_gb300", "config_space", "config_str", "DEFAULT_PRE_CONFIG", "COMBINE_BLOCK",
    "OPERAND_KINDS"
]

# --------------------------------------------------------------------------------------------
# pass 2, pre-sm_103 shape: `kernels._splitk_combine` with the epilogue folded in
# --------------------------------------------------------------------------------------------
_PRE_SRC = '''

@triton.jit
def combine_pre(W, C, {ptrs}M, N, ws, wm, wn, cm, cn, {strides}S, s,
                BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    mask = (ocm[:, None] < M) & (ocn[None, :] < N)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for i in range(0, S):
        p = tl.load(W + i.to(tl.int64) * ws + ocm[:, None] * wm + ocn[None, :] * wn, mask=mask, other=0.0)
        acc = tl.where(i == 0, p, acc + p)
    acc = acc * s
{loads}{body}
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], tmp_out, mask=mask)
'''

# --------------------------------------------------------------------------------------------
# pass 2, GB300 shape: `kernels._splitk_combine_flat`'s arithmetic on a (row, column block) grid
# --------------------------------------------------------------------------------------------
_GB300_SRC = '''

@triton.jit
def combine_gb300(W, C, {ptrs}M, N, ws, {strides}S, s, BLK: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    n = (tl.program_id(1) * BLK + tl.arange(0, BLK)).to(tl.int64)
    mask = n < N
    off = m * N + n
    acc = tl.zeros((BLK, ), dtype=tl.float32)
    for i in range(0, S):
        p = tl.load(W + i.to(tl.int64) * ws + off, mask=mask, other=0.0)
        acc = tl.where(i == 0, p, acc + p)
    acc = acc * s
{loads}{body}
    tl.store(C + off, tmp_out, mask=mask)
'''

COMBINE_BLOCK = 512  # output elements per combine program, as `kernels._COMBINE_BLOCK`

# Pass 1 is not searched: its tile is fixed by the mode's own launcher, exactly as it ships.
DEFAULT_PRE_CONFIG = {"BK": 64, "num_warps": 4, "num_stages": 3, "BMC": 128, "BNC": 128}


class FusedSplit:
    """The two compiled pass-2 kernels for one (epilogue, operand list, output dtype)."""

    def __init__(self, mod, path, n_extra):
        self.combine_pre = mod.combine_pre
        self.combine_gb300 = mod.combine_gb300
        self.path = path
        self.n_extra = n_extra


_BUILT: dict[tuple, FusedSplit] = {}


def build(epi_lines, operands, round_dtype, cache_dir, tag="epi"):
    """Generate and import the two pass-2 kernels.  Same contract as `fused_plain.build`."""
    for k in operands:
        if k not in OPERAND_KINDS:
            raise ValueError(f"unknown operand kind {k!r}")
    body = round_lines(epi_lines, round_dtype)
    n = len(operands)
    ptrs, strides = _decl(n)
    src = _HEAD
    src += _PRE_SRC.format(ptrs=ptrs, strides=strides, loads=_loads(n, "ocm[:, None]", "ocn[None, :]", "mask"),
                           body="\n".join("    " + ln for ln in body))
    src += _GB300_SRC.format(ptrs=ptrs, strides=strides, loads=_loads(n, "m", "n", "mask"),
                             body="\n".join("    " + ln for ln in body))
    hit = _BUILT.get((src, ))
    if hit is not None:
        return hit
    mod, path = _import_src(src, f"fused_split_{tag}", cache_dir)
    built = _BUILT[(src, )] = FusedSplit(mod, path, n)
    return built


def _extra_args(kinds, extras):
    ptrs = list(extras)
    strides = []
    for k, t in zip(kinds, extras):
        strides += list(operand_strides(k, t))
    return ptrs, strides


def _nsplit(K, chunk):
    return (K + chunk - 1) // chunk


def launch_pre(kern, a, b, out_dtype, kinds, extras, plan, cfg, scale=1.0, fp8_min_bm=64):
    """`kernels._triton_splitk`'s pre-sm_103 body, with the epilogue folded into pass 2.

    Pass 1 is `_splitk_partial` imported from `kernels`, at the tile `_tile` picks -- not a copy
    of it, so it cannot drift.  Only the combine is ours.
    """
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    S = _nsplit(K, plan.k_chunk)
    if S < 2 or S > 8192:
        return None
    BM, BN = _tile(M, N, a.dtype, fp8_min_bm)
    ntile = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    w = torch.empty(S, M, N, device=DEVICE, dtype=torch.float32)
    _splitk_partial[(ntile, S)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), w.stride(0),
                                w.stride(1), w.stride(2), plan.k_chunk, BM=BM, BN=BN, BK=cfg["BK"],
                                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    BMC, BNC = cfg["BMC"], cfg["BNC"]
    ptrs, strides = _extra_args(kinds, extras)
    kern.combine_pre[(triton.cdiv(M, BMC) * triton.cdiv(N, BNC), )](w, c, *ptrs, M, N, w.stride(0), w.stride(1),
                                                                    w.stride(2), c.stride(0), c.stride(1), *strides, S,
                                                                    scale, BM=BMC, BN=BNC, num_warps=4,
                                                                    enable_fp_fusion=_NO_FP_FUSION)
    return c


def launch_gb300(kern, a, b, out_dtype, kinds, extras, plan, scale=1.0, fp8_min_bm=64, BK=64):
    """`kernels._triton_splitk_fast`'s body, with the epilogue folded into pass 2.

    Pass 1 and its fitted tile are the shipped ones; only pass 2 changes.  Returns None when the
    chunk does not yield a real split, as the shipped launcher does.
    """
    from .arch import platform
    from .kernels import _split_vectorises
    M, K = a.shape
    N = b.shape[1]
    S = _nsplit(K, plan.k_chunk)
    if S < 2 or S > 8192:
        return None
    elsize = a.element_size()
    min_bm = fp8_min_bm if a.dtype == torch.float8_e4m3fn else 16
    vectorises = _split_vectorises(a, b, N)
    BM, BN, num_warps, num_stages = _split_tile(M, N, K, S, elsize, min_bm, platform().sm_count, BK, vectorises)
    ntile = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    w = torch.empty(S, M, N, device=DEVICE, dtype=torch.float32)
    fits32 = max(M * K, K * N, S * M * N) < 2**31
    pass1 = _splitk_partial_fast if (fits32 and vectorises) else _splitk_partial
    pass1[(ntile, S)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), w.stride(0), w.stride(1),
                      w.stride(2), plan.k_chunk, BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages)
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    ptrs, strides = _extra_args(kinds, extras)
    kern.combine_gb300[(M, triton.cdiv(N,
                                       COMBINE_BLOCK))](w, c, *ptrs, M, N, w.stride(0), *strides, S, scale,
                                                        BLK=COMBINE_BLOCK, num_warps=4, enable_fp_fusion=_NO_FP_FUSION)
    return c


def config_space(M, N, K, kind, plan):
    """What the pre-sm_103 split arm searches over: pass 1's k step, warps and depth, and the
    combine's tile.  The chunking is the plan's and is not in the space."""
    out = []
    for BK in (32, 64, 128):
        for nw in (4, 8):
            for ns in (2, 3, 4):
                for BMC, BNC in ((128, 128), (64, 128), (128, 64), (64, 64), (32, 256), (256, 32)):
                    if BMC > max(16, 2 * triton.next_power_of_2(M)) or BNC > max(16, 2 * triton.next_power_of_2(N)):
                        continue
                    out.append({"BK": BK, "num_warps": nw, "num_stages": ns, "BMC": BMC, "BNC": BNC})
    d = DEFAULT_PRE_CONFIG
    return [d] + [c for c in out if c != d]
