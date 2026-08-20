# Harvested from TorchInductor by the r2-inductor fuzzing line.
# program seed : 4153
# source file  : cxmjdcukn4qniyaegrfmz2xneneeqfk6j74dk3zleyo7jjiyfrs7.py
# inductor name: triton_red_fused_cumsum_elu_0
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

math = tl_math


@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0


@triton.jit
def triton_red_fused_cumsum_elu_0_ca612596(in_out_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3 = tl.full([XBLOCK, 1], float('nan'), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4, = tl.associative_scan((tmp2,), 1, _triton_helper_fn_add0)
        tmp5 = triton_helpers.select_one((tmp4), rbase == (RBLOCK - 1), dim=-1, keep_dims=True)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp3 + tmp4
        tmp8 = tl.where(roffset > 0, tmp7, tmp4)
        tmp3 = tl.where(roffset > 0, tmp6, tmp5)
        tmp9 = tl.full([1, 1], 0.0, tl.float32)
        tmp10 = tmp8 > tmp9
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp8 * tmp11
        tmp13 = libdevice.expm1(tmp12)
        tmp14 = tmp13 * tmp11
        tmp15 = tl.where(tmp10, tmp12, tmp14)
        tl.store(in_out_ptr0 + (r0_1 + ks0*x0), tmp8, r0_mask & xmask)
        tl.store(out_ptr0 + (r0_1 + ks0*x0), tmp15, r0_mask & xmask)
