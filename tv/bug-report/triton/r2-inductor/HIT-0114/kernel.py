# Harvested from TorchInductor by the r2-inductor fuzzing line.
# program seed : 1580
# source file  : ccunotlhuvhkji6znsbyg5wobk2c45xp24csdvwk2wwr2k343wrk.py
# inductor name: triton_per_fused__to_copy_abs_add_atan2_neg_pow_std_2
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

math = tl_math


@triton.jit
def triton_per_fused__to_copy_abs_add_atan2_neg_pow_std_2_4fa5bb1f(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 37
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = tl.full([1, 1], 0.5, tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp3
    tmp5 = tl.full([1, 1], 1.0, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.atan2(tmp0, tmp6)
    tmp8 = -tmp7
    tmp9 = tmp0.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = (tl.full([1, 1], 130, tl.int32)).to(tl.float32)
    tmp18 = (tmp16 / tmp17)
    tmp19 = tmp10 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = tl.full([1, 1], 129.0, tl.float32)
    tmp26 = (tmp24 / tmp25)
    tmp27 = tl.sqrt_rn(tmp26)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp8, r0_mask)
    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp27, None)
