# Harvested from TorchInductor by the r2-inductor fuzzing line.
# program seed : 2918
# source file  : cyqex3pj5qtu5vp7bb3d2hsw2vz4svuil5rkbbabvrkc4k2gcjbi.py
# inductor name: triton_per_fused__to_copy_amax_eq_mean_sub_view_0
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

math = tl_math


@triton.jit
def triton_per_fused__to_copy_amax_eq_mean_sub_view_0_2cef5045(in_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 1
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp6 = tl.full([1, 1], 1024.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp1 - tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = tmp12 == tmp8
    tl.store(out_ptr2 + (r0_1 + 1024*x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
