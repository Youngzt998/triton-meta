# Harvested from TorchInductor by the r2-inductor fuzzing line.
# program seed : 4307
# source file  : cbcz4wfmvwc4hcqilwnhs64zay7yyuhnvpuofwo73ymosu2cu224.py
# inductor name: triton_poi_fused__to_copy_add_constant_pad_nd_eq_expand_masked_fill_mul_neg_reciprocal_unsqueeze_0
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

math = tl_math


@triton.jit
def triton_poi_fused__to_copy_add_constant_pad_nd_eq_expand_masked_fill_mul_neg_reciprocal_unsqueeze_0_1c641992(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1
    xnumel = 63
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp9 = tl.load(in_ptr1 + (x1 + 31*y0), xmask & ymask).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (x1 + 31*y0), xmask & ymask)
    tmp0 = (1 + x1).to(tl.int32)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 33, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (0))
    tmp7 = tl.broadcast_to(tmp6, [1, 1])
    tmp8 = tl.where(tmp5, tmp7, 0.0)
    tmp10 = tl.full([1, 1], 1.0, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp8 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp11 * tmp11
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.full([1, 1], 0.0, tl.float32)
    tmp20 = tmp18 == tmp19
    tmp21 = (tmp10 / tmp18)
    tmp22 = tl.where(tmp20, tmp19, tmp21)
    tmp23 = tmp13 * tmp22
    tmp24 = -tmp14
    tmp25 = tmp8 * tmp24
    tmp26 = tmp25 * tmp22
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 31*y0), tmp23, xmask & ymask)
    tl.store(out_ptr1 + (x1 + 31*y0), tmp27, xmask & ymask)
