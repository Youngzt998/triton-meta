# Harvested from TorchInductor by the r2-inductor fuzzing line.
# program seed : 59
# source file  : cwjkywhqf3kopz3ajd6xiydjo6sfewb7m7tyid23fcs74tkqajer.py
# inductor name: triton_red_fused_abs_add_all_amax_eq_ge_mean_mul_ne_neg_pow_scalar_tensor_sub_unsqueeze_where_2
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

math = tl_math


@triton.jit
def triton_red_fused_abs_add_all_amax_eq_ge_mean_mul_ne_neg_pow_scalar_tensor_sub_unsqueeze_where_2_3551b7e9(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 28
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    _tmp31 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([1, 1], 1024.0, tl.float32)
        tmp3 = (tmp1 / tmp2)
        tmp4 = tmp0 - tmp3
        tmp6 = tl_math.abs(tmp5)
        tmp7 = tl.full([1, 1], 0.5, tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tmp4 * tmp9
        tmp11 = tmp10 == tmp10
        tmp12 = tl_math.abs(tmp10)
        tmp13 = tl.full([1, 1], float("inf"), tl.float32)
        tmp14 = tmp12 != tmp13
        tmp15 = tmp11 & tmp14
        tmp16 = tmp15 == 0
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 | tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
        tmp21 = tl.full([1, 1], 0.0, tl.float32)
        tmp22 = tmp9 >= tmp21
        tmp23 = tl.full([1, 1], 1.0, tl.float32)
        tmp24 = tl.full([1, 1], -1.0, tl.float32)
        tmp25 = tl.where(tmp22, tmp23, tmp24)
        tmp26 = tmp4 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(r0_mask, tmp29, _tmp28)
        tmp30 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp32 = triton_helpers.maximum(_tmp31, tmp30)
        _tmp31 = tl.where(r0_mask, tmp32, _tmp31)
    tmp20 = _tmp18.to(tl.int8)
    tmp18 = triton_helpers.any(tmp20, 1)[:, None]
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    tmp31 = triton_helpers.max2(_tmp31, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp31, None)
