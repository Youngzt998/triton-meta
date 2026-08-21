# Harvested from TorchInductor by the r2-inductor fuzzing line.
# program seed : 391
# source file  : cjoixwsaersogz3sjzu4hcuumubcvsmthugek63fydspj77wcodo.py
# inductor name: triton_red_fused_abs_add_amax_fmod_neg_tanh_1
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

math = tl_math


@triton.jit
def triton_red_fused_abs_add_amax_fmod_neg_tanh_1_05dfb1e9(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 101
    r0_numel = 71
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 4352*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (x0 + 4352*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -tmp0
        tmp3 = libdevice.tanh(tmp2)
        tmp4 = tl_math.abs(tmp3)
        tmp5 = tl.full([1, 1], 1.0, tl.float32)
        tmp6 = tmp4 + tmp5
        tmp7 = libdevice.fmod(tmp1, tmp6)
        tmp8 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp10 = triton_helpers.maximum(_tmp9, tmp8)
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
        tl.store(out_ptr0 + (x0 + 4352*r0_1), tmp1, r0_mask & xmask)
        tl.store(out_ptr1 + (x0 + 4352*r0_1), tmp7, r0_mask & xmask)
    tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp9, xmask)
