# Harvested from TorchInductor by the r2-inductor fuzzing line.
# program seed : 3175
# source file  : ck33yq2xcjgxtutyuhmpan6vxzfzgbpotx73pi7nsh4icuoog52q.py
# inductor name: triton_per_fused_amax_neg_prod_sin_1
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

math = tl_math


@triton.jit
def triton_per_fused_amax_neg_prod_sin_1_6e963cf2(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 1
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 1)
    tmp4 = triton_helpers.prod(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = tl_math.sin(tmp4)
    tmp6 = -tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
