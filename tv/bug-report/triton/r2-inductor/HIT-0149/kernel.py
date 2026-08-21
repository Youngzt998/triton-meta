# Harvested from TorchInductor by the r2-inductor fuzzing line.
# program seed : 3148
# source file  : cyvaykt5d4hyz4s5h5lpegw7popsvbw23nvkxib4gmzotzqrhdhc.py
# inductor name: triton_per_fused_amin_neg_sin_0
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

math = tl_math


@triton.jit
def triton_per_fused_amin_neg_sin_0_de0174a7(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 67
    r0_numel = 1
    R0_BLOCK: tl.constexpr = 32
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
    tmp0 = tl.load(in_ptr0 + (x0 + 37*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl_math.sin(tmp0)
    tmp2 = tl_math.sin(tmp1)
    tmp3 = -tmp2
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, float("inf"))
    tmp7 = triton_helpers.min2(tmp6, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0 + 37*r0_1), tmp3, r0_mask & xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
