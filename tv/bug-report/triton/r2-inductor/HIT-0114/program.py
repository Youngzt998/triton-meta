# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 12512020
import torch
import torch.nn.functional as F

SEED = 12512020
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': True, 'persistent_reductions': None, 'cooperative_reductions': None, 'multi_kernel': None, 'combo_kernels': True, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(12512020)
    a0 = torch.randn((65, ), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    a1 = torch.randn((65, ), device=device, dtype=torch.float32, generator=g).to(torch.bfloat16)
    c0 = torch.randint(0, 65, (65, ), device=device, dtype=torch.int64, generator=g)
    return [a0, a1, c0]


def prog(a0, a1, c0):
    v1 = torch.cumsum(a1.float(), dim=0)
    v2 = torch.minimum(a0, a0)
    v3 = torch.index_select(a1, 0, c0)
    v4 = torch.log(a0.abs() + 1.0)
    v5 = torch.var(a0.float(), dim=0, keepdim=True)
    v6 = torch.cat([v3, a1], dim=0)
    v7 = torch.atan2(a1, a1 + 1.0)
    v8 = torch.pow(v6.abs() + 0.5, 2)
    v9 = torch.atan2(v6, v8 + 1.0)
    v10 = torch.addcmul(a1, a1, v1, value=0.5)
    v11 = v8 * v8 + v8
    v12 = torch.amax(v11, dim=0, keepdim=True)
    v13 = torch.sort(v7, dim=0)[0]
    v14 = -v9
    v15 = torch.std(v6.float(), dim=0, keepdim=False)
    return (v13, v14, v15)
