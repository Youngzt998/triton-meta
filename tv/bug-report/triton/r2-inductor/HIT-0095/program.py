# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 6636122
import torch
import torch.nn.functional as F

SEED = 6636122
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': False, 'persistent_reductions': True, 'cooperative_reductions': None, 'multi_kernel': 1, 'combo_kernels': None, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(6636122)
    a0 = torch.randn((512, 17), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    a1 = torch.randn((512, 17), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    a2 = torch.randn((512, 17), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    c0 = torch.randn((17, 64), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    c1 = torch.randn((64, 65), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    c2 = torch.randn((65, 15), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    return [a0, a1, a2, c0, c1, c2]


def prog(a0, a1, a2, c0, c1, c2):
    v1 = torch.mm(a1, c0)
    v2 = a0.to(torch.int32)
    v3 = torch.sin(a0)
    v4 = v1 * -2.2887
    v5 = torch.var(v3.float(), dim=1, keepdim=False)
    v6 = torch.ceil(c0)
    v7 = torch.matmul(v6, c1)
    v8 = c0.narrow(1, 7, 25)
    v9 = torch.where(v8 > v8, v8, v8)
    v10 = torch.matmul(v7, c2)
    v11 = (v5 <= v5)
    v12 = torch.where(v11, v5, v5)
    v13 = v6 - c0
    v14 = torch.amax(c0, dim=0, keepdim=False)
    return (v12, v13, v14)
