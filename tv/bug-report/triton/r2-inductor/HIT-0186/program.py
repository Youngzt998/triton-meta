# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 467221
import torch
import torch.nn.functional as F

SEED = 467221
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': False, 'persistent_reductions': None, 'cooperative_reductions': None, 'multi_kernel': 1, 'combo_kernels': None, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(467221)
    a0 = torch.randn((32, 1024), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    a1 = torch.randn((1, 1), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    a2 = torch.randn((32, 1024), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    c0 = torch.randn((1024, ), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    c1 = torch.randn((1024, ), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    return [a0, a1, a2, c0, c1]


def prog(a0, a1, a2, c0, c1):
    v1 = torch.pow(a0.abs() + 0.5, 2)
    v2 = a0.unsqueeze(1)
    v3 = v1 * v2
    v4 = (v2.float() - v2.float().mean(-1, keepdim=True))
    v5 = torch.ceil(v1)
    v6 = (v1 < v4)
    v7 = torch.where(v6, v1, v4)
    v8 = v1 * v4
    v9 = F.layer_norm(v1.float(), (1024, ), c0, c1, 1e-5)
    v10 = torch.minimum(v3, a1)
    v11 = F.softmax(v8.float(), dim=0)
    return (v11, )
