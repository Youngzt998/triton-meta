# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 21785169
import torch
import torch.nn.functional as F

SEED = 21785169
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': False, 'persistent_reductions': True, 'cooperative_reductions': True, 'multi_kernel': 1, 'combo_kernels': None, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(21785169)
    a0 = torch.randn((71, ), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    a1 = torch.randn((71, ), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    c0 = torch.randn((71, 3), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    c1 = torch.randn((3, ), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    return [a0, a1, c0, c1]


def prog(a0, a1, c0, c1):
    v1 = torch.exp(a1)
    v2 = torch.cat([a1, v1], dim=0)
    v3 = a1.unsqueeze(0)
    v4 = v1.narrow(0, 20, 50)
    v5 = v3 * 0.1849
    v6 = F.hardtanh(a0)
    v7 = v2 * v2
    v8 = torch.addmm(c1, v3, c0)
    v9 = (c0 < c1)
    v10 = torch.where(v9, c0, c1)
    v11 = torch.ceil(c0)
    v12 = (v8.float() - v8.float().mean(-1, keepdim=True))
    v13 = torch.clamp(v10, -3.0, 3.0)
    v14 = torch.prod(v3, dim=0, keepdim=False)
    v15 = torch.amin(v3, dim=1, keepdim=True)
    return (v14, v15)
