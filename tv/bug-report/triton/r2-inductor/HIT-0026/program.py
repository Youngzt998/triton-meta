# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 3096329
import torch
import torch.nn.functional as F

SEED = 3096329
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': False, 'persistent_reductions': None, 'cooperative_reductions': None, 'multi_kernel': None, 'combo_kernels': None, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(3096329)
    a0 = torch.randn((101, 17, 256), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    a1 = torch.randn((101, 17, 256), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    return [a0, a1]


def prog(a0, a1):
    v1 = torch.erf(a1)
    v2 = torch.maximum(v1, a0)
    v3 = torch.amax(v1, dim=2, keepdim=True)
    v4 = torch.trunc(a1)
    v5 = torch.sort(v4, dim=1)[0]
    v6 = torch.tanh(v5)
    v7 = (v6.float() * torch.rsqrt(v6.float().pow(2).mean(-1, keepdim=True) + 1e-6))
    v8 = torch.cat([v1, v2], dim=0)
    v9 = v4.transpose(1, 2)
    v10 = v7.flatten()
    v11 = -a1
    v12 = torch.fmod(v11, v6.abs() + 1.0)
    v13 = torch.amax(v6, dim=0, keepdim=True)
    return (v11, v12, v13)
