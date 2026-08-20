# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 34107133
import torch
import torch.nn.functional as F

SEED = 34107133
KNOBS = {'dynamic': False, 'backward': True, 'max_autotune': False, 'persistent_reductions': True, 'cooperative_reductions': True, 'multi_kernel': None, 'combo_kernels': None, 'prefer_nd_tiling': True, 'unroll_reductions_threshold': 1, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = True


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(34107133)
    a0 = torch.randn((1, 8, 31), device=device, dtype=torch.float32, generator=g).to(torch.bfloat16)
    a1 = torch.randn((1, 8, 31), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    a0.requires_grad_(True)
    a1.requires_grad_(True)
    c0 = torch.randn((31, 17), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    c1 = torch.randint(0, 31, (31, ), device=device, dtype=torch.int64, generator=g)
    return [a0, a1, c0, c1]


def prog(a0, a1, c0, c1):
    v1 = torch.atan2(a1, a0 + 1.0)
    v2 = a1.to(torch.int32)
    v3 = torch.matmul(a1, c0)
    v4 = torch.argmin(c0, dim=0, keepdim=False)
    v5 = torch.index_select(c0, 0, c1)
    v6 = torch.std(c0.float(), dim=0, keepdim=True)
    v7 = v6.to(torch.int32)
    v8 = F.pad(v1, (1, 1))
    v9 = (v8.float() * torch.rsqrt(v8.float().pow(2).mean(-1, keepdim=True) + 1e-6))
    v10 = v8.squeeze(0)
    v11 = F.mish(v6)
    return v10.float().sum()
