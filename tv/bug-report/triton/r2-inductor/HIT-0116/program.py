# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 12789185
import torch
import torch.nn.functional as F

SEED = 12789185
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': False, 'persistent_reductions': None, 'cooperative_reductions': None, 'multi_kernel': None, 'combo_kernels': True, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(12789185)
    a0 = torch.randn((513, 33), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    a1 = torch.randn((513, 33), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    a2 = torch.randn((513, 33), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    return [a0, a1, a2]


def prog(a0, a1, a2):
    v1 = a2 - a0
    v2 = torch.prod(a2, dim=0, keepdim=True)
    v3 = torch.trunc(a2)
    v4 = v3.flatten()
    v5 = v1 + a2
    v6 = torch.abs(v2)
    return (v6, )
