# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 33473613
import torch
import torch.nn.functional as F

SEED = 33473613
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': False, 'persistent_reductions': None, 'cooperative_reductions': None, 'multi_kernel': None, 'combo_kernels': True, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': 1, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(33473613)
    a0 = torch.randn((16, ), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    a1 = torch.randn((1024, ), device=device, dtype=torch.float32, generator=g).to(torch.bfloat16)
    return [a0, a1]


def prog(a0, a1):
    v1 = torch.std(a1.float(), dim=0, keepdim=False)
    v2 = a1 * v1 + a1
    v3 = torch.where(v1 > v1, v1, v1)
    v4 = torch.mean(a1.float(), dim=0, keepdim=False)
    v5 = F.elu(v4)
    v6 = torch.flip(a0, [0])
    v7 = torch.amax(v6, dim=0, keepdim=False)
    return (v5, v6, v7)
