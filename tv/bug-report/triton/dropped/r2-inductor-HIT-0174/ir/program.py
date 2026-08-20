# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 32887607
import torch
import torch.nn.functional as F

SEED = 32887607
KNOBS = {'dynamic': True, 'backward': False, 'max_autotune': False, 'persistent_reductions': None, 'cooperative_reductions': None, 'multi_kernel': None, 'combo_kernels': True, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(32887607)
    a0 = torch.randn((13, 7), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    a1 = torch.randn((13, 7), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    a2 = torch.randn((13, 7), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    a3 = torch.randn((13, 7), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    c0 = torch.randn((7, 103), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    return [a0, a1, a2, a3, c0]


def prog(a0, a1, a2, a3, c0):
    v1 = torch.cummax(a0, dim=1)[0]
    v2 = torch.mm(a2, c0)
    v3 = c0 * c0 + c0
    v4 = torch.cumsum(v2.float(), dim=1)
    v5 = torch.sort(v4, dim=0)[0]
    v6 = F.elu(v4)
    return (v5, v6)
