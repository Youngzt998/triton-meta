# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 25142825
import torch
import torch.nn.functional as F

SEED = 25142825
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': False, 'persistent_reductions': None, 'cooperative_reductions': True, 'multi_kernel': None, 'combo_kernels': None, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': 1, 'assume_aligned_inputs': False, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(25142825)
    a0 = torch.randn((511, 2048), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    a1 = torch.randn((103, 1023), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    return [a0, a1]


def prog(a0, a1):
    v1 = a1.masked_fill(a1 < -1.25, -1.0)
    v2 = torch.amin(a1, dim=1, keepdim=True)
    v3 = torch.prod(a0, dim=0, keepdim=True)
    v4 = torch.sin(v3)
    v5 = -v4
    v6 = torch.amax(v5, dim=0, keepdim=False)
    return (v5, v6)
