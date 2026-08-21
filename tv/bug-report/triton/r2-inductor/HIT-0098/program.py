# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 7974433
import torch
import torch.nn.functional as F

SEED = 7974433
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': False, 'persistent_reductions': False, 'cooperative_reductions': None, 'multi_kernel': 1, 'combo_kernels': None, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': False, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(7974433)
    a0 = torch.randn((61, 512), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    return [a0]


def prog(a0):
    v1 = torch.where(a0 > a0, a0, a0)
    v2 = (v1.float() - v1.float().mean(-1, keepdim=True))
    v3 = v2.to(torch.int32)
    v4 = torch.amax(v1, dim=0, keepdim=False)
    return (v3, v4)
