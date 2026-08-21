# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 24929012
import torch
import torch.nn.functional as F

SEED = 24929012
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': False, 'persistent_reductions': True, 'cooperative_reductions': None, 'multi_kernel': None, 'combo_kernels': None, 'prefer_nd_tiling': True, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(24929012)
    a0 = torch.randn((17, 13), device=device, dtype=torch.float32, generator=g).to(torch.bfloat16)
    a1 = torch.randn((17, 13), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    c0 = torch.randn((13, 37), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    return [a0, a1, c0]


def prog(a0, a1, c0):
    v1 = torch.matmul(a1, c0)
    v2 = torch.sin(v1)
    v3 = F.gelu(c0)
    v4 = torch.sin(v2)
    v5 = torch.sort(v4, dim=1)[0]
    v6 = torch.where(v2 > v1, v2, v1)
    v7 = torch.amax(v5, dim=0, keepdim=False)
    v8 = v5 * 0.5415
    v9 = v5.to(torch.int32)
    v10 = torch.amin(v2, dim=0, keepdim=False)
    v11 = -v4
    v12 = torch.sum(v10, dim=0, keepdim=False)
    return (v11, v12)
