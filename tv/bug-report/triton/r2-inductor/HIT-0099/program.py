# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 7998190
import torch
import torch.nn.functional as F

SEED = 7998190
KNOBS = {'dynamic': False, 'backward': False, 'max_autotune': False, 'persistent_reductions': True, 'cooperative_reductions': True, 'multi_kernel': None, 'combo_kernels': None, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(7998190)
    a0 = torch.randn((64, 1024), device=device, dtype=torch.float32, generator=g).to(torch.bfloat16)
    c0 = torch.randn((1024, 2048), device=device, dtype=torch.float32, generator=g).to(torch.bfloat16)
    c1 = torch.randn((2048, ), device=device, dtype=torch.float32, generator=g).to(torch.bfloat16)
    c2 = torch.randn((2048, ), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    c3 = torch.randn((2048, ), device=device, dtype=torch.float32, generator=g).to(torch.float32)
    return [a0, c0, c1, c2, c3]


def prog(a0, c0, c1, c2, c3):
    v1 = -a0
    v2 = torch.addmm(c1, v1, c0)
    v3 = F.layer_norm(c0.float(), (2048, ), c2, c3, 1e-5)
    v4 = torch.tanh(a0)
    v5 = v3.to(torch.int32)
    v6 = torch.linalg.vector_norm(v3.float(), dim=1, keepdim=False)
    return (v4, v5, v6)
