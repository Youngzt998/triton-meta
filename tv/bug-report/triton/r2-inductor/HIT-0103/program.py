# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 9859155
import torch
import torch.nn.functional as F

SEED = 9859155
KNOBS = {'dynamic': True, 'backward': False, 'max_autotune': False, 'persistent_reductions': None, 'cooperative_reductions': None, 'multi_kernel': 1, 'combo_kernels': None, 'prefer_nd_tiling': None, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = False


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(9859155)
    a0 = torch.randn((65537, ), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    return [a0]


def prog(a0):
    v1 = torch.sign(a0)
    v2 = torch.logsumexp(a0.float(), dim=0, keepdim=False)
    v3 = torch.amax(v1, dim=0, keepdim=False)
    v4 = torch.cummax(v1, dim=0)[0]
    v5 = torch.logsumexp(v4.float(), dim=0, keepdim=False)
    v6 = F.mish(v2)
    v7 = torch.std(v4.float(), dim=0, keepdim=False)
    v8 = v7 * 0.1012
    v9 = torch.fmod(v6, v3.abs() + 1.0)
    v10 = v8.to(torch.bfloat16)
    v11 = v3.contiguous()
    v12 = torch.floor(v4)
    v13 = torch.trunc(v6)
    v14 = F.log_softmax(v12.float(), dim=0)
    return (v13, v14)
