# Auto-generated torch program, r2-inductor fuzzing line.
# seed = 23107642
import torch
import torch.nn.functional as F

SEED = 23107642
KNOBS = {'dynamic': False, 'backward': True, 'max_autotune': False, 'persistent_reductions': None, 'cooperative_reductions': None, 'multi_kernel': None, 'combo_kernels': None, 'prefer_nd_tiling': True, 'unroll_reductions_threshold': None, 'assume_aligned_inputs': None, 'benchmark_kernel': False}
BACKWARD = True


def make_inputs(device='cuda'):
    g = torch.Generator(device=device).manual_seed(23107642)
    a0 = torch.randn((11, 63, 1024), device=device, dtype=torch.float32, generator=g).to(torch.float16)
    a0.requires_grad_(True)
    return [a0]


def prog(a0):
    v1 = torch.std(a0.float(), dim=0, keepdim=True)
    v2 = torch.relu(v1)
    v3 = v2.masked_fill(v2 < -0.084, float('-inf'))
    v4 = F.elu(a0)
    v5 = torch.reciprocal(v4 + 2.0)
    v6 = torch.sigmoid(v5)
    v7 = v6 * v5
    v8 = (a0.float() - a0.float().mean(-1, keepdim=True))
    v9 = torch.mean(v6.float(), dim=1, keepdim=False)
    v10 = v7.reshape((48, 14784))
    v11 = (v10 != v10)
    v12 = torch.where(v11, v10, v10)
    v13 = torch.amax(v8, dim=2, keepdim=False)
    return v13.float().sum()
