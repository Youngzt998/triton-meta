"""The shapes and the epilogues the sweep covers.

An epilogue is a plain torch function of the GEMM result plus zero or more extra [M,N] tensors
or scalars. The interesting ones are the ones cuBLASLt cannot express -- its whole list is bias,
relu, gelu and their backward and bias-gradient forms -- so `bias_relu` and `gelu` are here only
as controls: cuBLAS CAN fuse those, so a Triton win on them is not a win over what a user can
actually reach today.
"""
from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn


def _silu(x):
    return x * torch.sigmoid(x)


def _gate(x, g):
    return x * torch.sigmoid(g)


def _resscale(x, r):
    return (x + r) * 1.7


def _chain(x, r, b):
    # four ops cuBLASLt has no fused form for
    return torch.tanh(x * 0.5 + b) * (x + r)


def _fp8cast(x):
    return (x * 0.375).clamp(-448.0, 448.0).to(FP8)


def _rowsum(x):
    # a reduction after the GEMM: an mm template epilogue is pointwise-only, so Inductor cannot
    # fold this into the template. Included to bound the claim, not because it is expected to win.
    return x.sum(dim=1)


def _bias_relu(x, b):
    return torch.relu(x + b)


def _gelu(x):
    return torch.nn.functional.gelu(x)


def _softcap(x):
    return torch.tanh(x * 0.0625) * 16.0


def _mul3(x, r, g):
    return x * r * torch.sigmoid(g)


EPILOGUES = {
    "silu": _silu,
    "gate": _gate,
    "resscale": _resscale,
    "chain": _chain,
    "fp8cast": _fp8cast,
    "rowsum": _rowsum,
    "softcap": _softcap,
    "mul3": _mul3,
    "bias_relu": _bias_relu,  # control: cuBLASLt can fuse this
    "gelu": _gelu,  # control: cuBLASLt can fuse this
}

# how many extra [M,N] tensors each epilogue takes, and whether one of them is a [N] bias
EXTRA_MN = {"gate": 1, "resscale": 1, "chain": 1, "mul3": 2}
EXTRA_N = {"chain": 1, "bias_relu": 1}

# cuBLASLt has a fused form for these, so a win here is not reachable-today headroom
CUBLAS_CAN_FUSE = {"bias_relu", "gelu"}


def make_epi_args(name, M, N, dtype, seed=1234):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(EXTRA_MN.get(name, 0)):
        out.append((torch.randn(M, N, generator=g, device="cuda") / 4).to(dtype))
    for _ in range(EXTRA_N.get(name, 0)):
        out.append((torch.randn(N, generator=g, device="cuda") / 4).to(dtype))
    return tuple(out)


# ------------------------------------------------------------------------------------------
# shapes. K is varied hard on purpose: the fusion win should grow as K shrinks, because the
# epilogue's bytes are a fixed M*N while the GEMM's work is M*N*K.
# ------------------------------------------------------------------------------------------

SHAPES = {}


def _add(group, M, N, K):
    SHAPES.setdefault(group, []).append((M, N, K))


# K sweep on a square-ish output
for _k in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
    _add("ksweep", 4096, 4096, _k)
# K sweep on a bigger output
for _k in (64, 256, 1024, 4096):
    _add("ksweep_big", 8192, 8192, _k)
# LoRA-like: very small K, tall and wide
for _k in (8, 16, 32, 64):
    _add("lora", 8192, 4096, _k)
    _add("lora", 2048, 11008, _k)
# lm_head-like: huge N
_add("lmhead", 2048, 32000, 4096)
_add("lmhead", 512, 32000, 4096)
_add("lmhead", 2048, 128256, 2048)
_add("lmhead", 4096, 32000, 1024)
# MLP-ish shapes from real models
_add("mlp", 8192, 11008, 4096)
_add("mlp", 4096, 14336, 4096)
_add("mlp", 8192, 3584, 4096)
_add("mlp", 16384, 4096, 1024)
_add("mlp", 32768, 2048, 512)
# thin M (prefill tail / small batch)
for _m in (16, 64, 256, 1024):
    _add("thinm", _m, 8192, 4096)
    _add("thinm", _m, 8192, 512)
# decode-ish: M very small. The bit-exact plan for these is a gemv mode, which needs two
# kernels, so they cannot be carried to step 2 -- but they still bound the step-1 claim.
for _m in (1, 4, 8):
    _add("decode", _m, 8192, 4096)

ALL_SHAPES = [(g, s) for g, ss in SHAPES.items() for s in ss]
