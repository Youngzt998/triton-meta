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


def _sigmoid(x):
    return torch.sigmoid(x)


def _relu_sq(x):
    # "squared ReLU". Two eager kernels, so the fp16 rounding of relu(x) is observable.
    t = torch.relu(x)
    return t * t


def _leaky(x):
    return torch.nn.functional.leaky_relu(x, 0.01)


def _hardswish(x):
    return torch.nn.functional.hardswish(x)


# ------------------------------------------------------------------------------------------
# The epilogues that follow a real model's GEMM. Which one follows which GEMM was read out of
# official modelling code -- see `fusion_moe/models_2026.py`, which is where the 419 cases and
# their epilogue names come from. Everything below is torch eager, one op per line, because THAT
# is the reference: the GEMM's output written to memory in the output dtype, read back, and the
# epilogue applied. Every line is one eager kernel, so every line rounds to the output dtype.
# ------------------------------------------------------------------------------------------


def _swiglu(x, g):
    """After an MoE expert's `up_proj`: `silu(gate_proj(h)) * up_proj(h)`. `x` is the GEMM being
    fused (up), `g` is the gate projection's output, already in memory.

    `F.silu` is NOT `x * sigmoid(x)`. torch's CUDA silu for a half dtype is `x / (1 + expf(-x))`
    evaluated in fp32 -- one correctly rounded divide, not a reciprocal and a multiply. The two
    spellings do not give the same bytes, and Inductor's own generated text agrees with torch
    here: it emits `tmp1 / (exp(-tmp1) + 1.0)`.
    """
    return torch.nn.functional.silu(g) * x


def _rweight(x, w):
    """After an MoE expert's `down_proj`: multiply by the routing weight, ONE SCALAR PER TOKEN,
    before the expert outputs are summed. `w` is [M,1], so it varies along M only."""
    return x * w


def _resid(x, r):
    """After `o_proj`: the post-attention residual add."""
    return x + r


def _lora(x, base, scale=0.25):
    """peft's `Linear.forward` after `lora_B`: the frozen layer's output plus the scaled delta.
    `scale` is alpha/rank, a python float, so it reaches the kernel as one fp32 constant."""
    return base + x * scale


def _swiglu_cw(x, g, w, lim=10.0):
    """DeepSeek-V4's `Expert.forward`: a clamped SwiGLU with the routing weight applied one GEMM
    earlier, on the SwiGLU output. The gate is clamped above only and the up projection on both
    sides, which is the clamped-SwiGLU convention `fusion_moe/fused_moe.py` records.

    `torch.clamp` KEEPS a NaN -- it tests the input for NaN before comparing. `tl.minimum` and
    `tl.maximum` lower to `min.f32`/`max.f32`, which return the non-NaN operand, so spelling the
    clamp with them is exact on every finite input and wrong on all 2046 fp16 NaN patterns.
    """
    gc = torch.clamp(g, max=lim)
    xc = torch.clamp(x, -lim, lim)
    return w * (xc * torch.nn.functional.silu(gc))


def _relu2(x):
    """Nemotron-3.5's `mlp_hidden_act`. Identical to `relu_sq` above; carried under the name the
    model table uses so a case name maps straight to an epilogue."""
    return _relu_sq(x)


def _fp8q(x, scale=2.0):
    """After a fused `qkv_proj` in an fp8-served model: scale, then cast to fp8. No clamp -- this
    is `fusion_moe/fused_moe.py`'s `EPI_FP8Q` verbatim, and the missing clamp is exactly why it
    is not the same function as `fp8cast` above."""
    return (x * scale).to(FP8)


EPILOGUES = {
    "silu": _silu,
    "gate": _gate,
    "resscale": _resscale,
    "chain": _chain,
    "fp8cast": _fp8cast,
    "rowsum": _rowsum,
    "softcap": _softcap,
    "mul3": _mul3,
    "sigmoid": _sigmoid,
    "relu_sq": _relu_sq,
    "leaky": _leaky,
    "hardswish": _hardswish,
    "bias_relu": _bias_relu,  # control: cuBLASLt can fuse this
    "gelu": _gelu,  # control: cuBLASLt can fuse this
    # the real-model epilogues, from fusion_moe/models_2026.py
    "swiglu": _swiglu,
    "rweight": _rweight,
    "resid": _resid,
    "lora": _lora,
    "swiglu_cw": _swiglu_cw,
    "relu2": _relu2,
    "fp8q": _fp8q,
}

# What each epilogue reads besides the GEMM result, IN THE ORDER `make_epi_args` produces it and
# in the order the torch function above takes it.
#
#   mn  a full [M,N] tensor
#   m1  a per-token [M,1] column, broadcast along N
#   n   a [N] row, broadcast along M
#
# The kind is what the patcher matches on: Inductor indexes an [M,N] operand with `xindex` (i.e.
# `idx_n + N*idx_m`) and an [M,1] operand with `idx_m` alone, so the two are told apart by the
# index expression and never by argument position -- which is not stable, see `bind` in
# `inductor_kernel.py`.
EPI_OPERANDS = {
    "gate": ("mn", ),
    "resscale": ("mn", ),
    "chain": ("mn", "n"),
    "mul3": ("mn", "mn"),
    "bias_relu": ("n", ),
    "swiglu": ("mn", ),
    "resid": ("mn", ),
    "lora": ("mn", ),
    "rweight": ("m1", ),
    "swiglu_cw": ("mn", "m1"),
}

# The scalars baked into an epilogue. They reach the generated kernel as fp32 constants, so a
# case measured at one value is not a case measured at another and the compile cache key has to
# carry them. Defaults are the values `fusion_moe/run_moe.py` uses: LORA_ALPHA/rank with rank 64,
# DeepSeek-V4's swiglu_limit, and its FP8_STATIC_SCALE.
EPI_PARAMS = {
    "lora": {"scale": 0.25},
    "swiglu_cw": {"lim": 10.0},
    "fp8q": {"scale": 2.0},
}

# Epilogues that read nothing but the GEMM result. Inductor's mm template epilogue is
# pointwise-only over the accumulator, so these are the ones it can fold into the template with
# no second kernel -- which is what arm 3 needs to exist at all.
X_ONLY = tuple(k for k in EPILOGUES if k not in EPI_OPERANDS and k != "rowsum")

# derived, kept because older files read them
EXTRA_MN = {k: sum(x == "mn" for x in v) for k, v in EPI_OPERANDS.items()}
EXTRA_N = {k: sum(x == "n" for x in v) for k, v in EPI_OPERANDS.items()}
EXTRA_M1 = {k: sum(x == "m1" for x in v) for k, v in EPI_OPERANDS.items()}

# cuBLASLt has a fused form for these, so a win here is not reachable-today headroom
CUBLAS_CAN_FUSE = {"bias_relu", "gelu"}

# suffixes `inductor_kernel.EPI_SRC` uses for the alternative spellings of one epilogue
KEY_SUFFIXES = ("_approx", "_helpers", "_divrn", "_cheap", "_fast")

# The suffixes that mark a spelling as PRICED-ONLY: carried to say what the exact one costs
# against, and measured NOT to be byte-exact. Every other key has to be exact, and the probes
# fail if it is not.
PRICED_SUFFIXES = ("_approx", "_cheap", "_fast")


def must_be_exact(key):
    return not key.endswith(PRICED_SUFFIXES)


# the epilogues whose output is narrower than the GEMM's
FP8_OUT = ("fp8cast", "fp8q")


def out_dtype_for(epi, dtype=torch.float16):
    return FP8 if base_key(epi) in FP8_OUT else dtype


def base_key(key):
    """`silu_approx` -> `silu`: the epilogue an `EPI_SRC` spelling is a spelling OF."""
    for suf in KEY_SUFFIXES:
        if key.endswith(suf):
            return key[:-len(suf)]
    return key


def params_for(name, override=None):
    out = dict(EPI_PARAMS.get(base_key(name), {}))
    out.update(override or {})
    return out


def params_tag(name, params):
    """A short, stable string for the scalars, so two cases that differ only in `alpha/rank` do
    not collide in the compile cache."""
    p = params_for(name, params)
    return "".join(f"_{k}{v!r}" for k, v in sorted(p.items()))


def _draw(g, shape, dtype, wide):
    """`wide` spreads the exponents over the dtype's range. A narrow gaussian keeps every value
    within a couple of binades, which hides both a changed order of additions and a rounding that
    only bites near the top or the bottom of the range."""
    if wide:
        sign = torch.where(torch.rand(*shape, generator=g, device="cuda") < 0.5, -1.0, 1.0)
        return (sign * torch.exp2(torch.rand(*shape, generator=g, device="cuda") * 20.0 - 12.0)).to(dtype)
    return (torch.randn(*shape, generator=g, device="cuda") / 4).to(dtype)


def make_epi_args(name, M, N, dtype, seed=1234, wide=False):
    """The extra operands, in `EPI_OPERANDS[name]` order.

    An `m1` operand is drawn in (0, 1) on the even draws because that is the range a normalised
    routing weight lives in, and spread over the exponent range on the odd ones.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for kind in EPI_OPERANDS.get(name, ()):
        if kind == "mn":
            out.append(_draw(g, (M, N), dtype, wide))
        elif kind == "m1":
            out.append(
                _draw(g, (M, 1), dtype, True) if wide else torch.rand(M, 1, generator=g, device="cuda").to(dtype))
        elif kind == "n":
            out.append(_draw(g, (N, ), dtype, wide))
        else:
            raise ValueError(f"unknown operand kind {kind!r} for {name}")
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

# The scale-up grid. Eight output shapes drawn from real model layers, each at six K values, so
# the sweep walks from "the GEMM is nothing, the epilogue is everything" (K=8) to "the GEMM
# dominates" (K=256) on every one of them. Every shape here was checked to have a cuBLAS plan of
# mode `plain`, which is what lets arm 3 keep arm 1's mainloop unchanged.
for _mn in ((4096, 4096), (8192, 4096), (4096, 8192), (8192, 8192), (2048, 11008), (4096, 14336), (16384, 2048),
            (2048, 32000)):
    for _k in (8, 16, 32, 64, 128, 256):
        _add("grid", _mn[0], _mn[1], _k)

ALL_SHAPES = [(g, s) for g, ss in SHAPES.items() for s in ss]
