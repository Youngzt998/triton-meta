"""A small starter suite of deterministic Triton kernels.

Each entry is a KernelSpec. Kernels here avoid atomics and cross-program
reductions so that two *correct* compilations should agree bitwise. (Note:
float reassociation from optimizations -- e.g. tf32 or FMA contraction in
matmul -- can still change bits; that is an expected signal to inspect, not
automatically a bug.)

Point the runner at this file with ``--kernels eq_fuzzing/kernels/basic.py``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from eq_fuzzing.kernel_spec import KernelSpec


# --------------------------------------------------------------------------- #
# vector add
# --------------------------------------------------------------------------- #
_ADD_BLOCK = 1024


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def _add_inputs(gen, device):
    n = 4096
    return {
        "x_ptr": torch.randn(n, generator=gen, device=device),
        "y_ptr": torch.randn(n, generator=gen, device=device),
        "out_ptr": torch.empty(n, device=device),
        "n_elements": n,
    }


# --------------------------------------------------------------------------- #
# row softmax (one program per row)
# --------------------------------------------------------------------------- #
_SOFTMAX_ROWS = 128
_SOFTMAX_COLS = 500
_SOFTMAX_BLOCK = 512


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + cols, num / den, mask=mask)


def _softmax_inputs(gen, device):
    x = torch.randn(_SOFTMAX_ROWS, _SOFTMAX_COLS, generator=gen, device=device)
    return {
        "out_ptr": torch.empty_like(x),
        "in_ptr": x,
        "in_stride": _SOFTMAX_COLS,
        "out_stride": _SOFTMAX_COLS,
        "n_cols": _SOFTMAX_COLS,
    }


# --------------------------------------------------------------------------- #
# layer norm (forward, per-row)
# --------------------------------------------------------------------------- #
_LN_ROWS = 64
_LN_N = 320
_LN_BLOCK = 512


@triton.jit
def layernorm_kernel(X, Y, W, B, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    tl.store(Y + cols, xc * rstd * w + b, mask=mask)


def _ln_inputs(gen, device):
    x = torch.randn(_LN_ROWS, _LN_N, generator=gen, device=device)
    return {
        "X": x,
        "Y": torch.empty_like(x),
        "W": torch.randn(_LN_N, generator=gen, device=device),
        "B": torch.randn(_LN_N, generator=gen, device=device),
        "stride": _LN_N,
        "N": _LN_N,
        "eps": 1e-5,
    }


# --------------------------------------------------------------------------- #
# matmul (fp16 in, fp32 accum, fp16 out)
# --------------------------------------------------------------------------- #
_MM_M = _MM_N = _MM_K = 256
_MM_BM = _MM_BN = 64
_MM_BK = 32


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _mm_inputs(gen, device):
    a = torch.randn(_MM_M, _MM_K, generator=gen, device=device)
    b = torch.randn(_MM_K, _MM_N, generator=gen, device=device)
    c = torch.empty(_MM_M, _MM_N, device=device)
    return {
        "a_ptr": a, "b_ptr": b, "c_ptr": c,
        "M": _MM_M, "N": _MM_N, "K": _MM_K,
        "stride_am": a.stride(0), "stride_ak": a.stride(1),
        "stride_bk": b.stride(0), "stride_bn": b.stride(1),
        "stride_cm": c.stride(0), "stride_cn": c.stride(1),
    }


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
KERNELS = {
    "vector_add": KernelSpec(
        name="vector_add", fn=add_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": _ADD_BLOCK},
        gen_inputs=_add_inputs,
        grid=lambda inp: (triton.cdiv(inp["n_elements"], _ADD_BLOCK), 1, 1),
        out_args=["out_ptr"],
    ),
    "softmax": KernelSpec(
        name="softmax", fn=softmax_kernel,
        signature={"out_ptr": "*fp32", "in_ptr": "*fp32", "in_stride": "i32",
                   "out_stride": "i32", "n_cols": "i32"},
        constexprs={"BLOCK_SIZE": _SOFTMAX_BLOCK},
        gen_inputs=_softmax_inputs,
        grid=lambda inp: (_SOFTMAX_ROWS, 1, 1),
        out_args=["out_ptr"],
    ),
    "layer_norm": KernelSpec(
        name="layer_norm", fn=layernorm_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "W": "*fp32", "B": "*fp32",
                   "stride": "i32", "N": "i32", "eps": "fp32"},
        constexprs={"BLOCK_SIZE": _LN_BLOCK},
        gen_inputs=_ln_inputs,
        grid=lambda inp: (_LN_ROWS, 1, 1),
        out_args=["Y"],
    ),
    "matmul": KernelSpec(
        name="matmul", fn=matmul_kernel,
        signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
                   "M": "i32", "N": "i32", "K": "i32",
                   "stride_am": "i32", "stride_ak": "i32", "stride_bk": "i32",
                   "stride_bn": "i32", "stride_cm": "i32", "stride_cn": "i32"},
        constexprs={"BLOCK_M": _MM_BM, "BLOCK_N": _MM_BN, "BLOCK_K": _MM_BK},
        gen_inputs=_mm_inputs,
        grid=lambda inp: (triton.cdiv(_MM_M, _MM_BM), triton.cdiv(_MM_N, _MM_BN), 1),
        out_args=["c_ptr"],
    ),
}
