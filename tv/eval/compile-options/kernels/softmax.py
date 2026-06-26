"""
Self-contained softmax kernel for compile-options / permutation generation.

Imports only triton (no torch), so it can be compiled on a box without torch.
One program handles one row: load the row, subtract its max, exponentiate, divide
by the sum. Exercises reductions (tt.reduce: max and sum), exp, and broadcast —
the new ops the validator must model beyond the elementwise add kernel.
"""

import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    in_ptrs = in_ptr + row * in_row_stride + cols
    x = tl.load(in_ptrs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    out_ptrs = out_ptr + row * out_row_stride + cols
    tl.store(out_ptrs, y, mask=mask)
