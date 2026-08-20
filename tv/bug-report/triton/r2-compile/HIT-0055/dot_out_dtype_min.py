#!/usr/bin/env python
"""Minimal repro for r2-compile/HIT-0055 -- hand written, 40 lines, no fuzzer.

`tl.dot(a, b, out_dtype=D)` with FLOAT operands and NO explicit accumulator
builds the implicit zero accumulator with the WRONG element type whenever D is
neither f16 nor f32.  `python/triton/language/semantic.py`, `Semantic.dot`:

    else:                                                        # line 1510
        _0 = self.builder.get_fp16(0) if out_dtype.is_fp16() \
             else self.builder.get_fp32(0)                       # line 1511
        ret_scalar_ty = out_dtype                                # line 1512
    ...
    ret_ty = tl.block_type(ret_scalar_ty, [M, N])                # line 1518
    if acc is None:
        acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder), _0)   # 1520

`_0` is an f32 zero while `ret_ty`'s element type is `out_dtype`, so `tt.splat`
gets an f32 operand and an i32 (or f64) result:

    %434 = "tt.splat"(%433) : (f32) -> tensor<64x64xi32>
    error: 'tt.splat' op requires the same element type for all operands and results

Needs no GPU.  Run:

    source /home/youngzt/fuzz/r2-compile/env.sh
    python dot_out_dtype_min.py
"""
import os
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend


@triton.jit
def k(A, B, C, OUT: tl.constexpr):
    m = tl.arange(0, 32)[:, None]
    n = tl.arange(0, 32)[None, :]
    kk = tl.arange(0, 32)
    a = tl.load(A + m * 32 + kk[None, :])
    b = tl.load(B + kk[:, None] * 32 + n)
    tl.store(C + m * 32 + n, tl.dot(a, b, out_dtype=OUT))   # acc is None


def build(arch, ptr_ty, out_dtype, c_ty):
    target = GPUTarget("cuda", arch, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    src = ASTSource(fn=k, attrs=None, constexprs={"OUT": out_dtype},
                    signature={"A": ptr_ty, "B": ptr_ty, "C": c_ty, "OUT": "constexpr"})
    # MLIR prints the verifier error and the whole failing module from C++, so
    # it bypasses sys.stdout / sys.stderr.  Redirect the real fds 1 and 2.
    keep1, keep2 = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+")
    try:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        try:
            src.make_ir(target, options, backend.get_codegen_implementation(options),
                        backend.get_module_map(), ctx)
            out = "OK"
        except Exception as e:
            out = "FAIL %s" % type(e).__name__
    finally:
        os.dup2(keep1, 1)
        os.dup2(keep2, 2)
        os.close(keep1)
        os.close(keep2)
    if out.startswith("FAIL"):
        tmp.seek(0)
        first = next((ln for ln in tmp.read().splitlines() if "error:" in ln), "")
        out += " | " + first.strip()[:92]
    tmp.close()
    return out


CASES = [
    ("*fp16", tl.int32, "*i32", "fp16 x fp16, out_dtype=int32"),
    ("*fp8e5", tl.int32, "*i32", "fp8e5 x fp8e5, out_dtype=int32   (the drawn case)"),
    ("*fp16", tl.float64, "*fp32", "fp16 x fp16, out_dtype=float64"),
    ("*fp16", tl.float32, "*fp32", "fp16 x fp16, out_dtype=float32  (control, must be OK)"),
    ("*fp16", tl.float16, "*fp16", "fp16 x fp16, out_dtype=float16  (control, must be OK)"),
    ("*i8", tl.int32, "*i32", "int8 x int8,  out_dtype=int32   (control, must be OK)"),
]

for arch in (80, 89, 90):
    for ptr_ty, out_dtype, c_ty, label in CASES:
        print("sm_%-4d %-52s %s" % (arch, label, build(arch, ptr_ty, out_dtype, c_ty)))
