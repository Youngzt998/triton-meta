#!/usr/bin/env python
"""Reproduce HIT-0004 — invalid-ir.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    invalid-ir  (MLIR verifier)
    verify:'llvm.fcmp' op operand #N must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got 'iN'
"""
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # this line needs no GPU
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
os.environ.setdefault("MLIR_DISABLE_MULTITHREADING", "1")
sys.path.insert(0, "/home/youngzt/fuzz/r2-compile")

from fz.corpus import Spec, resolve_fn
from fz.axes import build_source_inputs
from fz.compile1 import compile_stages

# embedded as JSON text, not as python literals: JSON writes null/true/false,
# which are not python names
SPEC = json.loads(r"""{"id": "bch:argmin_kernel_f8fcbf2b", "corpus": "bench", "fn": "argmin_kernel_f8fcbf2b", "meta": {"from": "ttir-broad", "grid": [1, 2, 1]}, "args": [{"name": "inp", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "out_index", "kind": "ptr", "ty": "*i64", "value": null}, {"name": "M", "kind": "int", "ty": "i32", "value": 1}, {"name": "N", "kind": "int", "ty": "i32", "value": 1}, {"name": "K", "kind": "int", "ty": "i32", "value": 2}, {"name": "BLOCK_M", "kind": "constexpr", "ty": "constexpr", "value": 4}, {"name": "BLOCK_N", "kind": "constexpr", "ty": "constexpr", "value": 1}]}""")
ARCH = 89
SHAPE = json.loads(r"""{"ptr_off": {"inp": 0, "out_index": 0}, "ptr_ty": {"inp": "*fp8e4nv", "out_index": "*i1"}, "scalars": {"M": 1, "N": 11, "K": 1}, "consts": {"BLOCK_M": 2, "BLOCK_N": 1}, "dtype_swapped": true}""")
CONFIG = json.loads(r"""{"num_warps": 4, "num_stages": 3, "num_ctas": 1, "maxnreg": null, "default_dot_input_precision": "tf32", "launch_cooperative_grid": false, "launch_pdl": false, "sanitize_overflow": true, "enable_fp_fusion": false}""")

spec = Spec.from_json(SPEC)
fn = resolve_fn(spec)
sig, cx, attrs = build_source_inputs(spec, SHAPE)
print("kernel   :", spec.fn)
print("arch     : sm_%s" % ARCH)
print("signature:", sig)
print("constexpr:", cx)
print("attrs    :", attrs)
print("config   :", CONFIG)
compile_stages(fn, sig, cx, attrs, ARCH, CONFIG)
print("NO FAILURE — did not reproduce")
