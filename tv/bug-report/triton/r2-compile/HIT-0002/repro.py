#!/usr/bin/env python
"""Reproduce HIT-0002 — invalid-ir-frontend.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    invalid-ir-frontend  (MLIR verifier)
    verify:'arith.cmpf' op operand #N must be floating-point-like, but got 'iN'
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
SPEC = json.loads(r"""{"id": "bch:argmin_kernel_1_60cfa9bd", "corpus": "bench", "fn": "argmin_kernel_1_60cfa9bd", "meta": {"from": "ttir-broad", "grid": [1, 1, 1]}, "args": [{"name": "inp", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "mid_value", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "mid_index", "kind": "ptr", "ty": "*i64", "value": null}, {"name": "M", "kind": "int", "ty": "i32", "value": 2}, {"name": "BLOCK_SIZE", "kind": "constexpr", "ty": "constexpr", "value": 2}]}""")
ARCH = 80
SHAPE = json.loads(r"""{"ptr_off": {"inp": 0, "mid_value": 0, "mid_index": 0}, "ptr_ty": {"inp": "*fp8e4b15", "mid_value": "*fp16", "mid_index": "*i64"}, "scalars": {"M": 2}, "consts": {"BLOCK_SIZE": 32}, "dtype_swapped": true}""")
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
