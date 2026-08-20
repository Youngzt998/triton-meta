#!/usr/bin/env python
"""Reproduce HIT-0066 — crash-python.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-python  (RuntimeError with no user-facing message)
    RuntimeError@compiler.py:371:PassManager::run failed
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
SPEC = json.loads(r"""{"id": "bch:reflection_pad2d_kernel_00c19208", "corpus": "bench", "fn": "reflection_pad2d_kernel_00c19208", "meta": {"from": "ttir-broad", "grid": [3, 5, 1]}, "args": [{"name": "in_ptr", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "out_ptr", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "B", "kind": "int", "ty": "i32", "value": 3}, {"name": "H_in", "kind": "int", "ty": "i32", "value": 33}, {"name": "W_in", "kind": "int", "ty": "i32", "value": 33}, {"name": "pad_left", "kind": "int", "ty": "i32", "value": 1}, {"name": "pad_top", "kind": "int", "ty": "i32", "value": 1}, {"name": "H_out", "kind": "int", "ty": "i32", "value": 35}, {"name": "W_out", "kind": "int", "ty": "i32", "value": 35}, {"name": "BLOCK_HW", "kind": "constexpr", "ty": "constexpr", "value": 256}]}""")
ARCH = 80
SHAPE = json.loads(r"""{"ptr_off": {"in_ptr": 0, "out_ptr": 0}, "ptr_ty": {"in_ptr": "*fp32", "out_ptr": "*fp32"}, "scalars": {"B": 127, "H_in": 7, "W_in": 1, "pad_left": 1, "pad_top": 1024, "H_out": 35, "W_out": 1}, "consts": {"BLOCK_HW": 1}, "dtype_swapped": false}""")
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
