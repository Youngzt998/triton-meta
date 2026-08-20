#!/usr/bin/env python
"""Reproduce HIT-0038 — crash-assert.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-assert  (assertion)
    assert@ViewOpToLLVM.cpp:246:!isExpensiveView(op.getSrc().getType(), op.getType())
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
SPEC = json.loads(r"""{"id": "bch:mode_kernel_b978d037", "corpus": "bench", "fn": "mode_kernel_b978d037", "meta": {"from": "ttir-broad", "grid": [1, 1, 1]}, "args": [{"name": "sorted_inp", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "sorted_indices", "kind": "ptr", "ty": "*i64", "value": null}, {"name": "out_value", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "out_index", "kind": "ptr", "ty": "*i64", "value": null}, {"name": "M", "kind": "int", "ty": "i32", "value": 2}, {"name": "N", "kind": "int", "ty": "i32", "value": 1}, {"name": "BLOCK_M", "kind": "constexpr", "ty": "constexpr", "value": 8}, {"name": "BLOCK_N", "kind": "constexpr", "ty": "constexpr", "value": 1024}]}""")
ARCH = 90
SHAPE = json.loads(r"""{"ptr_off": {"sorted_inp": 0, "sorted_indices": 0, "out_value": 0, "out_index": 0}, "ptr_ty": {"sorted_inp": "*fp16", "sorted_indices": "*i64", "out_value": "*fp16", "out_index": "*i64"}, "scalars": {"M": 2, "N": 3051}, "consts": {"BLOCK_M": 2, "BLOCK_N": 256}, "dtype_swapped": false}""")
CONFIG = json.loads(r"""{"num_warps": 4, "num_stages": 3, "num_ctas": 8, "maxnreg": null, "default_dot_input_precision": "tf32", "launch_cooperative_grid": false, "launch_pdl": false, "sanitize_overflow": true, "enable_fp_fusion": false}""")

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
