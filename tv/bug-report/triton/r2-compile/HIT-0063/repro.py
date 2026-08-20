#!/usr/bin/env python
"""Reproduce HIT-0063 — crash-assert.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-assert  (assertion)
    assert@MMAHelpers.h:171:block == N
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
SPEC = json.loads(r"""{"id": "ind:triton_mm_cb5ec818", "corpus": "inductor", "fn": "triton_mm_cb5ec818", "meta": {"from": "ttgir-broad", "grid": [2, 1, 1]}, "args": [{"name": "arg_A", "kind": "ptr", "ty": "*fp8e4nv", "value": null}, {"name": "arg_B", "kind": "ptr", "ty": "*fp8e4nv", "value": null}, {"name": "in_ptr2", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "in_ptr3", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "out_ptr0", "kind": "ptr", "ty": "*bf16", "value": null}]}""")
ARCH = 90
SHAPE = json.loads(r"""{"ptr_off": {"arg_A": 0, "arg_B": 0, "in_ptr2": 0, "in_ptr3": 0, "out_ptr0": 0}, "ptr_ty": {"arg_A": "*fp8e4nv", "arg_B": "*fp8e4b15", "in_ptr2": "*fp16", "in_ptr3": "*i64", "out_ptr0": "*bf16"}, "scalars": {}, "consts": {}, "dtype_swapped": true}""")
CONFIG = json.loads(r"""{"num_warps": 4, "num_stages": 3, "num_ctas": 2, "maxnreg": null, "default_dot_input_precision": "tf32", "launch_cooperative_grid": false, "launch_pdl": false, "sanitize_overflow": true, "enable_fp_fusion": false}""")

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
