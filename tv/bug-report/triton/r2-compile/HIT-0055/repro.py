#!/usr/bin/env python
"""Reproduce HIT-0055 — invalid-ir-frontend.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    invalid-ir-frontend  (MLIR verifier)
    verify:'tt.splat' op requires the same element type for all operands and results
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
SPEC = json.loads(r"""{"id": "ind:triton_mm_4942c5ab", "corpus": "inductor", "fn": "triton_mm_4942c5ab", "meta": {"from": "ttgir-broad", "grid": [4, 1, 1]}, "args": [{"name": "arg_A", "kind": "ptr", "ty": "*i8", "value": null}, {"name": "arg_B", "kind": "ptr", "ty": "*i8", "value": null}, {"name": "out_ptr0", "kind": "ptr", "ty": "*i32", "value": null}]}""")
ARCH = 80
SHAPE = json.loads(r"""{"ptr_off": {"arg_A": 0, "arg_B": 0, "out_ptr0": 0}, "ptr_ty": {"arg_A": "*fp8e5", "arg_B": "*fp8e5", "out_ptr0": "*i32"}, "scalars": {}, "consts": {}, "dtype_swapped": true}""")
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
