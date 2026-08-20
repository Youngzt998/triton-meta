#!/usr/bin/env python
"""Reproduce HIT-0009 — crash-assert.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-assert  (assertion)
    assert@ElementwiseOpToLLVM.cpp:482:roundingMode.has_value() && "Rounding mode must be specified for convertsions to fpN"
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
SPEC = json.loads(r"""{"id": "bch:special_i1_kernel_7e502f07", "corpus": "bench", "fn": "special_i1_kernel_7e502f07", "meta": {"from": "ttir-broad", "grid": [1, 1, 1]}, "args": [{"name": "x_ptr", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "out_ptr", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "n_elements", "kind": "int", "ty": "i32", "value": 1}, {"name": "BLOCK_SIZE", "kind": "constexpr", "ty": "constexpr", "value": 1024}]}""")
ARCH = 89
SHAPE = json.loads(r"""{"ptr_off": {"x_ptr": 0, "out_ptr": 0}, "ptr_ty": {"x_ptr": "*fp8e5", "out_ptr": "*fp8e4nv"}, "scalars": {"n_elements": 1}, "consts": {"BLOCK_SIZE": 512}, "dtype_swapped": true}""")
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
