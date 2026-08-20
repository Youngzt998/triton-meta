#!/usr/bin/env python
"""Reproduce HIT-0019 — crash-assert.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-assert  (assertion)
    assert@PlanCTA.cpp:164:step < maxSteps && "Maximum number of steps exceeded"
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
SPEC = json.loads(r"""{"id": "bch:_small_jacobi_svd_kernel_791182a2", "corpus": "bench", "fn": "_small_jacobi_svd_kernel_791182a2", "meta": {"from": "ttir-broad", "grid": [1, 1, 1]}, "args": [{"name": "A", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "A_WORK", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "V_WORK", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "U", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "S", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "V", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "M", "kind": "constexpr", "ty": "constexpr", "value": 16}, {"name": "N", "kind": "constexpr", "ty": "constexpr", "value": 8}, {"name": "K", "kind": "constexpr", "ty": "constexpr", "value": 8}, {"name": "ROWS", "kind": "constexpr", "ty": "constexpr", "value": 16}, {"name": "TALL", "kind": "constexpr", "ty": "constexpr", "value": 1}, {"name": "BLOCK_R", "kind": "constexpr", "ty": "constexpr", "value": 16}, {"name": "BLOCK_K", "kind": "constexpr", "ty": "constexpr", "value": 8}, {"name": "SWEEPS", "kind": "constexpr", "ty": "constexpr", "value": 5}]}""")
ARCH = 90
SHAPE = json.loads(r"""{"ptr_off": {"A": 0, "A_WORK": 0, "V_WORK": 0, "U": 0, "S": 0, "V": 0}, "ptr_ty": {"A": "*fp32", "A_WORK": "*fp32", "V_WORK": "*fp32", "U": "*fp32", "S": "*fp32", "V": "*fp32"}, "scalars": {}, "consts": {"M": 4, "N": 4, "K": 4, "ROWS": 8, "TALL": 1, "BLOCK_R": 8, "BLOCK_K": 1, "SWEEPS": 2}, "dtype_swapped": false}""")
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
