#!/usr/bin/env python
"""Reproduce HIT-0005 — crash-assert.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-assert  (assertion)
    assert@PlanCTA.cpp:359:stores.size() > N && "Cannot find store-like ops"
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
SPEC = json.loads(r"""{"id": "bch:_assert_async_kernel_ffc659d7", "corpus": "bench", "fn": "_assert_async_kernel_ffc659d7", "meta": {"from": "ttir-broad", "grid": [1, 1, 1]}, "args": [{"name": "x_ptr", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "MSG", "kind": "constexpr", "ty": "constexpr", "value": "Assertion failed!"}]}""")
ARCH = 90
SHAPE = json.loads(r"""{"ptr_off": {"x_ptr": 0}, "ptr_ty": {"x_ptr": "*fp32"}, "scalars": {}, "consts": {"MSG": "Assertion failed!"}, "dtype_swapped": false}""")
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
