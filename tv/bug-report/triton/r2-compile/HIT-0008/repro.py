#!/usr/bin/env python
"""Reproduce HIT-0008 — invalid-ir-frontend.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    invalid-ir-frontend  (raw MLIR error)
    mlirerr:number of input elements N must be a multiple of the op's packed_element attribute, N
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
SPEC = json.loads(r"""{"id": "bch:_rank2_svd_kernel_813eb03e", "corpus": "bench", "fn": "_rank2_svd_kernel_813eb03e", "meta": {"from": "ttir-broad", "grid": [1, 1, 1]}, "args": [{"name": "A", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "U", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "S", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "V", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "M", "kind": "constexpr", "ty": "constexpr", "value": 2}, {"name": "N", "kind": "constexpr", "ty": "constexpr", "value": 2}, {"name": "TALL", "kind": "constexpr", "ty": "constexpr", "value": 1}, {"name": "BLOCK_R", "kind": "constexpr", "ty": "constexpr", "value": 2}]}""")
ARCH = 80
SHAPE = json.loads(r"""{"ptr_off": {"A": 0, "U": 0, "S": 0, "V": 0}, "ptr_ty": {"A": "*fp32", "U": "*fp8e4b15", "S": "*fp32", "V": "*fp64"}, "scalars": {}, "consts": {"M": 1, "N": 32, "TALL": 1, "BLOCK_R": 1}, "dtype_swapped": true}""")
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
