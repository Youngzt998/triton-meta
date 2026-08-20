#!/usr/bin/env python
"""Reproduce HIT-0001 — invalid-ir.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    invalid-ir  (raw MLIR error)
    mlirerr:Result has an invalid layout: #ttg.slice<{dim = N, parent = #ttg.blocked<>}>.
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
SPEC = json.loads(r"""{"id": "bch:logsumexp_kernel_non_inner_baaf36b4", "corpus": "bench", "fn": "logsumexp_kernel_non_inner_baaf36b4", "meta": {"from": "ttir-broad", "grid": [1, 2, 1]}, "args": [{"name": "output_ptr", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "input_ptr", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "M", "kind": "int", "ty": "i32", "value": 1}, {"name": "N", "kind": "int", "ty": "i32", "value": 1}, {"name": "K", "kind": "int", "ty": "i32", "value": 2}, {"name": "TILE_N", "kind": "constexpr", "ty": "constexpr", "value": 8192}, {"name": "TILE_K", "kind": "constexpr", "ty": "constexpr", "value": 1}, {"name": "ONE_TILE_PER_CTA", "kind": "constexpr", "ty": "constexpr", "value": 1}]}""")
ARCH = 90
SHAPE = json.loads(r"""{"ptr_off": {"output_ptr": 0, "input_ptr": 0}, "ptr_ty": {"output_ptr": "*fp16", "input_ptr": "*fp16"}, "scalars": {"M": 11, "N": 17, "K": 2319}, "consts": {"TILE_N": 32, "TILE_K": 1, "ONE_TILE_PER_CTA": 32}, "dtype_swapped": false}""")
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
