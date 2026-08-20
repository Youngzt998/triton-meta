#!/usr/bin/env python
"""Reproduce HIT-0007 — invalid-ir-frontend.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    invalid-ir-frontend  (MLIR verifier)
    verify:'tt.store' op failed to verify that value type matches ptr type
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
SPEC = json.loads(r"""{"id": "bch:reshape_and_cache_flash_kernel_476250f6", "corpus": "bench", "fn": "reshape_and_cache_flash_kernel_476250f6", "meta": {"from": "ttir-broad", "grid": [42, 1, 1]}, "args": [{"name": "key", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "value", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "key_cache", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "value_cache", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "slot_mapping", "kind": "ptr", "ty": "*i64", "value": null}, {"name": "block_stride", "kind": "int", "ty": "i32", "value": 4096}, {"name": "key_stride", "kind": "int", "ty": "i32", "value": 1536}, {"name": "value_stride", "kind": "int", "ty": "i32", "value": 1536}, {"name": "num_heads", "kind": "int", "ty": "i32", "value": 8}, {"name": "head_size", "kind": "int", "ty": "i32", "value": 64}, {"name": "block_size", "kind": "int", "ty": "i32", "value": 8}, {"name": "k_scale", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "v_scale", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "n", "kind": "constexpr", "ty": "constexpr", "value": 512}]}""")
ARCH = 80
SHAPE = json.loads(r"""{"ptr_off": {"key": 0, "value": 0, "key_cache": 0, "value_cache": 0, "slot_mapping": 0, "k_scale": 0, "v_scale": 0}, "ptr_ty": {"key": "*fp8e4b15", "value": "*fp32", "key_cache": "*i32", "value_cache": "*fp16", "slot_mapping": "*i64", "k_scale": "*fp32", "v_scale": "*fp8e5"}, "scalars": {"block_stride": 221, "key_stride": 2, "value_stride": 8192, "num_heads": 1, "head_size": 64, "block_size": 8}, "consts": {"n": 256}, "dtype_swapped": true}""")
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
