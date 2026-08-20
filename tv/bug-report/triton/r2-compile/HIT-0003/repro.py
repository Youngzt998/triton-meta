#!/usr/bin/env python
"""Reproduce HIT-0003 — crash-assert.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-assert  (LLVM ERROR)
    llvm-error:Unexpected parent op of block argument
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
SPEC = json.loads(r"""{"id": "bch:upsample_trilinear3d_kernel_6c02f6c8", "corpus": "bench", "fn": "upsample_trilinear3d_kernel_6c02f6c8", "meta": {"from": "ttgir-broad", "grid": [48, 2, 1]}, "args": [{"name": "ptr_o", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "ptr_i", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "N", "kind": "int", "ty": "i32", "value": 4}, {"name": "C", "kind": "int", "ty": "i32", "value": 8}, {"name": "OD", "kind": "int", "ty": "i32", "value": 64}, {"name": "OH", "kind": "int", "ty": "i32", "value": 64}, {"name": "OW", "kind": "int", "ty": "i32", "value": 64}, {"name": "ID", "kind": "int", "ty": "i32", "value": 32}, {"name": "IH", "kind": "int", "ty": "i32", "value": 32}, {"name": "IW", "kind": "int", "ty": "i32", "value": 32}, {"name": "scale_d", "kind": "float", "ty": "fp32", "value": 0.5}, {"name": "scale_h", "kind": "float", "ty": "fp32", "value": 0.5}, {"name": "scale_w", "kind": "float", "ty": "fp32", "value": 0.5}, {"name": "bias_d", "kind": "float", "ty": "fp32", "value": -0.25}, {"name": "bias_h", "kind": "float", "ty": "fp32", "value": -0.25}, {"name": "bias_w", "kind": "float", "ty": "fp32", "value": -0.25}, {"name": "BLOCK_SIZE", "kind": "constexpr", "ty": "constexpr", "value": 512}, {"name": "SAME_D", "kind": "constexpr", "ty": "constexpr", "value": false}, {"name": "SAME_H", "kind": "constexpr", "ty": "constexpr", "value": false}, {"name": "SAME_W", "kind": "constexpr", "ty": "constexpr", "value": false}, {"name": "USE_INT32_IDX", "kind": "constexpr", "ty": "constexpr", "value": true}]}""")
ARCH = 90
SHAPE = json.loads(r"""{"ptr_off": {"ptr_o": 0, "ptr_i": 0}, "ptr_ty": {"ptr_o": "*fp16", "ptr_i": "*fp16"}, "scalars": {"N": 32, "C": 8, "OD": 31, "OH": 64, "OW": 16385, "ID": 2048, "IH": 32, "IW": 16}, "consts": {"BLOCK_SIZE": 256, "SAME_D": false, "SAME_H": false, "SAME_W": false, "USE_INT32_IDX": true}, "dtype_swapped": false}""")
CONFIG = json.loads(r"""{"num_warps": 4, "num_stages": 3, "num_ctas": 4, "maxnreg": null, "default_dot_input_precision": "tf32", "launch_cooperative_grid": false, "launch_pdl": false, "sanitize_overflow": true, "enable_fp_fusion": false}""")

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
