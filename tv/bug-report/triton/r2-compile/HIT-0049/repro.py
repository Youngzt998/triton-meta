#!/usr/bin/env python
"""Reproduce HIT-0049 — crash-assert.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-assert  (LLVM ERROR)
    llvm-error:Layout conflict for block arg
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
SPEC = json.loads(r"""{"id": "bch:bmm_kernel_a62513ac", "corpus": "bench", "fn": "bmm_kernel_a62513ac", "meta": {"from": "ttir-broad", "grid": [1, 1, 4]}, "args": [{"name": "A", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "B", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "O", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "M", "kind": "int", "ty": "i32", "value": 1}, {"name": "N", "kind": "int", "ty": "i32", "value": 1}, {"name": "K", "kind": "int", "ty": "i32", "value": 32}, {"name": "stride_ab", "kind": "int", "ty": "i32", "value": 32}, {"name": "stride_am", "kind": "int", "ty": "i32", "value": 32}, {"name": "stride_ak", "kind": "int", "ty": "i32", "value": 1}, {"name": "stride_bb", "kind": "int", "ty": "i32", "value": 32}, {"name": "stride_bk", "kind": "int", "ty": "i32", "value": 1}, {"name": "stride_bn", "kind": "int", "ty": "i32", "value": 1}, {"name": "stride_ob", "kind": "int", "ty": "i32", "value": 1}, {"name": "stride_om", "kind": "int", "ty": "i32", "value": 1}, {"name": "stride_on", "kind": "int", "ty": "i32", "value": 1}, {"name": "TILE_M", "kind": "constexpr", "ty": "constexpr", "value": 32}, {"name": "TILE_N", "kind": "constexpr", "ty": "constexpr", "value": 32}, {"name": "TILE_K", "kind": "constexpr", "ty": "constexpr", "value": 32}, {"name": "GROUP_M", "kind": "constexpr", "ty": "constexpr", "value": 1}, {"name": "DIVISIBLE_M", "kind": "constexpr", "ty": "constexpr", "value": 0}, {"name": "DIVISIBLE_N", "kind": "constexpr", "ty": "constexpr", "value": 0}, {"name": "DIVISIBLE_K", "kind": "constexpr", "ty": "constexpr", "value": 1}, {"name": "IS_FP64", "kind": "constexpr", "ty": "constexpr", "value": 0}]}""")
ARCH = 90
SHAPE = json.loads(r"""{"ptr_off": {"A": 0, "B": 0, "O": 0}, "ptr_ty": {"A": "*fp16", "B": "*fp16", "O": "*fp16"}, "scalars": {"M": 1, "N": 1, "K": 127, "stride_ab": 32, "stride_am": 1048576, "stride_ak": 1, "stride_bb": 32, "stride_bk": 1, "stride_bn": 4095, "stride_ob": 17, "stride_om": 17, "stride_on": 1}, "consts": {"TILE_M": 32, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 1, "DIVISIBLE_M": 1, "DIVISIBLE_N": 1, "DIVISIBLE_K": 32, "IS_FP64": 1}, "dtype_swapped": false}""")
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
