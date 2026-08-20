#!/usr/bin/env python
"""Reproduce HIT-0045 — crash-assert.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-assert  (assertion)
    assert@StorageUniquerSupport.h:180:succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))
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
SPEC = json.loads(r"""{"id": "bch:_cp_gather_indexer_quant_cache_kernel_d5cecde2", "corpus": "bench", "fn": "_cp_gather_indexer_quant_cache_kernel_d5cecde2", "meta": {"from": "ttir-broad", "grid": [22, 1, 1]}, "args": [{"name": "kv_cache_ptr", "kind": "ptr", "ty": "*u8", "value": null}, {"name": "kv_cache_scale_ptr", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "k_fp8_ptr", "kind": "ptr", "ty": "*u8", "value": null}, {"name": "k_scale_ptr", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "block_table_ptr", "kind": "ptr", "ty": "*i32", "value": null}, {"name": "cu_seqlen_ptr", "kind": "ptr", "ty": "*i32", "value": null}, {"name": "block_size", "kind": "int", "ty": "i32", "value": 16}, {"name": "block_table_stride", "kind": "int", "ty": "i32", "value": 2}, {"name": "kv_cache_stride", "kind": "int", "ty": "i32", "value": 2112}, {"name": "kv_cache_scale_stride", "kind": "int", "ty": "i32", "value": 528}, {"name": "k_fp8_stride", "kind": "int", "ty": "i32", "value": 128}, {"name": "num_quant_blocks", "kind": "int", "ty": "i32", "value": 1}, {"name": "batch_size", "kind": "constexpr", "ty": "constexpr", "value": 1}, {"name": "HEAD_DIM", "kind": "constexpr", "ty": "constexpr", "value": 128}, {"name": "QUANT_BLOCK_SIZE", "kind": "constexpr", "ty": "constexpr", "value": 128}, {"name": "TOKEN_BLOCK", "kind": "constexpr", "ty": "constexpr", "value": 1}, {"name": "BATCH_SCAN_SIZE", "kind": "constexpr", "ty": "constexpr", "value": 1}, {"name": "SEARCH_STEPS", "kind": "constexpr", "ty": "constexpr", "value": 1}]}""")
ARCH = 90
SHAPE = json.loads(r"""{"ptr_off": {"kv_cache_ptr": 0, "kv_cache_scale_ptr": 0, "k_fp8_ptr": 0, "k_scale_ptr": 0, "block_table_ptr": 0, "cu_seqlen_ptr": 0}, "ptr_ty": {"kv_cache_ptr": "*u8", "kv_cache_scale_ptr": "*fp32", "k_fp8_ptr": "*u8", "k_scale_ptr": "*fp32", "block_table_ptr": "*i32", "cu_seqlen_ptr": "*i32"}, "scalars": {"block_size": 2, "block_table_stride": 2, "kv_cache_stride": 2112, "kv_cache_scale_stride": 262145, "k_fp8_stride": 128, "num_quant_blocks": 17}, "consts": {"batch_size": 32, "HEAD_DIM": 64, "QUANT_BLOCK_SIZE": 64, "TOKEN_BLOCK": 1, "BATCH_SCAN_SIZE": 4, "SEARCH_STEPS": 1}, "dtype_swapped": false}""")
CONFIG = json.loads(r"""{"num_warps": 4, "num_stages": 3, "num_ctas": 8, "maxnreg": null, "default_dot_input_precision": "tf32", "launch_cooperative_grid": false, "launch_pdl": false, "sanitize_overflow": true, "enable_fp_fusion": false}""")

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
