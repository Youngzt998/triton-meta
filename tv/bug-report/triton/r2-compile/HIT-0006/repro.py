#!/usr/bin/env python
"""Reproduce HIT-0006 — crash-assert.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-assert  (assertion)
    assert@PlanCTA.cpp:211:!tiled && "CTA tiling is already determined"
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
SPEC = json.loads(r"""{"id": "ind:triton_flex_attention_backward_2c768ac2", "corpus": "inductor", "fn": "triton_flex_attention_backward_2c768ac2", "meta": {"from": "ttgir-broad", "grid": [4, 1, 1]}, "args": [{"name": "arg_Q", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "arg_K", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "arg_V", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "arg_LSE", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "arg_DELTA", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "arg_DO", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "arg_DQ", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "arg_DV", "kind": "ptr", "ty": "*fp16", "value": null}, {"name": "arg_KV_NUM_BLKS", "kind": "ptr", "ty": "*i32", "value": null}, {"name": "arg_KV_IDX", "kind": "ptr", "ty": "*i32", "value": null}, {"name": "arg_Q_NUM_BLKS", "kind": "ptr", "ty": "*i32", "value": null}, {"name": "arg_Q_IDX", "kind": "ptr", "ty": "*i32", "value": null}, {"name": "arg_FULL_KV_NUM_BLKS", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "arg_FULL_KV_IDX", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "arg_FULL_Q_NUM_BLKS", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "arg_FULL_Q_IDX", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "out_ptr0", "kind": "ptr", "ty": "*fp16", "value": null}]}""")
ARCH = 90
SHAPE = json.loads(r"""{"ptr_off": {"arg_Q": 0, "arg_K": 0, "arg_V": 0, "arg_LSE": 0, "arg_DELTA": 0, "arg_DO": 0, "arg_DQ": 0, "arg_DV": 0, "arg_KV_NUM_BLKS": 0, "arg_KV_IDX": 0, "arg_Q_NUM_BLKS": 0, "arg_Q_IDX": 0, "arg_FULL_KV_NUM_BLKS": 0, "arg_FULL_KV_IDX": 0, "arg_FULL_Q_NUM_BLKS": 0, "arg_FULL_Q_IDX": 0, "out_ptr0": 0}, "ptr_ty": {"arg_Q": "*fp16", "arg_K": "*fp16", "arg_V": "*fp16", "arg_LSE": "*fp32", "arg_DELTA": "*fp32", "arg_DO": "*fp16", "arg_DQ": "*fp16", "arg_DV": "*fp16", "arg_KV_NUM_BLKS": "*i32", "arg_KV_IDX": "*i32", "arg_Q_NUM_BLKS": "*i32", "arg_Q_IDX": "*i32", "arg_FULL_KV_NUM_BLKS": "*fp32", "arg_FULL_KV_IDX": "*fp32", "arg_FULL_Q_NUM_BLKS": "*fp32", "arg_FULL_Q_IDX": "*fp32", "out_ptr0": "*fp16"}, "scalars": {}, "consts": {}, "dtype_swapped": false}""")
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
