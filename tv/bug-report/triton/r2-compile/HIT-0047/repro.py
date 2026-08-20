#!/usr/bin/env python
"""Reproduce HIT-0047 — crash-python.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-python  (unclassified PTXASError)
    PTXASError@compiler.py:630:Repro command: ptxas -lineinfo -suppress-debug-info --fmad=false -v --regAllocOptLevel=N --gpu-name=sm_N tmpfrNmcNwo.ptx -o tmpfrNmcN
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
SPEC = json.loads(r"""{"id": "ind:triton_per_fused_native_group_norm_0_9294c0a8", "corpus": "inductor", "fn": "triton_per_fused_native_group_norm_0_9294c0a8", "meta": {"from": "ttir-broad", "grid": [4, 1, 1]}, "args": [{"name": "in_ptr0", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "out_ptr0", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "out_ptr1", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "out_ptr2", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "xnumel", "kind": "int", "ty": "i32", "value": 32}, {"name": "r0_numel", "kind": "int", "ty": "i32", "value": 800}, {"name": "XBLOCK", "kind": "constexpr", "ty": "constexpr", "value": 8}]}""")
ARCH = 89
SHAPE = json.loads(r"""{"ptr_off": {"in_ptr0": 0, "out_ptr0": 0, "out_ptr1": 2, "out_ptr2": 4}, "ptr_ty": {"in_ptr0": "*bf16", "out_ptr0": "*i16", "out_ptr1": "*fp8e4nv", "out_ptr2": "*i16"}, "scalars": {"xnumel": 1, "r0_numel": 8192}, "consts": {"XBLOCK": 8}, "dtype_swapped": true}""")
CONFIG = json.loads(r"""{"num_warps": 1, "num_stages": 7, "num_ctas": 1, "maxnreg": null, "default_dot_input_precision": "tf32", "launch_cooperative_grid": false, "launch_pdl": true, "sanitize_overflow": false, "enable_fp_fusion": false}""")

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
