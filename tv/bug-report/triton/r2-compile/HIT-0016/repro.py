#!/usr/bin/env python
"""Reproduce HIT-0016 — crash-assert.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    crash-assert  (assertion)
    assert@ScanOpToLLVM.cpp:265:numScanBlocks * numParallelBlocks * parallelElementsPerThread * scanElementsPerThreads == srcValues.size()
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
SPEC = json.loads(r"""{"id": "ind:triton_per_fused_logcumsumexp_0_1f363aee", "corpus": "inductor", "fn": "triton_per_fused_logcumsumexp_0_1f363aee", "meta": {"from": "ttgir-broad", "grid": [1, 1, 1]}, "args": [{"name": "in_ptr0", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "out_ptr0", "kind": "ptr", "ty": "*fp32", "value": null}, {"name": "xnumel", "kind": "int", "ty": "i32", "value": 32}, {"name": "r0_numel", "kind": "int", "ty": "i32", "value": 128}, {"name": "XBLOCK", "kind": "constexpr", "ty": "constexpr", "value": 32}]}""")
ARCH = 90
SHAPE = json.loads(r"""{"ptr_off": {"in_ptr0": 0, "out_ptr0": 0}, "ptr_ty": {"in_ptr0": "*fp32", "out_ptr0": "*fp32"}, "scalars": {"xnumel": 11, "r0_numel": 16}, "consts": {"XBLOCK": 128}, "dtype_swapped": false}""")
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
