#!/usr/bin/env python
"""Reproduce HIT-0048 — invalid-ir-frontend.

    source /home/youngzt/fuzz/r2-compile/env.sh
    python repro.py

Expected: the compile below fails with
    invalid-ir-frontend  (MLIR verifier)
    verify:'tt.int_to_ptr' op operand #N must be N-bit signless integer or tensor of N-bit signless integer values, but got 'iN'
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
SPEC = json.loads(r"""{"id": "bch:grouped_matmul_kernel_87cbfee0", "corpus": "bench", "fn": "grouped_matmul_kernel_87cbfee0", "meta": {"from": "ttgir-broad", "grid": [96, 1, 1]}, "args": [{"name": "group_a_ptrs", "kind": "ptr", "ty": "*i64", "value": null}, {"name": "group_b_ptrs", "kind": "ptr", "ty": "*i64", "value": null}, {"name": "group_c_ptrs", "kind": "ptr", "ty": "*i64", "value": null}, {"name": "group_gemm_sizes", "kind": "ptr", "ty": "*i32", "value": null}, {"name": "g_lds", "kind": "ptr", "ty": "*i32", "value": null}, {"name": "group_size", "kind": "int", "ty": "i32", "value": 4}, {"name": "DTYPE", "kind": "constexpr", "ty": "constexpr", "value": {"__tl__": "bf16"}}, {"name": "NUM_SMS", "kind": "constexpr", "ty": "constexpr", "value": 132}, {"name": "BLOCK_SIZE_M", "kind": "constexpr", "ty": "constexpr", "value": 128}, {"name": "BLOCK_SIZE_N", "kind": "constexpr", "ty": "constexpr", "value": 128}, {"name": "BLOCK_SIZE_K", "kind": "constexpr", "ty": "constexpr", "value": 128}]}""")
ARCH = 80
SHAPE = json.loads(r"""{"ptr_off": {"group_a_ptrs": 0, "group_b_ptrs": 0, "group_c_ptrs": 0, "group_gemm_sizes": 0, "g_lds": 0}, "ptr_ty": {"group_a_ptrs": "*u8", "group_b_ptrs": "*i64", "group_c_ptrs": "*i64", "group_gemm_sizes": "*i32", "g_lds": "*i32"}, "scalars": {"group_size": 65536}, "consts": {"DTYPE": {"__tl__": "bf16"}, "NUM_SMS": 32, "BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 1, "BLOCK_SIZE_K": 64}, "dtype_swapped": true}""")
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
