# r2-compile (R4) — crash / invalid-IR classes

One row per distinct root cause. `count` is how many (kernel x shape x config x arch) points showed it; only the first point of each class gets a full report, and every occurrence is counted here. Same pattern as round 1's `ttir-broad/HITS/CLASSES.md`, where 730 kernels collapsed into one row.

classes: 75   total occurrences: 14059

| report | kind | count | archs | culprit | signature | examples |
|---|---|---|---|---|---|---|
| HIT-0001 | invalid-ir | 1863 | 90 | `ttgir` | `mlirerr:Result has an invalid layout: #ttg.slice<{dim = N, parent = #ttg.blocked<>}>.` | bch:logsumexp_kernel_non_inner_baaf36b4, bch:_complete_svd_factor_kernel_e266834b, ind:triton_unk_fused_amax_sum_0_8525b515 (+1860) |
| - | timeout | 1538 | 80,89,90 | `cubin` | `timeout@cubin` | bch:max_pool3d_forward_kernel_1e97696b, bch:grid_sample_3d_trilinear_zeros_tiled_kernel_717eb6ff, bch:_fused_backward_kernel_dbedd307 (+1535) |
| - | timeout | 1436 | 80,89,90 | `ttgir` | `timeout@ttgir` | bch:max_pool3d_forward_kernel_1e97696b, bch:_small_jacobi_svals_kernel_f9679a95, bch:_small_jacobi_svd_kernel_791182a2 (+1433) |
| HIT-0008 | invalid-ir-frontend | 1025 | 80,89,90 | `frontend` | `mlirerr:number of input elements N must be a multiple of the op's packed_element attribute, N` | bch:_rank2_svd_kernel_813eb03e, bch:_as_strided_copy_1d_kernel_d11a073d, ind:triton_poi_fused_add_mul_relu_sigmoid_sub_tanh_0_e1f7d5c0 (+1022) |
| HIT-0006 | crash-assert | 827 | 90 | `ttgir` | `assert@PlanCTA.cpp:211:!tiled && "CTA tiling is already determined"` | ind:triton_flex_attention_backward_2c768ac2, bch:_conv_transpose2d_residue_kernel_bc97b619, ind:triton_flex_attention_65dbdfce (+824) |
| - | timeout | 772 | 80,89,90 | `ttir` | `timeout@ttir` | bch:_small_jacobi_svd_kernel_791182a2, bch:_thin_reorthogonalize_kernel_4046334c, bch:mhc_pre_generic_kernel_095393f1 (+769) |
| HIT-0011 | crash-assert | 728 | 80,89,90 | `llir` | `llvm-error:Unsupported rounding mode for conversion.` | ind:triton_poi_fused__softmax_4_04f5cb59, bch:mhc_split_sinkhorn_kernel_hcmult_4_5187f1d9, ind:triton_per_fused_mul_sum_0_f904167e (+725) |
| HIT-0033 | invalid-ir | 653 | 80,89,90 | `llir` | `verify:'llvm.<float-op>' op operand #N must be floating point LLVM type or LLVM dialect-compatible vector of floating po` | ind:triton_per_fused__softmax_backward_data_masked_fill_mul_view_2_5ec13c63, ind:triton_poi_fused_index_sum_0_a92d64b0, bch:median_small_flat_kernel_3b4cee92 (+650) |
| HIT-0003 | crash-assert | 604 | 90 | `ttgir` | `llvm-error:Unexpected parent op of block argument` | bch:upsample_trilinear3d_kernel_6c02f6c8, bch:upsample_nearest3d_kernel_a4b3312b, bch:gcd_kernel_i16_5748aa00 (+601) |
| HIT-0019 | crash-assert | 558 | 90 | `ttgir` | `assert@PlanCTA.cpp:164:step < maxSteps && "Maximum number of steps exceeded"` | bch:_small_jacobi_svd_kernel_791182a2, bch:_fused_backward_kernel_dbedd307, bch:max_pool3d_forward_kernel_1e97696b (+555) |
| HIT-0007 | invalid-ir-frontend | 544 | 80,89,90 | `frontend` | `verify:'tt.store' op failed to verify that value type matches ptr type` | bch:reshape_and_cache_flash_kernel_476250f6, bch:pixel_unshuffle_kernel_4e20b588, bch:narrow_copy_kernel_325c2faa (+541) |
| HIT-0035 | invalid-ir-frontend | 484 | 80,89,90 | `frontend` | `verify:'arith.<float-op>' op operand #N must be floating-point-like, but got 'tensor<>'` | bch:_fused_adam_kernel_fa131bdd, bch:mean_dim_kernel_non_inner_vec_42dc3a7b, ind:triton_per_fused_mul_sum_0_f904167e (+481) |
| HIT-0012 | crash-assert | 444 | 80,89,90 | `llir` | `llvm-error:element type mismatch when packing elements for LLVM struct` | bch:_complex_svd_pick_orthonormal_v_kernel_e65b022b, ind:triton_per_fused_native_layer_norm_native_layer_norm_backward_1_437059a9, bch:argmin_kernel_2_edc08764 (+441) |
| - | timeout | 392 | 80,89,90 | `llir` | `timeout@llir` | bch:_complete_svd_factor_kernel_e266834b, bch:grid_sample_3d_trilinear_zeros_tiled_kernel_717eb6ff, bch:log_softmax_backward_kernel_537d7ace (+389) |
| HIT-0017 | crash-assert | 384 | 90 | `llir` | `assert@ScanOpToLLVM.cpp:159:numScanBlocks * numParallelBlocks * parallelElementsPerThread * scanElementsPerThreads == sr` | ind:triton_per_fused_cumsum_0_71300b0d, ind:triton_per_fused_cummax_0_6dc3f614, ind:triton_red_fused_cumprod_0_9c7c3568 (+381) |
| HIT-0005 | crash-assert | 315 | 90 | `ttgir` | `assert@PlanCTA.cpp:359:stores.size() > N && "Cannot find store-like ops"` | bch:_assert_async_kernel_ffc659d7, ind:triton_per_fused__assert_async_all_ge_lt_0_69d7fda8, bch:_conv_transpose2d_scatter_no_overlap_kernel_d1a6faca (+312) |
| HIT-0013 | invalid-ir | 294 | 90 | `ttgir` | `mlirerr:Result has an invalid layout: #ttg.blocked<>.` | ind:triton_per_fused_native_layer_norm_native_layer_norm_backward_0_92be64e3, bch:median_f16_strided_key_select_kernel_70e258cd, bch:_cp_gather_indexer_quant_cache_kernel_d5cecde2 (+291) |
| HIT-0034 | invalid-ir-frontend | 269 | 80,89,90 | `frontend` | `verify:'arith.<float-op>' op operand #N must be floating-point-like, but got 'iN'` | ind:triton_per_fused_native_group_norm_0_9294c0a8, bch:layernorm_kernel_85da57b8, ind:triton_per_fused_native_layer_norm_0_23b8aeb1 (+266) |
| HIT-0016 | crash-assert | 147 | 90 | `llir` | `assert@ScanOpToLLVM.cpp:265:numScanBlocks * numParallelBlocks * parallelElementsPerThread * scanElementsPerThreads == sr` | ind:triton_per_fused_logcumsumexp_0_1f363aee, ind:triton_per_fused_cummax_0_6dc3f614, ind:triton_per_fused_cumsum_0_71300b0d (+144) |
| HIT-0014 | crash-assert | 116 | 90 | `ttgir` | `llvm-error:Unexpected parent op of YieldOp` | ind:triton_poi_fused_add_bucketize_0_8b33912d (+113) |
| HIT-0009 | crash-assert | 95 | 89,90 | `llir` | `assert@ElementwiseOpToLLVM.cpp:482:roundingMode.has_value() && "Rounding mode must be specified for convertsions to fpN"` | bch:special_i1_kernel_7e502f07, ind:triton_poi_fused__to_copy_clamp_mul_0_3cc69ca3, bch:output_counts_flat_kernel_6fe8ff3b (+92) |
| HIT-0022 | crash-assert | 56 | 80,89,90 | `llir` | `llvm-error:out of memory` | bch:sum_dim_kernel_1a10795a, bch:triton_red_fused_native_layer_norm_0_4dd87e5b, bch:matmul_kernel_7df0cce6 (+53) |
| HIT-0032 | crash-ptxas | 50 | 80,89,90 | `cubin` | `ptxas-fatal:ptxas fatal : Memory allocation failure` | bch:matmul_kernel_d953b7cf, bch:conv_transpose1d_forward_kernel_95ad226e, bch:_complex_svd_pick_orthonormal_v_kernel_e65b022b (+47) |
| HIT-0039 | invalid-ir-frontend | 47 | 80,89,90 | `frontend` | `verify:'arith.addf' op requires the same type for all operands and results` | ind:triton_for_fused_0_ca70313c, ind:triton_poi_fused_add_mul_sub_0_d6a0033c, ind:triton_per_fused_add_native_layer_norm_2_97c454b3 (+44) |
| HIT-0046 | invalid-ir-frontend | 46 | 80,89,90 | `frontend` | `verify:'arith.truncf' op result #N must be floating-point-like, but got 'iN'` | ind:triton_per_fused_prod_0_60e2ed77, bch:_fused_adam_kernel_fa131bdd (+43) |
| HIT-0037 | timeout | 42 | 80,89,90 | `ptx` | `timeout@ptx` | bch:mhc_split_sinkhorn_kernel_generic_caf7808f, bch:sparse_semi_structured_mm_kernel_d23446af, bch:conv3d_forward_kernel_79b91f9d (+39) |
| HIT-0043 | invalid-ir | 42 | 80,89,90 | `llir` | `verify:'llvm.fptrunc' op result #N must be floating point LLVM type or LLVM dialect-compatible vector of floating point ` | bch:_fused_adam_kernel_fa131bdd (+39) |
| HIT-0024 | invalid-ir-frontend | 37 | 80,89,90 | `frontend` | `verify:'arith.mulf' op requires the same type for all operands and results` | ind:triton_per_fused_add_div_expand_mul_pow_sum_1_ffb224c0, bch:softmax_bwd_kernel_04ce474d, ind:triton_per_fused__softmax_backward_data_masked_fill_mul_view_2_5ec13c63 (+34) |
| HIT-0023 | crash-signal | 30 | 80,89,90 | `llir` | `SIGIOT@llir` | bch:sum_dim_kernel_1a10795a, bch:matmul_kernel_d953b7cf, bch:grouped_matmul_kernel_87cbfee0 (+27) |
| HIT-0048 | invalid-ir-frontend | 27 | 80,89,90 | `frontend` | `verify:'tt.int_to_ptr' op operand #N must be N-bit signless integer or tensor of N-bit signless integer values, but got ` | bch:grouped_matmul_kernel_87cbfee0 (+24) |
| HIT-0041 | invalid-ir-frontend | 22 | 80,89,90 | `frontend` | `verify:'arith.select' op operand #N must be bool-like, but got 'tensor<>'` | bch:_triton_dropout_eef0b8a0 (+19) |
| HIT-0045 | crash-assert | 19 | 90 | `ttgir` | `assert@StorageUniquerSupport.h:180:succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))` | bch:_cp_gather_indexer_quant_cache_kernel_d5cecde2 (+16) |
| HIT-0053 | invalid-ir-frontend | 14 | 80,89,90 | `frontend` | `verify:'arith.subf' op requires the same type for all operands and results` | ind:triton_per_fused__log_softmax_backward_data_0_eb6eb3c3, ind:triton_per_fused_native_layer_norm_native_layer_norm_backward_1_437059a9, ind:triton_red_fused_native_batch_norm_backward_0_0ec65333 (+11) |
| HIT-0004 | invalid-ir | 13 | 80,89,90 | `llir` | `verify:'llvm.fcmp' op operand #N must be floating point LLVM type or LLVM dialect-compatible vector of floating point LL` | bch:argmin_kernel_f8fcbf2b, bch:max_pool2d_forward_kernel_8b2aab1a, bch:_cyclic_jacobi_finalize_kernel_4665c432 (+10) |
| HIT-0036 | crash-ptxas | 9 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Register allocation failed with register count of 'N'. Compile the program with a higher ` | ind:triton_flex_attention_backward_2c768ac2 (+6) |
| HIT-0015 | invalid-ir-frontend | 9 | 80,89,90 | `frontend` | `verify:'arith.addf' op operand #N must be floating-point-like, but got 'iN'` | bch:moe_align_block_size_stage2_vec_88dc7283, ind:triton_per_fused_native_layer_norm_0_23b8aeb1, ind:triton_per_fused_native_group_norm_0_9294c0a8 (+6) |
| HIT-0056 | crash-python | 7 | 80 | `irverify` | `IRVerifyFailure@irverify:pass tritongpu-coalesce (#N of make_ttgir) exit N` | bch:_small_jacobi_svals_kernel_f9679a95, bch:_complete_svd_factor_kernel_e266834b, bch:max_pool3d_forward_kernel_1e97696b (+4) |
| HIT-0002 | invalid-ir-frontend | 6 | 80,89,90 | `frontend` | `verify:'arith.cmpf' op operand #N must be floating-point-like, but got 'iN'` | bch:argmin_kernel_1_60cfa9bd, bch:argmax_kernel_1_7322e48b (+3) |
| HIT-0029 | invalid-ir-frontend | 6 | 80,89,90 | `frontend` | `verify:'arith.cmpf' op operand #N must be floating-point-like, but got 'tensor<>'` | bch:max_pool3d_forward_kernel_1e97696b, bch:median_small_dim_kernel_0f44d526 (+3) |
| HIT-0055 | invalid-ir-frontend | 6 | 80,89,90 | `frontend` | `verify:'tt.splat' op requires the same element type for all operands and results` | ind:triton_mm_4942c5ab (+3) |
| HIT-0051 | crash-signal | 5 | 80,89,90 | `ptx` | `SIGIOT@ptx` | bch:matmul_kernel_10e0e451 (+2) |
| HIT-0044 | crash-ptxas | 5 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function sparse_attn_trito` | bch:sparse_attn_triton_kernel_cc5e6c13 (+2) |
| HIT-0066 | crash-python | 4 | 80,89,90 | `ttgir` | `RuntimeError@compiler.py:371:PassManager::run failed` | bch:reflection_pad2d_kernel_00c19208 (+1) |
| HIT-0049 | crash-assert | 4 | 90 | `ttgir` | `llvm-error:Layout conflict for block arg` | bch:bmm_kernel_a62513ac (+1) |
| HIT-0021 | invalid-ir | 4 | 90 | `ttgir` | `mlirerr:Result has an invalid layout: #ttg.linear<>.` | bch:median_small_flat_kernel_3b4cee92, bch:median_small_dim_kernel_0f44d526 (+1) |
| HIT-0026 | crash-ptxas | 4 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function convNd_forward_ke` | bch:conv3d_forward_kernel_79b91f9d (+1) |
| HIT-0018 | invalid-ir | 4 | 80,89,90 | `llir` | `verify:'llvm.fsub' op operand #N must be floating point LLVM type or LLVM dialect-compatible vector of floating point LL` | bch:diff_kernel_1d_3317231b (+1) |
| HIT-0030 | invalid-ir | 4 | 80,89,90 | `llir` | `verify:'llvm.intr.minnum' op operand #N must be floating point LLVM type or LLVM dialect-compatible vector of floating p` | bch:amin_kernel_b5df8e51 (+1) |
| HIT-0064 | invalid-ir | 3 | 90 | `ttgir` | `mlirerr:Expected result encoding #ttg.linear<> but was #ttg.blocked<>` | bch:mode_kernel_b978d037 |
| HIT-0052 | crash-ptxas | 3 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function linear_kernel_cNa` | bch:linear_kernel_c39a1dbd |
| HIT-0070 | invalid-ir-frontend | 3 | 80,89,90 | `frontend` | `verify:'arith.minnumf' op requires the same type for all operands and results` | bch:fmin_kernel_83e46d9f |
| HIT-0028 | invalid-ir-frontend | 3 | 80,89,90 | `frontend` | `verify:'arith.mulf' op operand #N must be floating-point-like, but got 'tensor<>'` | ind:triton_per_fused_linalg_vector_norm_0_c7c12e54 |
| HIT-0025 | invalid-ir-frontend | 3 | 80,89,90 | `frontend` | `verify:'arith.subf' op operand #N must be floating-point-like, but got 'tensor<>'` | bch:diff_kernel_1d_3317231b |
| HIT-0010 | invalid-ir | 3 | 89,90 | `llir` | `verify:'llvm.fmul' op operand #N must be floating point LLVM type or LLVM dialect-compatible vector of floating point LL` | ind:triton_per_fused_add_div_expand_mul_pow_sum_1_ffb224c0 |
| HIT-0027 | invalid-ir | 3 | 89,90 | `llir` | `verify:'llvm.fneg' op operand #N must be floating point LLVM type or LLVM dialect-compatible vector of floating point LL` | bch:_fused_adam_kernel_fa131bdd |
| HIT-0031 | invalid-ir | 3 | 89,90 | `llir` | `verify:'llvm.intr.maxnum' op operand #N must be floating point LLVM type or LLVM dialect-compatible vector of floating p` | bch:amax_kernel_be15d5b2 |
| HIT-0059 | invalid-ir-frontend | 3 | 80,89,90 | `frontend` | `verify:'tt.atomic_rmw' op failed to verify that ptr type matches value type` | ind:triton_poi_fused_scatter_add_1_2ca45926 |
| HIT-0063 | crash-assert | 2 | 90 | `llir` | `assert@MMAHelpers.h:171:block == N` | ind:triton_mm_cb5ec818 |
| HIT-0038 | crash-assert | 2 | 90 | `llir` | `assert@ViewOpToLLVM.cpp:246:!isExpensiveView(op.getSrc().getType(), op.getType())` | bch:mode_kernel_b978d037 |
| HIT-0062 | invalid-ir | 2 | 90 | `llir` | `mlirerr:cp.async does not support non-trivial block dimension` | ind:triton_mm_cb5ec818 |
| HIT-0054 | crash-ptxas | 2 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function matmuladd_kernel_` | bch:matmuladd_kernel_7b7e7452 |
| HIT-0060 | crash-python | 1 | 89 | `cubin` | `PTXASError@compiler.py:630:Repro command: ptxas -lineinfo -suppress-debug-info --fmad=false -v --regAllocOptLevel=N --gp` | bch:median_bool_reduce_counts_kernel_589d0d2a |
| HIT-0067 | crash-python | 1 | 89 | `cubin` | `PTXASError@compiler.py:630:Repro command: ptxas -lineinfo -suppress-debug-info --fmad=false -v --regAllocOptLevel=N --gp` | ind:triton_per_fused_sum_1_0da31e31 |
| HIT-0068 | crash-python | 1 | 89 | `cubin` | `PTXASError@compiler.py:630:Repro command: ptxas -lineinfo -suppress-debug-info --fmad=false -v --regAllocOptLevel=N --gp` | ind:triton_per_fused_sum_1_0da31e31 |
| HIT-0047 | crash-python | 1 | 89 | `cubin` | `PTXASError@compiler.py:630:Repro command: ptxas -lineinfo -suppress-debug-info --fmad=false -v --regAllocOptLevel=N --gp` | ind:triton_per_fused_native_group_norm_0_9294c0a8 |
| HIT-0050 | crash-python | 1 | 90 | `cubin` | `PTXASError@compiler.py:630:Repro command: ptxas -lineinfo -suppress-debug-info --fmad=false -v --regAllocOptLevel=N --gp` | ind:triton_per_fused_logcumsumexp_0_1f363aee |
| HIT-0071 | crash-python | 1 | 89 | `cubin` | `PTXASError@compiler.py:630:Repro command: ptxas -lineinfo -suppress-debug-info -v --regAllocOptLevel=N --gpu-name=sm_N t` | ind:triton_per_fused_add_mean_mul_pow_rsqrt_0_5474b763 |
| HIT-0065 | crash-python | 1 | 89 | `cubin` | `PTXASError@compiler.py:630:Repro command: ptxas -lineinfo -suppress-debug-info -v --regAllocOptLevel=N --gpu-name=sm_N t` | ind:triton_per_fused_native_group_norm_0_9294c0a8 |
| HIT-0069 | crash-ptxas | 1 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function _conv_transposeNd` | bch:_conv_transpose2d_residue_kernel_bc97b619 |
| HIT-0020 | crash-ptxas | 1 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function _conv_transposeNd` | bch:_conv_transpose2d_stride2_pad1_3x3_kernel_312550c9 |
| HIT-0042 | crash-ptxas | 1 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function _matmul_partition` | bch:_matmul_partition_k_9ee5ee96 |
| HIT-0061 | crash-ptxas | 1 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function matmul_kernel_Ndf` | bch:matmul_kernel_7df0cce6 |
| HIT-0058 | crash-ptxas | 1 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function matmul_kernel_NeN` | bch:matmul_kernel_10e0e451 |
| HIT-0040 | crash-ptxas | 1 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function mm_kernel_general` | bch:mm_kernel_general_64b5afae |
| HIT-0057 | crash-ptxas | 1 | 90 | `cubin` | `ptxas-fatal:ptxas fatal : (CN) Insufficient registers (N) to compile instruction at line N in function mm_kernel_splitk_` | bch:mm_kernel_splitk_d10c35a0 |

## Reviewed reports — R4 leaves the repo copy alone

A reviewing agent rewrote `tv/bug-report/triton/r2-compile/HIT-*.md` for these classes, so R4 no longer regenerates that file and the count printed inside it is frozen at review time. The live count is here, and the full regenerated report with the same count is in this line's own `HITS/` mirror.

| report | kind | count | archs | examples |
|---|---|---|---|---|
| HIT-0001 | invalid-ir | 1863 | 90 | bch:logsumexp_kernel_non_inner_baaf36b4, bch:_complete_svd_factor_kernel_e266834b, ind:triton_unk_fused_amax_sum_0_8525b515 |
| HIT-0008 | invalid-ir-frontend | 1025 | 80,89,90 | bch:_rank2_svd_kernel_813eb03e, bch:_as_strided_copy_1d_kernel_d11a073d, ind:triton_poi_fused_add_mul_relu_sigmoid_sub_tanh_0_e1f7d5c0 |
| HIT-0006 | crash-assert | 827 | 90 | ind:triton_flex_attention_backward_2c768ac2, bch:_conv_transpose2d_residue_kernel_bc97b619, ind:triton_flex_attention_65dbdfce |
| HIT-0003 | crash-assert | 604 | 90 | bch:upsample_trilinear3d_kernel_6c02f6c8, bch:upsample_nearest3d_kernel_a4b3312b, bch:gcd_kernel_i16_5748aa00 |
| HIT-0019 | crash-assert | 558 | 90 | bch:_small_jacobi_svd_kernel_791182a2, bch:_fused_backward_kernel_dbedd307, bch:max_pool3d_forward_kernel_1e97696b |
| HIT-0007 | invalid-ir-frontend | 544 | 80,89,90 | bch:reshape_and_cache_flash_kernel_476250f6, bch:pixel_unshuffle_kernel_4e20b588, bch:narrow_copy_kernel_325c2faa |
| HIT-0005 | crash-assert | 315 | 90 | bch:_assert_async_kernel_ffc659d7, ind:triton_per_fused__assert_async_all_ge_lt_0_69d7fda8, bch:_conv_transpose2d_scatter_no_overlap_kernel_d1a6faca |
| HIT-0016 | crash-assert | 147 | 90 | ind:triton_per_fused_logcumsumexp_0_1f363aee, ind:triton_per_fused_cummax_0_6dc3f614, ind:triton_per_fused_cumsum_0_71300b0d |
| HIT-0009 | crash-assert | 95 | 89,90 | bch:special_i1_kernel_7e502f07, ind:triton_poi_fused__to_copy_clamp_mul_0_3cc69ca3, bch:output_counts_flat_kernel_6fe8ff3b |
| HIT-0048 | invalid-ir-frontend | 27 | 80,89,90 | bch:grouped_matmul_kernel_87cbfee0 |
| HIT-0045 | crash-assert | 19 | 90 | bch:_cp_gather_indexer_quant_cache_kernel_d5cecde2 |
| HIT-0004 | invalid-ir | 13 | 80,89,90 | bch:argmin_kernel_f8fcbf2b, bch:max_pool2d_forward_kernel_8b2aab1a, bch:_cyclic_jacobi_finalize_kernel_4665c432 |
| HIT-0002 | invalid-ir-frontend | 6 | 80,89,90 | bch:argmin_kernel_1_60cfa9bd, bch:argmax_kernel_1_7322e48b |
| HIT-0055 | invalid-ir-frontend | 6 | 80,89,90 | ind:triton_mm_4942c5ab |
| HIT-0066 | crash-python | 4 | 80,89,90 | bch:reflection_pad2d_kernel_00c19208 |
| HIT-0049 | crash-assert | 4 | 90 | bch:bmm_kernel_a62513ac |
| HIT-0063 | crash-assert | 2 | 90 | ind:triton_mm_cb5ec818 |
| HIT-0038 | crash-assert | 2 | 90 | bch:mode_kernel_b978d037 |
| HIT-0047 | crash-python | 1 | 89 | ind:triton_per_fused_native_group_norm_0_9294c0a8 |
| HIT-0050 | crash-python | 1 | 90 | ind:triton_per_fused_logcumsumexp_0_1f363aee |

## Rejected reports — R4 does not re-create the repo copy

A reviewing agent rejected these and moved the report to `/home/youngzt/tv/bug-report-rejected/triton/r2-compile` (`/home/youngzt/tv/bug-report-rejected/REJECTED.md` holds the criterion). R4 keeps counting the class but no longer writes its `HIT-*.md` or its `HIT-*/` artifacts into `/home/youngzt/tv/triton/tv/bug-report/triton/r2-compile`. The live count is here, and the full regenerated report with the same count is in this line's own `HITS/` mirror.

| report | kind | count | archs | rejected as | examples |
|---|---|---|---|---|---|
| HIT-0011 | crash-assert | 728 | 80,89,90 | B4 (of triton/r2-compile/HIT-0009) | ind:triton_poi_fused__softmax_4_04f5cb59, bch:mhc_split_sinkhorn_kernel_hcmult_4_5187f1d9, ind:triton_per_fused_mul_sum_0_f904167e |
| HIT-0033 | invalid-ir | 653 | 80,89,90 | B4 (of triton/r2-compile/HIT-0004) | ind:triton_per_fused__softmax_backward_data_masked_fill_mul_view_2_5ec13c63, ind:triton_poi_fused_index_sum_0_a92d64b0, bch:median_small_flat_kernel_3b4cee92 |
| HIT-0035 | invalid-ir-frontend | 484 | 80,89,90 | B4 (of triton/r2-compile/HIT-0002) | bch:_fused_adam_kernel_fa131bdd, bch:mean_dim_kernel_non_inner_vec_42dc3a7b, ind:triton_per_fused_mul_sum_0_f904167e |
| HIT-0012 | crash-assert | 444 | 80,89,90 | B4 (of triton/r2-compile/HIT-0004) | bch:_complex_svd_pick_orthonormal_v_kernel_e65b022b, ind:triton_per_fused_native_layer_norm_native_layer_norm_backward_1_437059a9, bch:argmin_kernel_2_edc08764 |
| HIT-0017 | crash-assert | 384 | 90 | B4 (of triton/r2-compile/HIT-0016) | ind:triton_per_fused_cumsum_0_71300b0d, ind:triton_per_fused_cummax_0_6dc3f614, ind:triton_red_fused_cumprod_0_9c7c3568 |
| HIT-0013 | invalid-ir | 294 | 90 | B4 (of triton/r2-compile/HIT-0001) | ind:triton_per_fused_native_layer_norm_native_layer_norm_backward_0_92be64e3, bch:median_f16_strided_key_select_kernel_70e258cd, bch:_cp_gather_indexer_quant_cache_kernel_d5cecde2 |
| HIT-0034 | invalid-ir-frontend | 269 | 80,89,90 | B4 (of triton/r2-compile/HIT-0002) | ind:triton_per_fused_native_group_norm_0_9294c0a8, bch:layernorm_kernel_85da57b8, ind:triton_per_fused_native_layer_norm_0_23b8aeb1 |
| HIT-0014 | crash-assert | 116 | 90 | B4 (of triton/r2-compile/HIT-0003) | ind:triton_poi_fused_add_bucketize_0_8b33912d |
| HIT-0022 | crash-assert | 56 | 80,89,90 | A1 | bch:sum_dim_kernel_1a10795a, bch:triton_red_fused_native_layer_norm_0_4dd87e5b, bch:matmul_kernel_7df0cce6 |
| HIT-0032 | crash-ptxas | 50 | 80,89,90 | A1 | bch:matmul_kernel_d953b7cf, bch:conv_transpose1d_forward_kernel_95ad226e, bch:_complex_svd_pick_orthonormal_v_kernel_e65b022b |
| HIT-0039 | invalid-ir-frontend | 47 | 80,89,90 | B4 (of triton/r2-compile/HIT-0007) | ind:triton_for_fused_0_ca70313c, ind:triton_poi_fused_add_mul_sub_0_d6a0033c, ind:triton_per_fused_add_native_layer_norm_2_97c454b3 |
| HIT-0046 | invalid-ir-frontend | 46 | 80,89,90 | B4 (of triton/r2-compile/HIT-0002) | ind:triton_per_fused_prod_0_60e2ed77, bch:_fused_adam_kernel_fa131bdd |
| HIT-0037 | timeout | 42 | 80,89,90 | A1 | bch:mhc_split_sinkhorn_kernel_generic_caf7808f, bch:sparse_semi_structured_mm_kernel_d23446af, bch:conv3d_forward_kernel_79b91f9d |
| HIT-0043 | invalid-ir | 42 | 80,89,90 | B4 (of triton/r2-compile/HIT-0004) | bch:_fused_adam_kernel_fa131bdd |
| HIT-0024 | invalid-ir-frontend | 37 | 80,89,90 | B4 (of triton/r2-compile/HIT-0007) | ind:triton_per_fused_add_div_expand_mul_pow_sum_1_ffb224c0, bch:softmax_bwd_kernel_04ce474d, ind:triton_per_fused__softmax_backward_data_masked_fill_mul_view_2_5ec13c63 |
| HIT-0023 | crash-signal | 30 | 80,89,90 | A1 | bch:sum_dim_kernel_1a10795a, bch:matmul_kernel_d953b7cf, bch:grouped_matmul_kernel_87cbfee0 |
| HIT-0041 | invalid-ir-frontend | 22 | 80,89,90 | B4 (of triton/r2-compile/HIT-0007) | bch:_triton_dropout_eef0b8a0 |
| HIT-0053 | invalid-ir-frontend | 14 | 80,89,90 | B4 (of triton/r2-compile/HIT-0002) | ind:triton_per_fused__log_softmax_backward_data_0_eb6eb3c3, ind:triton_per_fused_native_layer_norm_native_layer_norm_backward_1_437059a9, ind:triton_red_fused_native_batch_norm_backward_0_0ec65333 |
| HIT-0036 | crash-ptxas | 9 | 90 | not a defect (resource-limit) | ind:triton_flex_attention_backward_2c768ac2 |
| HIT-0015 | invalid-ir-frontend | 9 | 80,89,90 | B4 (of triton/r2-compile/HIT-0002) | bch:moe_align_block_size_stage2_vec_88dc7283, ind:triton_per_fused_native_layer_norm_0_23b8aeb1, ind:triton_per_fused_native_group_norm_0_9294c0a8 |
| HIT-0056 | crash-python | 7 | 80 | A1 | bch:_small_jacobi_svals_kernel_f9679a95, bch:_complete_svd_factor_kernel_e266834b, bch:max_pool3d_forward_kernel_1e97696b |
| HIT-0029 | invalid-ir-frontend | 6 | 80,89,90 | B4 (of triton/r2-compile/HIT-0002) | bch:max_pool3d_forward_kernel_1e97696b, bch:median_small_dim_kernel_0f44d526 |
| HIT-0051 | crash-signal | 5 | 80,89,90 | A1 | bch:matmul_kernel_10e0e451 |
| HIT-0044 | crash-ptxas | 5 | 90 | not a defect (resource-limit) | bch:sparse_attn_triton_kernel_cc5e6c13 |
| HIT-0021 | invalid-ir | 4 | 90 | B4 (of triton/r2-compile/HIT-0001) | bch:median_small_flat_kernel_3b4cee92, bch:median_small_dim_kernel_0f44d526 |
| HIT-0026 | crash-ptxas | 4 | 90 | not a defect (resource-limit) | bch:conv3d_forward_kernel_79b91f9d |
| HIT-0018 | invalid-ir | 4 | 80,89,90 | B4 (of triton/r2-compile/HIT-0004) | bch:diff_kernel_1d_3317231b |
| HIT-0030 | invalid-ir | 4 | 80,89,90 | B4 (of triton/r2-compile/HIT-0004) | bch:amin_kernel_b5df8e51 |
| HIT-0064 | invalid-ir | 3 | 90 | B4 (of triton/r2-compile/HIT-0038) | bch:mode_kernel_b978d037 |
| HIT-0052 | crash-ptxas | 3 | 90 | not a defect (resource-limit) | bch:linear_kernel_c39a1dbd |
| HIT-0070 | invalid-ir-frontend | 3 | 80,89,90 | B4 (of triton/r2-compile/HIT-0007) | bch:fmin_kernel_83e46d9f |
| HIT-0028 | invalid-ir-frontend | 3 | 80,89,90 | B4 (of triton/r2-compile/HIT-0002) | ind:triton_per_fused_linalg_vector_norm_0_c7c12e54 |
| HIT-0025 | invalid-ir-frontend | 3 | 80,89,90 | B4 (of triton/r2-compile/HIT-0002) | bch:diff_kernel_1d_3317231b |
| HIT-0010 | invalid-ir | 3 | 89,90 | B4 (of triton/r2-compile/HIT-0004) | ind:triton_per_fused_add_div_expand_mul_pow_sum_1_ffb224c0 |
| HIT-0027 | invalid-ir | 3 | 89,90 | B4 (of triton/r2-compile/HIT-0004) | bch:_fused_adam_kernel_fa131bdd |
| HIT-0031 | invalid-ir | 3 | 89,90 | B4 (of triton/r2-compile/HIT-0004) | bch:amax_kernel_be15d5b2 |
| HIT-0059 | invalid-ir-frontend | 3 | 80,89,90 | B4 (of triton/r2-compile/HIT-0007) | ind:triton_poi_fused_scatter_add_1_2ca45926 |
| HIT-0062 | invalid-ir | 2 | 90 | B4 (of triton/r2-compile/HIT-0063) | ind:triton_mm_cb5ec818 |
| HIT-0054 | crash-ptxas | 2 | 90 | not a defect (resource-limit) | bch:matmuladd_kernel_7b7e7452 |
| HIT-0060 | crash-python | 1 | 89 | B4 (of triton/r2-compile/HIT-0047) | bch:median_bool_reduce_counts_kernel_589d0d2a |
| HIT-0067 | crash-python | 1 | 89 | B4 (of triton/r2-compile/HIT-0047) | ind:triton_per_fused_sum_1_0da31e31 |
| HIT-0068 | crash-python | 1 | 89 | B4 (of triton/r2-compile/HIT-0047) | ind:triton_per_fused_sum_1_0da31e31 |
| HIT-0071 | crash-python | 1 | 89 | B4 (of triton/r2-compile/HIT-0047) | ind:triton_per_fused_add_mean_mul_pow_rsqrt_0_5474b763 |
| HIT-0065 | crash-python | 1 | 89 | B4 (of triton/r2-compile/HIT-0047) | ind:triton_per_fused_native_group_norm_0_9294c0a8 |
| HIT-0069 | crash-ptxas | 1 | 90 | not a defect (resource-limit) | bch:_conv_transpose2d_residue_kernel_bc97b619 |
| HIT-0020 | crash-ptxas | 1 | 90 | not a defect (resource-limit) | bch:_conv_transpose2d_stride2_pad1_3x3_kernel_312550c9 |
| HIT-0042 | crash-ptxas | 1 | 90 | not a defect (resource-limit) | bch:_matmul_partition_k_9ee5ee96 |
| HIT-0061 | crash-ptxas | 1 | 90 | not a defect (resource-limit) | bch:matmul_kernel_7df0cce6 |
| HIT-0058 | crash-ptxas | 1 | 90 | not a defect (resource-limit) | bch:matmul_kernel_10e0e451 |
| HIT-0040 | crash-ptxas | 1 | 90 | not a defect (resource-limit) | bch:mm_kernel_general_64b5afae |
| HIT-0057 | crash-ptxas | 1 | 90 | not a defect (resource-limit) | bch:mm_kernel_splitk_d10c35a0 |
