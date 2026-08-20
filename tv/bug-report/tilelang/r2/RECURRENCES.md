# TileLang round 2 (R5) -- every difference signature seen

One row per distinct (minimal pass set, pass_configs keys, family, strength).
`n` counts every (instance, input mode, variant) occurrence, not just the
reported one.  The full per-occurrence records are in
`/home/youngzt/fuzz-tilelang-r2/run/results.jsonl`.

updated 2026-08-20 12:33:14

The `review verdict` column is the reviewer's disposition of the report that
represents the signature, under `tv/bug-report/REVIEW-CRITERIA.md`.  Only the
**KEPT** rows still have a report in this directory; every `rejected` row's
report and artifacts were moved to
`/home/youngzt/tv/bug-report-rejected/tilelang/r2/`, where `REJECTED.md` gives
the reason.  `-> r1 X` / `-> r2 X` after a B4 names the report the duplicate
folds into (`r1` = round 1, in `tv/bug-report/tilelang/`).

A large `n` on a rejected row is still worth reading: it says how wide that
class is across the whole sweep, which is evidence no single report carries.

The line is stopped, so this table is final.

| n | report | minimal pass set | pass_configs | family | strength | example instances | review verdict |
|---|---|---|---|---|---|---|---|
| 453 | HIT-0010 | `ProducerConsumerWarpSpecialized` | `-` | fa | - | gen_fa_3ad7de74cd3954, gen_fa_b6d78c9040bcba, gen_fa_7026bc61bdb6b8 | rejected B3 |
| 114 | HIT-0002 | `-` | `tl.config_index_bitwidth` | cast | gross, pass-class:bit-preserving | gen_cast_ab422754d7e4a1, gen_cast_92916ba4444ab2, gen_cast_4431c23412dc4d | rejected B4 -> r1 HIT-0001 |
| 86 | HIT-0103 | `ProducerConsumerWarpSpecialized` | `-` | fa | gross | gen_fa_6a2b83e163628f, gen_fa_bf4da4c1c8fb08, gen_fa_7e57082f209549 | rejected B3 |
| 51 | HIT-0001 | `ConfigIndexBitwidth` | `-` | cast | gross, pass-class:bit-preserving | gen_cast_ab422754d7e4a1, gen_cast_6e30699efaa193, gen_cast_595fce049672ed | rejected B4 -> r1 HIT-0001 |
| 36 | HIT-0012 | `-` | `tl.disable_wgmma` | cap | gross, pass-class:bit-preserving | cap_2bae46f31c2f27cd, cap_59dae696e2a0250f, cap_8499fd7090826a59 | rejected B1 + B2 |
| 26 | HIT-0011 | `-` | `tl.disable_warp_specialized` | fa | - | gen_fa_3ad7de74cd3954, gen_fa_b6d78c9040bcba, gen_fa_7026bc61bdb6b8 | rejected B3 (also B4 -> r2 HIT-0010 |
| 14 | HIT-0003 | `ProducerConsumerWarpSpecialized` | `-` | fa | exact-int-accum:reassociation-excluded | gen_fa_b79103583e1427, gen_fa_a53246704cb34f, gen_fa_b157c0e8637659 | rejected B3 |
| 12 | HIT-0035 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | cumsum | - | gen_cumsum_6a08015c14f6ca, gen_cumsum_079e004477eb76, gen_cumsum_d17b14d8fc2c6f | rejected B1 |
| 11 | HIT-0004 | `-` | `tl.disable_warp_specialized` | fa | exact-int-accum:reassociation-excluded | gen_fa_b79103583e1427, gen_fa_a53246704cb34f, gen_fa_b157c0e8637659 | rejected B3 (also B4 -> r2 HIT-0003 |
| 11 | HIT-0144 | `-` | `tl.enable_aggressive_shared_memory_merge` | rr | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | gen_rr_2bbbfaa14043e0, gen_rr_ac1639fe6ecf3e, gen_rr_185b66d1580a52 | rejected B4 -> r1 HIT-0040 |
| 9 | HIT-0005 | `-` | `tl.config_index_bitwidth` | cap | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | cap_081a26caff856329, cap_bde15a1202582b07, cap_401ab8ec5f03638a | rejected B4 -> r1 HIT-0001 |
| 9 | HIT-0014 | `StorageRewrite` | `tl.enable_fast_math` | layernorm | - | gen_layernorm_ffdc97a9b5025d, gen_layernorm_a7bd611d5215b4, gen_layernorm_658ad3b132752a | rejected B1 |
| 9 | HIT-0032 | `-` | `tl.config_index_bitwidth` | cast | gross | gen_cast_c6ca1d1d65892e, gen_cast_d06df2123163b4, gen_cast_84a8d6148d5f7e | rejected B4 -> r1 HIT-0001 |
| 8 | HIT-0072 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | cap | - | cap_6c8b6b1f9ea4e6de, cap_804640f4df005517, cap_91b5b330efbdb4d9 | rejected B1 |
| 8 | HIT-0091 | `ProducerConsumerWarpSpecialized` | `-` | gemv | exact-int-accum:reassociation-excluded, nan-pattern-changed | gen_gemv_e030b819eb9800, gen_gemv_b1835c6de37125 | rejected B4 -> r2 HIT-0008 |
| 7 | HIT-0013 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg_predicated` | cumsum | - | gen_cumsum_d93c8b5eedecce, gen_cumsum_d17b14d8fc2c6f, gen_cumsum_1bdf59a66c06ca | rejected B1 |
| 7 | HIT-0018 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | rr | - | gen_rr_eeae4584abb681, gen_rr_f6a3904f26c508, gen_rr_f9a48fa1ede396 | rejected B1 |
| 7 | HIT-0034 | `Simplify#5` | `tl.enable_fast_math` | rr | - | gen_rr_71e2a24d872c43, gen_rr_935cf4ad0fc70d, gen_rr_321831a18672b2 | rejected B1 |
| 7 | HIT-0148 | `ProducerConsumerWarpSpecialized` | `-` | gemv | - | gen_gemv_4da8042d09cfec | rejected B3 |
| 6 | HIT-0007 | `ConfigIndexBitwidth` | `-` | gemv | pass-class:bit-preserving | gen_gemv_ef525af0d65e95, gen_gemv_3cdc68e3131074 | rejected B4 -> r2 HIT-0008 |
| 6 | HIT-0008 | `ProducerConsumerWarpSpecialized` | `-` | gemv | exact-int-accum:reassociation-excluded, gross, nan-pattern-changed | gen_gemv_ef525af0d65e95, gen_gemv_3cdc68e3131074, gen_gemv_6258f7a07b4606 | **KEPT** |
| 6 | HIT-0033 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | rr | - | gen_rr_71e2a24d872c43, gen_rr_34bc715948ae98, gen_rr_3c2d688fb1e763 | rejected B1 |
| 6 | HIT-0038 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | layernorm | - | gen_layernorm_7b6dfe71c776e8, gen_layernorm_e525e601897415, gen_layernorm_01b71142da7002 | rejected B1 |
| 6 | HIT-0062 | `-` | `tl.disable_wgmma` | cap | gross | cap_59dae696e2a0250f, cap_e021704c462f560f, cap_2bae46f31c2f27cd | rejected B1 + B2 |
| 6 | HIT-0064 | `Simplify#5` | `tl.enable_fast_math` | cumsum | - | gen_cumsum_d17b14d8fc2c6f, gen_cumsum_649260d0c6386b, gen_cumsum_5ca3a00c3a5a19 | rejected B1 |
| 6 | HIT-0070 | `Simplify#5` | `tl.enable_fast_math` | softmax | - | gen_softmax_677401fd500674, gen_softmax_53daeeadddf0b4, gen_softmax_56bfeeec24c6a4 | rejected B1 |
| 5 | HIT-0031 | `ConfigIndexBitwidth` | `tl.enable_fast_math` | layernorm | - | gen_layernorm_11b4887d5f5c52, gen_layernorm_4ceef0a14d5fae, gen_layernorm_a36b572a7e9c9e | rejected B1 |
| 5 | HIT-0046 | `Simplify#5` | `tl.enable_fast_math` | layernorm | - | gen_layernorm_3013e11c8c35df, gen_layernorm_577ed6af052ced, gen_layernorm_9edf67e48bdfcc | rejected B1 |
| 5 | HIT-0049 | `RenormalizeSplitPattern` | `tl.enable_fast_math` | fa | - | gen_fa_b6d78c9040bcba, gen_fa_4029a913ba9d03, gen_fa_beef985652e21d | rejected B1 |
| 5 | HIT-0059 | `-` | `tl.enable_fast_math, tl.storage_rewrite_detect_inplace` | fa | - | gen_fa_20174efacd74c3, gen_fa_0336a4475d6010, gen_fa_3095c9af3ea021 | rejected B1 |
| 5 | HIT-0088 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | cap | - | cap_7584be372a01683f, cap_91733369d00de38f, cap_91b5b330efbdb4d9 | rejected B1 |
| 5 | HIT-0089 | `Simplify#5` | `tl.enable_fast_math` | cap | - | cap_7584be372a01683f, cap_a4a0f5aedb06c42b, cap_ac133e314ed85c35 | rejected B1 |
| 5 | HIT-0094 | `ConfigIndexBitwidth` | `tl.enable_fast_math` | fa | - | gen_fa_d0350ce89194ec, gen_fa_fba2636211bb9d, gen_fa_d03404393f99d8 | rejected B1 |
| 4 | HIT-0030 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | cumsum | - | gen_cumsum_7fdcde72fc54e8, gen_cumsum_b71dc6f8abf42d, gen_cumsum_d7c8c369e89621 | rejected B1 |
| 4 | HIT-0037 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg_predicated` | layernorm | - | gen_layernorm_7b6dfe71c776e8, gen_layernorm_be8bcea7326a8b, gen_layernorm_4ceef0a14d5fae | rejected B1 |
| 4 | HIT-0040 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg_predicated` | softmax | - | gen_softmax_a59315a01d5949, gen_softmax_5b041c6da057b7, gen_softmax_6c2b3e2742e839 | rejected B1 |
| 4 | HIT-0042 | `-` | `tl.enable_fast_math` | fa | exact-int-accum:reassociation-excluded, nan-pattern-changed | gen_fa_ca4121f27fc1c5 | rejected B1 |
| 4 | HIT-0053 | `Simplify#5` | `tl.enable_fast_math` | fa | - | gen_fa_4691ae14e171e5, gen_fa_29172202383083, gen_fa_f5c0717b0f83ef | rejected B1 |
| 4 | HIT-0055 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | softmax | - | gen_softmax_32424a60f6baf1, gen_softmax_2e9b26003f6630, gen_softmax_565cb4e3456440 | rejected B1 |
| 4 | HIT-0058 | `-` | `tl.disable_tma_lower, tl.enable_fast_math` | fa | - | gen_fa_20174efacd74c3, gen_fa_fba2636211bb9d, gen_fa_2bc14cb07fb6e1 | rejected B1 |
| 4 | HIT-0061 | `-` | `tl.enable_fast_math` | rr | - | gen_rr_97f192f92a98e6, gen_rr_abad1d0e2d1135 | rejected B1 |
| 4 | HIT-0104 | `-` | `tl.disable_warp_specialized` | fa | gross | gen_fa_6a2b83e163628f, gen_fa_bf4da4c1c8fb08 | rejected B3 + B4 -> r2 HIT-0103 |
| 4 | HIT-0123 | `-` | `tl.enable_aggressive_shared_memory_merge` | gemv | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | gen_gemv_543b85074859a0, gen_gemv_83efff63db70b8, gen_gemv_a25b9b8fd59ad2 | rejected B4 -> r1 HIT-0040 |
| 4 | HIT-0146 | `-` | `tirx.disable_vectorize` | cap | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | cap_adc744f0e1cb07bf | rejected B4 -> r2 HIT-0145 and tilelang/HIT-0004 |
| 3 | HIT-0015 | `FuseMBarrierArriveExpectTx` | `tl.enable_fast_math` | rr | - | gen_rr_785bb79233c6e0, gen_rr_3ad7e81590f1a7, gen_rr_1fa5c3a2003244 | rejected B1 |
| 3 | HIT-0017 | `NarrowDataType` | `tl.enable_fast_math` | cap | - | cap_376f3c86ab2c8d2f, cap_9482c1a382819f21, cap_baafd4cdfe08a1f8 | rejected B1 |
| 3 | HIT-0022 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | layernorm | - | gen_layernorm_cb8fb93cd6be23, gen_layernorm_1e8454ae8ff986, gen_layernorm_d70004f584ad76 | rejected B1 |
| 3 | HIT-0036 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | cumsum | - | gen_cumsum_6a08015c14f6ca, gen_cumsum_b71dc6f8abf42d, gen_cumsum_f0bba0841613fc | rejected B1 |
| 3 | HIT-0056 | `ConfigIndexBitwidth` | `tl.enable_fast_math` | cumsum | - | gen_cumsum_079e004477eb76, gen_cumsum_38a89e6543a92e, gen_cumsum_5ca3a00c3a5a19 | rejected B1 |
| 3 | HIT-0079 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | softmax | - | gen_softmax_9085c15f278943, gen_softmax_53daeeadddf0b4, gen_softmax_9e14a42a0d0290 | rejected B1 |
| 3 | HIT-0083 | `ConfigIndexBitwidth` | `tl.enable_fast_math` | rr | - | gen_rr_3ad7e81590f1a7, gen_rr_2252a70a519a20, gen_rr_334177a36a0483 | rejected B1 |
| 3 | HIT-0086 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | softmax | - | gen_softmax_570206d2cdd3ba, gen_softmax_f9b2993a8bb70d, gen_softmax_473553beecdbbc | rejected B1 |
| 3 | HIT-0110 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | cumsum | - | gen_cumsum_cb6560826868e4, gen_cumsum_5719479a0e062d, gen_cumsum_96b73f8a3b09e5 | rejected B1 |
| 3 | HIT-0111 | `-` | `tl.disable_shared_memory_reuse, tl.enable_fast_math` | fa | - | gen_fa_3095c9af3ea021, gen_fa_e9b9e46e21035f, gen_fa_058dbc28f8ffc6 | rejected B1 |
| 3 | HIT-0129 | `-` | `tirx.use_async_copy, tl.enable_fast_math` | rr | - | gen_rr_4f9cd457d162d8, gen_rr_8694d9f8cef003 | rejected B1 |
| 3 | HIT-0142 | `ConfigIndexBitwidth` | `-` | cast | gross | gen_cast_a254db48fd63cd, gen_cast_92f3b2fbfc24de, gen_cast_8262baebe32eb7 | rejected B4 -> r1 HIT-0001 |
| 2 | HIT-0026 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | rr | - | gen_rr_de4809e993c368, gen_rr_9a9175c24fe7fe | rejected B1 |
| 2 | HIT-0041 | `Simplify#5` | `tl.enable_fast_math` | strided | - | gen_strided_be63cf4d57cda3, gen_strided_79721b38c1a60a | rejected B1 |
| 2 | HIT-0048 | `-` | `tl.disable_shared_memory_reuse, tl.enable_fast_math` | softmax | - | gen_softmax_a01a02597e2d4c, gen_softmax_8cb57a23d0deec | rejected B1 |
| 2 | HIT-0057 | `StorageRewrite` | `tl.enable_fast_math` | rr | - | gen_rr_51d7845b8b73f3, gen_rr_5112068c81570a | rejected B1 |
| 2 | HIT-0069 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | bcast | nan-pattern-changed | gen_bcast_58c1d74da4d255, gen_bcast_7465ca166fe87b | rejected B1 |
| 2 | HIT-0075 | `Simplify#5` | `tl.enable_fast_math` | elem | - | gen_elem_72b45f5d7013f3 | rejected B1 |
| 2 | HIT-0080 | `-` | `tl.disable_shuffle_elect, tl.enable_fast_math` | cap | - | cap_6fc5657f5c561dda, cap_766551a23059b482 | rejected B1 |
| 2 | HIT-0082 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | fa | gross, nan-pattern-changed | gen_fa_edcc4d42966c01, gen_fa_c1cd3104d66b5e | rejected B1 |
| 2 | HIT-0087 | `NarrowDataType` | `tl.enable_fast_math` | fa | - | gen_fa_0336a4475d6010 | rejected B1 |
| 2 | HIT-0099 | `DecoupleTypeCast` | `tl.enable_fast_math` | cap | gross | cap_82592c2e5e1f5133, cap_ef25e27811080278 | rejected B1 |
| 2 | HIT-0100 | `-` | `tl.enable_fast_math` | cap | gross | cap_82592c2e5e1f5133, cap_9e5bbec266901edb | rejected B1 |
| 2 | HIT-0102 | `NarrowDataType` | `tl.enable_fast_math` | rr | - | gen_rr_64c5180275c19c, gen_rr_abb5eb55725971 | rejected B1 |
| 2 | HIT-0106 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | cap | - | cap_9482c1a382819f21, cap_b5fb88b0e5131217 | rejected B1 |
| 2 | HIT-0109 | `FuseMBarrierArriveExpectTx` | `tl.enable_fast_math` | fa | - | gen_fa_17a4e67fe93cf0, gen_fa_bf4da4c1c8fb08 | rejected B1 |
| 2 | HIT-0112 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | fa | - | gen_fa_beef985652e21d | rejected B1 |
| 2 | HIT-0114 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | fa | - | gen_fa_2aeb3e7ea366b1, gen_fa_ec684f1e36abce | rejected B1 |
| 2 | HIT-0117 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | cap | - | cap_be9ff46593d43e0f, cap_ed92b4ce1df8dc3b | rejected B1 |
| 2 | HIT-0118 | `ConfigIndexBitwidth` | `tl.enable_fast_math` | softmax | - | gen_softmax_9e14a42a0d0290, gen_softmax_051b80040d7df6 | rejected B1 |
| 2 | HIT-0119 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | rr | - | gen_rr_93b6b33e1ae550, gen_rr_55b822bd8d0e80 | rejected B1 |
| 2 | HIT-0133 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | fa | - | gen_fa_6960fdccbe7fd5, gen_fa_5fc239c2464bd1 | rejected B1 |
| 2 | HIT-0134 | `-` | `tl.disable_shared_memory_reuse, tl.enable_fast_math` | layernorm | - | gen_layernorm_0890ec48ea56bc, gen_layernorm_d70004f584ad76 | rejected B1 |
| 2 | HIT-0135 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | layernorm | - | gen_layernorm_3d4c812640ca93, gen_layernorm_d70004f584ad76 | rejected B1 |
| 1 | HIT-0006 | `-` | `tl.disable_wgmma` | cap | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | cap_0b881ee73fbdf83c | rejected B1 + B2 |
| 1 | HIT-0009 | `-` | `tl.disable_warp_specialized` | gemv | exact-int-accum:reassociation-excluded, gross, nan-pattern-changed | gen_gemv_ef525af0d65e95 | rejected B4 -> r2 HIT-0008 |
| 1 | HIT-0016 | `RenormalizeSplitPattern` | `tl.enable_fast_math` | fa | exact-int-accum:reassociation-excluded | gen_fa_915604d75722ed | rejected B1 |
| 1 | HIT-0019 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | softmax | exact-int-accum:reassociation-excluded | gen_softmax_de6674ba60b530 | rejected B1 |
| 1 | HIT-0020 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | softmax | exact-int-accum:reassociation-excluded | gen_softmax_de6674ba60b530 | rejected B1 |
| 1 | HIT-0021 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | layernorm | - | gen_layernorm_cb8fb93cd6be23 | rejected B1 |
| 1 | HIT-0023 | `Simplify#5` | `tl.enable_fast_math` | bcast | - | gen_bcast_b31bf86506944d | rejected B1 |
| 1 | HIT-0024 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | fa | exact-int-accum:reassociation-excluded | gen_fa_d13e1488703474 | rejected B1 |
| 1 | HIT-0025 | `RenormalizeSplitPattern` | `tl.enable_fast_math` | cumsum | - | gen_cumsum_9e10f61a8de9e6 | rejected B1 |
| 1 | HIT-0027 | `Simplify#5` | `tl.enable_fast_math` | elem | nan-pattern-changed | gen_elem_be66cc204e6a82 | rejected B1 |
| 1 | HIT-0028 | `-` | `tl.config_index_bitwidth` | cap | exact-int-accum:reassociation-excluded, gross | cap_401ab8ec5f03638a | rejected B4 -> r1 HIT-0001 |
| 1 | HIT-0029 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | softmax | exact-int-accum:reassociation-excluded | gen_softmax_9915d56f9d1910 | rejected B1 |
| 1 | HIT-0039 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg_predicated` | bcast | nan-pattern-changed | gen_bcast_d6105c9f50542b | rejected B1 |
| 1 | HIT-0043 | `FuseMBarrierArriveExpectTx` | `tl.if_stmt_binding_inline_replayable_binds` | fa | gross, nan-pattern-changed, pass-class:bit-preserving | gen_fa_ca4121f27fc1c5 | **KEPT** |
| 1 | HIT-0044 | `Simplify#1` | `tl.loop_unswitching_allow_non_trivial_else` | fa | gross, nan-pattern-changed, pass-class:bit-preserving | gen_fa_ca4121f27fc1c5 | rejected B4 -> r2 HIT-0043 |
| 1 | HIT-0045 | `InjectSoftwarePipeline, Simplify#5` | `tirx.debug_keep_trivial_loop` | fa | gross, nan-pattern-changed, pass-class:bit-preserving | gen_fa_ca4121f27fc1c5 | rejected B4 -> r2 HIT-0043 |
| 1 | HIT-0047 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | strided | - | gen_strided_b2ee2123fb92b2 | rejected B1 |
| 1 | HIT-0050 | `AnnotateWarpGroupRegAlloc` | `tl.enable_fast_math` | rr | - | gen_rr_24e3e2ea1f084e | rejected B1 |
| 1 | HIT-0051 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | elem | nan-pattern-changed | gen_elem_04d52c08139e87 | rejected B1 |
| 1 | HIT-0052 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | bcast | - | gen_bcast_1b16e6c0362bf4 | rejected B1 |
| 1 | HIT-0054 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | softmax | - | gen_softmax_32424a60f6baf1 | rejected B1 |
| 1 | HIT-0060 | `AnnotateWarpGroupRegAlloc` | `tl.enable_fast_math` | fa | - | gen_fa_20174efacd74c3 | rejected B1 |
| 1 | HIT-0063 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | bcast | nan-pattern-changed | gen_bcast_7fd6c9f3f04dad | rejected B1 |
| 1 | HIT-0065 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | cap | exact-int-accum:reassociation-excluded | cap_5fd9ad26de40625d | rejected B1 |
| 1 | HIT-0066 | `Simplify#5` | `tl.enable_fast_math` | cap | exact-int-accum:reassociation-excluded | cap_5fd9ad26de40625d | rejected B1 |
| 1 | HIT-0067 | `-` | `tl.disable_shared_memory_reuse, tl.enable_fast_math` | cap | gross | cap_5f7b24891a1fad81 | rejected B1 |
| 1 | HIT-0068 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | cap | gross | cap_5f7b24891a1fad81 | rejected B1 |
| 1 | HIT-0071 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | cap | nan-pattern-changed | cap_6c3c978802144450 | rejected B1 |
| 1 | HIT-0073 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | fa | nan-pattern-changed | gen_fa_50a732cae882e1 | rejected B1 |
| 1 | HIT-0074 | `FuseMBarrierArriveExpectTx` | `tl.enable_fast_math` | fa | nan-pattern-changed | gen_fa_50a732cae882e1 | rejected B1 |
| 1 | HIT-0076 | `-` | `tl.enable_fast_math, tl.storage_rewrite_detect_inplace` | cap | gross | cap_6cb27073322886de | rejected B1 |
| 1 | HIT-0077 | `RenormalizeSplitPattern` | `tl.enable_fast_math` | cap | gross | cap_6cb27073322886de | rejected B1 |
| 1 | HIT-0078 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg_predicated` | rr | - | gen_rr_aae9fdfa5e7533 | rejected B1 |
| 1 | HIT-0081 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg_predicated` | bcast | - | gen_bcast_596ccb38c2aa57 | rejected B1 |
| 1 | HIT-0084 | `AnnotateReadOnlyParams, StorageRewrite` | `tirx.disable_vectorize, tl.enable_aggressive_shared_memory_merge` | rr | exact-int-accum:reassociation-excluded, gross | gen_rr_3ad7e81590f1a7 | rejected B4 -> r1 HIT-0040 |
| 1 | HIT-0085 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg_predicated` | strided | - | gen_strided_90ec4c52bde98c | rejected B1 |
| 1 | HIT-0090 | `StorageRewrite` | `tl.enable_fast_math` | cap | - | cap_766551a23059b482 | rejected B1 |
| 1 | HIT-0092 | `-` | `tl.disable_warp_specialized` | gemv | exact-int-accum:reassociation-excluded, nan-pattern-changed | gen_gemv_e030b819eb9800 | rejected B4 -> r2 HIT-0008 |
| 1 | HIT-0093 | `-` | `tl.disable_tma_lower, tl.enable_fast_math, tl.if_stmt_binding_inline_replayable_binds` | strided | - | gen_strided_64f671b45a9cc6 | rejected B1 |
| 1 | HIT-0095 | `-` | `tl.disable_warp_specialized, tl.enable_fast_math` | rr | - | gen_rr_cad1893b04d59c | rejected B1 |
| 1 | HIT-0096 | `StorageRewrite` | `tl.disable_warp_specialized` | fa | nan-pattern-changed | gen_fa_f2030a506461ab | rejected B4 -> r1 HIT-0043 |
| 1 | HIT-0097 | `StorageRewrite` | `tl.enable_fast_math` | fa | nan-pattern-changed | gen_fa_f2030a506461ab | rejected B1 |
| 1 | HIT-0098 | `-` | `tirx.noalias, tl.enable_fast_math` | cap | gross | cap_82592c2e5e1f5133 | rejected B1 |
| 1 | HIT-0101 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | strided | - | gen_strided_99564509da8204 | rejected B1 |
| 1 | HIT-0105 | `RenormalizeSplitPattern` | `tl.enable_fast_math` | softmax | - | gen_softmax_37e6ae8d088a24 | rejected B1 |
| 1 | HIT-0107 | `-` | `tl.disable_shared_memory_reuse, tl.enable_fast_math` | cap | - | cap_9946608f09497fe5 | rejected B1 |
| 1 | HIT-0108 | `ProducerConsumerWarpSpecialized` | `tl.enable_fast_math` | fa | - | gen_fa_17a4e67fe93cf0 | rejected B1 + B3 |
| 1 | HIT-0113 | `HoistIfThenElse, RemoveNoOp` | `tl.enable_fast_math` | strided | - | gen_strided_dcb69d809f9c31 | rejected B1 |
| 1 | HIT-0115 | `StorageRewrite` | `tl.enable_fast_math` | fa | - | gen_fa_2aeb3e7ea366b1 | rejected B1 + B4 -> r2 HIT-0114 |
| 1 | HIT-0116 | `RenormalizeSplitPattern` | `tl.enable_fast_math` | strided | - | gen_strided_ded59c7bd52ea5 | rejected B1 |
| 1 | HIT-0120 | `-` | `tl.disable_wgmma, tl.enable_fast_math` | fa | gross, nan-pattern-changed | gen_fa_6a5c1a5a6a70a3 | rejected B1 + B4 -> r1 HIT-0009 |
| 1 | HIT-0121 | `NarrowDataType` | `tl.enable_fast_math` | fa | gross, nan-pattern-changed | gen_fa_6a5c1a5a6a70a3 | rejected B1 + B4 -> r2 HIT-0120 |
| 1 | HIT-0122 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | fa | - | gen_fa_ddb0851b5e4014 | **PARKED** (owner rule: NaN payload / sign of zero) |
| 1 | HIT-0124 | `NarrowDataType` | `tl.enable_fast_math` | strided | - | gen_strided_f04060802f9804 | rejected B1 |
| 1 | HIT-0125 | `HoistIfThenElse, tirxSimplify#2` | `tl.enable_fast_math` | rr | - | gen_rr_abb5eb55725971 | rejected B1 |
| 1 | HIT-0126 | `-` | `tl.enable_fast_math, tl.storage_rewrite_detect_inplace` | fa | gross, nan-pattern-changed | gen_fa_8503044aadb39f | rejected B1 + B4 -> r1 HIT-0009 |
| 1 | HIT-0127 | `-` | `tl.enable_fast_math, tl.storage_rewrite_detect_inplace` | cap | - | cap_d683426e80ad44b5 | rejected B1 |
| 1 | HIT-0128 | `ProducerConsumerWarpSpecialized` | `tl.enable_fast_math` | rr | - | gen_rr_55b822bd8d0e80 | rejected B1 |
| 1 | HIT-0130 | `ConfigIndexBitwidth` | `tl.enable_fast_math` | bcast | nan-pattern-changed | gen_bcast_69304f922ede97 | rejected B1 + B4 -> r1 HIT-0009 |
| 1 | HIT-0131 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg_predicated` | elem | nan-pattern-changed | gen_elem_16be5b2190db7a | rejected B1 + B4 -> r1 HIT-0009 |
| 1 | HIT-0132 | `HoistIfThenElse, tirxSimplify#2` | `tl.enable_fast_math` | fa | - | gen_fa_ddda62abcc2b3f | rejected B1 |
| 1 | HIT-0136 | `InjectAssumes` | `tl.enable_fast_math` | cap | gross | cap_f0afd623671fbda9 | rejected B1 |
| 1 | HIT-0137 | `ProducerConsumerWarpSpecialized` | `tl.enable_fast_math` | gemv | nan-pattern-changed | gen_gemv_d4bc1ef49dd602 | rejected B1 + B4 -> r1 HIT-0009 |
| 1 | HIT-0138 | `UnrollLoop` | `tl.enable_fast_math` | strided | - | gen_strided_79721b38c1a60a | rejected B1 |
| 1 | HIT-0139 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | strided | - | gen_strided_79721b38c1a60a | rejected B1 + B4 -> r2 HIT-0138 |
| 1 | HIT-0140 | `-` | `tl.disable_tma_lower, tl.enable_fast_math` | cap | - | cap_fc071272090830ec | rejected B1 |
| 1 | HIT-0141 | `ProducerConsumerWarpSpecialized, StorageRewrite` | `-` | cap | nan-pattern-changed | cap_1b923435ca892b54 | rejected B4 -> r1 HIT-0043 |
| 1 | HIT-0143 | `ProducerConsumerWarpSpecialized` | `tirx.use_async_copy` | fa | gross | gen_fa_b4fc5848cc6acc | rejected B3 |
| 1 | HIT-0145 | `-` | `tirx.disable_vectorize` | cap | exact-int-accum:reassociation-excluded, gross | cap_adc744f0e1cb07bf | rejected B4 -> r1 HIT-0004 |
| 1 | HIT-0147 | `-` | `tl.enable_aggressive_shared_memory_merge` | gemv | exact-int-accum:reassociation-excluded, pass-class:bit-preserving | gen_gemv_1b40e158965065 | rejected B4 -> r1 HIT-0040 |
| 1 | HIT-0149 | `-` | `tl.config_index_bitwidth` | gemv | - | gen_gemv_4da8042d09cfec | rejected B3 |
| 1 | HIT-0150 | `-` | `tl.config_index_bitwidth` | gemv | pass-class:bit-preserving | gen_gemv_4da8042d09cfec | rejected B3 |
| 1 | HIT-0151 | `-` | `tl.config_index_bitwidth` | cap | gross | cap_401ab8ec5f03638a | rejected B4 -> r1 HIT-0001 |
| 1 | HIT-0152 | `-` | `tl.config_index_bitwidth` | cap | gross, pass-class:bit-preserving | cap_401ab8ec5f03638a | rejected B4 -> r1 HIT-0001 |
