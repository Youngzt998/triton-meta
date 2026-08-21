# TileLang differential fuzzing -- every difference signature seen

One row per distinct (minimal pass set, pass_configs keys, strength class).
`n` counts every (kernel, input mode, variant) occurrence, not just the
reported one.  The full per-occurrence records are in
`/home/youngzt/fuzz-tilelang/run/results.jsonl`.

updated 2026-08-19 14:39:43

| n | report | minimal pass set | pass_configs | strength | example kernels |
|---|---|---|---|---|---|
| 60 | HIT-0029 | `ProducerConsumerWarpSpecialized` | `-` | exact-int-accum:reassociation-excluded | hw_fa_2x2x384x384x64_c1_s1, hw_fa_1x1x256x512x64_c1_s2, hw_fa_1x2x260x512x64_c0_s1, hw_fa_1x2x260x512x64_c0_s2 |
| 48 | HIT-0003 | `-` | `tl.disable_wgmma` | gross, pass-class:bit-preserving | cap_2bae46f31c2f27cd, cap_59dae696e2a0250f, cap_8499fd7090826a59, cap_df90fc49f1a27655 |
| 37 | HIT-0030 | `-` | `tl.disable_warp_specialized` | exact-int-accum:reassociation-excluded | hw_fa_2x2x384x384x64_c1_s1, hw_fa_1x2x260x512x64_c1_s2, hw_fa_1x2x512x512x64_c0_s1, hw_fa_1x2x512x512x64_c0_s2 |
| 23 | HIT-0001 | `-` | `tl.config_index_bitwidth` | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | cap_081a26caff856329, cap_401ab8ec5f03638a, cap_bde15a1202582b07 |
| 23 | HIT-0048 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | - | cap_7584be372a01683f, cap_acdcb72613cb6e94, cap_c3415df9d2952c5f, cap_e4d27c3e2b606450 |
| 20 | HIT-0024 | `-` | `tl.config_index_bitwidth` | gross, pass-class:bit-preserving | hw_cast_1000x300, hw_cast_2048x64, hw_cast_1024x512 |
| 17 | HIT-0008 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | - | cap_6fc5657f5c561dda, cap_d8ade0dc89bca68a, cap_fefcfb3b931acba4, cap_3eb16a96b358771f |
| 17 | HIT-0012 | `Simplify#5` | `tl.enable_fast_math` | - | cap_766551a23059b482, hw_cumsum_600x128, cap_7584be372a01683f, cap_d72cde3fc74508b3 |
| 11 | HIT-0016 | `NarrowDataType` | `tl.enable_fast_math` | - | cap_9482c1a382819f21, cap_bfe214c058531d24, cap_d72cde3fc74508b3, cap_d8ade0dc89bca68a |
| 11 | HIT-0019 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | - | cap_e98f312c604500e1, cap_ee6646f06f7897c7, cap_01e7ee8fde8f2afc, cap_b7fdabd65a70ddb2 |
| 11 | HIT-0031 | `-` | `tl.disable_warp_specialized` | exact-int-accum:reassociation-excluded, gross | hw_fa_1x1x256x512x64_c0_s1, hw_fa_1x1x256x512x64_c1_s1, hw_fa_2x2x384x384x64_c0_s2, hw_fa_1x2x260x512x64_c0_s1 |
| 11 | HIT-0032 | `ProducerConsumerWarpSpecialized` | `-` | exact-int-accum:reassociation-excluded, gross | hw_fa_1x1x256x512x64_c0_s1, hw_fa_1x1x256x512x64_c1_s1, hw_fa_2x2x384x384x64_c0_s2, hw_fa_1x2x260x512x64_c0_s1 |
| 10 | HIT-0052 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | - | cap_de3a1582dd064f93, hw_cumsum_600x128, cap_3eb16a96b358771f, cap_8db2674b90a3c838 |
| 9 | HIT-0040 | `-` | `tl.enable_aggressive_shared_memory_merge` | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | hw_rrsum_256x4096 |
| 8 | HIT-0015 | `-` | `tl.disable_wgmma` | gross | cap_8499fd7090826a59, cap_2bae46f31c2f27cd, cap_e021704c462f560f, cap_59dae696e2a0250f |
| 8 | HIT-0017 | `Simplify#5` | `tl.enable_fast_math` | nan-pattern-changed | cap_bd482bcabcadd5cf, cap_a9e4f77a1e5c77a2, cap_02f8a7d289bdf0c2, hw_elem_2048x1024 |
| 8 | HIT-0034 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | exact-int-accum:reassociation-excluded | hw_fa_1x2x512x512x64_c0_s2, hw_fa_1x2x260x512x64_c0_s2, hw_fa_1x2x260x512x64_c0_s1, cap_0f24f6111164937a |
| 8 | HIT-0041 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | nan-pattern-changed | cap_0f433a98ee55b9d1, hw_elem_128x64, hw_elem_2000x1000, hw_elem_666x517 |
| 7 | HIT-0002 | `-` | `tl.disable_wgmma` | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | cap_0b881ee73fbdf83c |
| 7 | HIT-0063 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | nan-pattern-changed | cap_02f8a7d289bdf0c2, cap_92ce0e7a3875f908, hw_elem_666x517, cap_01e7ee8fde8f2afc |
| 6 | HIT-0018 | `Simplify#5` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded | cap_adc744f0e1cb07bf, cap_5fd9ad26de40625d, cap_707dc2d174fbeaf0, hw_softmax_512x256 |
| 5 | HIT-0014 | `-` | `tl.disable_tma_lower, tl.enable_fast_math` | - | cap_8f4233db429a0006, cap_d72cde3fc74508b3, cap_6a6d11ccb7bbef31, cap_cc965a2a6e5783a2 |
| 4 | HIT-0005 | `-` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross | cap_82592c2e5e1f5133, cap_9e5bbec266901edb |
| 4 | HIT-0011 | `-` | `tl.enable_lower_ldgstg` | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | cap_7fd65a04c9705768 |
| 4 | HIT-0020 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded | cap_ed92b4ce1df8dc3b, hw_softmax_1024x128, cap_707dc2d174fbeaf0, hw_softmax_600x128 |
| 4 | HIT-0023 | `-` | `tl.config_index_bitwidth` | gross | hw_cast_1024x512, hw_cast_2048x64, hw_cast_1000x300 |
| 4 | HIT-0039 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | exact-int-accum:reassociation-excluded | hw_softmax_512x256, cap_f8321c21940adfea, hw_softmax_600x128, cap_ed92b4ce1df8dc3b |
| 4 | HIT-0046 | `StorageRewrite` | `tl.enable_fast_math` | - | cap_766551a23059b482, cap_d72cde3fc74508b3, cap_3ccd404616e5b446, cap_18431ea074005218 |
| 4 | HIT-0059 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | exact-int-accum:reassociation-excluded | hw_fa_2x2x384x384x64_c0_s1, hw_fa_1x2x260x512x64_c1_s2, hw_fa_1x1x256x512x64_c0_s1, cap_5fd9ad26de40625d |
| 3 | HIT-0007 | `PipelinePlanning` | `-` | exact-int-accum:reassociation-excluded, pass-class:bit-preserving | cap_c84d73cfbb167409 |
| 3 | HIT-0009 | `-` | `tirx.disable_vectorize, tl.enable_fast_math` | nan-pattern-changed | cap_6c3c978802144450, cap_02f8a7d289bdf0c2 |
| 3 | HIT-0021 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | nan-pattern-changed | cap_f87fe4bd25031dd0, cap_6c3c978802144450 |
| 3 | HIT-0033 | `StorageRewrite` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded | hw_fa_1x2x260x512x64_c0_s2, hw_fa_1x2x512x512x64_c0_s1, hw_fa_2x2x384x384x64_c0_s1 |
| 3 | HIT-0072 | `-` | `tl.enable_aggressive_shared_memory_merge` | gross, pass-class:bit-preserving | hw_rrmax_256x4096 |
| 3 | HIT-0073 | `-` | `tl.config_index_bitwidth` | exact-int-accum:reassociation-excluded, gross | cap_bde15a1202582b07, cap_081a26caff856329, cap_401ab8ec5f03638a |
| 2 | HIT-0006 | `InjectSoftwarePipeline` | `-` | exact-int-accum:reassociation-excluded, pass-class:bit-preserving | cap_c84d73cfbb167409 |
| 2 | HIT-0022 | `AnnotateReadOnlyParams` | `tl.enable_fast_math` | gross | cap_f0afd623671fbda9, cap_e6ea4bebff5e95c5 |
| 2 | HIT-0043 | `StorageRewrite` | `tl.disable_warp_specialized` | exact-int-accum:reassociation-excluded, nan-pattern-changed | cap_5be24248d2229c36, cap_1b923435ca892b54 |
| 2 | HIT-0044 | `-` | `tl.disable_shuffle_elect, tl.enable_fast_math` | - | cap_6fc5657f5c561dda, cap_18431ea074005218 |
| 2 | HIT-0045 | `-` | `tl.enable_fast_math, tl.storage_rewrite_detect_inplace` | gross | cap_6cb27073322886de |
| 2 | HIT-0047 | `-` | `tl.enable_lower_ldgstg` | exact-int-accum:reassociation-excluded, gross | cap_7fd65a04c9705768 |
| 2 | HIT-0050 | `-` | `tl.enable_fast_math, tl.enable_lower_ldgstg` | exact-int-accum:reassociation-excluded, gross | cap_82592c2e5e1f5133, cap_aae10cfcad60a975 |
| 2 | HIT-0053 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | gross | cap_e6ea4bebff5e95c5, cap_7c3b3c8d9410c766 |
| 2 | HIT-0054 | `PipelinePlanning` | `tl.enable_fast_math` | gross | cap_f0afd623671fbda9 |
| 2 | HIT-0055 | `Simplify#5` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross | cap_ef25e27811080278, cap_9e5bbec266901edb |
| 2 | HIT-0058 | `-` | `tl.disable_tma_lower, tl.enable_fast_math` | exact-int-accum:reassociation-excluded | hw_fa_2x2x384x384x64_c0_s1, hw_fa_1x2x512x512x64_c1_s2 |
| 2 | HIT-0060 | `RenormalizeSplitPattern` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded | hw_fa_1x2x260x512x64_c1_s2, hw_softmax_600x128 |
| 2 | HIT-0061 | `FuseMBarrierArriveExpectTx` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded | hw_fa_2x2x384x384x64_c1_s1, hw_fa_1x1x256x512x64_c1_s1 |
| 2 | HIT-0062 | `-` | `tl.enable_aggressive_shared_memory_merge` | pass-class:bit-preserving | hw_rrmax_256x4096 |
| 2 | HIT-0069 | `-` | `tl.enable_aggressive_shared_memory_merge, tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross | cap_aae10cfcad60a975, cap_0b881ee73fbdf83c |
| 2 | HIT-0071 | `AnnotateWarpGroupRegAlloc` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded | hw_fa_1x2x260x512x64_c0_s1, hw_fa_1x1x256x512x64_c1_s1 |
| 2 | HIT-0075 | `ProducerConsumerWarpSpecialized, StorageRewrite` | `-` | exact-int-accum:reassociation-excluded, nan-pattern-changed | cap_5be24248d2229c36 |
| 1 | HIT-0004 | `-` | `tirx.disable_vectorize` | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | cap_adc744f0e1cb07bf |
| 1 | HIT-0010 | `StorageRewrite` | `tl.enable_fast_math` | gross | cap_6cb27073322886de |
| 1 | HIT-0013 | `FuseMBarrierArriveExpectTx` | `tl.enable_fast_math` | - | cap_766551a23059b482 |
| 1 | HIT-0025 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross | cap_ef25e27811080278 |
| 1 | HIT-0026 | `-` | `tl.disable_wgmma` | exact-int-accum:reassociation-excluded, pass-class:bit-preserving | cap_c84d73cfbb167409 |
| 1 | HIT-0027 | `-` | `tl.disable_wgmma` | exact-int-accum:reassociation-excluded | cap_c84d73cfbb167409 |
| 1 | HIT-0028 | `InjectSoftwarePipeline` | `-` | exact-int-accum:reassociation-excluded | cap_c84d73cfbb167409 |
| 1 | HIT-0035 | `-` | `tl.disable_warp_specialized` | exact-int-accum:reassociation-excluded, gross, nan-pattern-changed | hw_fa_1x2x260x512x64_c1_s1 |
| 1 | HIT-0036 | `Simplify#5` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross, nan-pattern-changed | hw_fa_1x2x260x512x64_c1_s1 |
| 1 | HIT-0037 | `StorageRewrite` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross, nan-pattern-changed | hw_fa_1x2x260x512x64_c1_s1 |
| 1 | HIT-0038 | `-` | `tl.disable_tma_lower, tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross, nan-pattern-changed | hw_fa_1x2x260x512x64_c1_s1 |
| 1 | HIT-0042 | `StorageRewrite` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross | cap_0b881ee73fbdf83c |
| 1 | HIT-0049 | `UnrollLoop` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross | cap_82592c2e5e1f5133 |
| 1 | HIT-0051 | `DecoupleTypeCast` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross | cap_82592c2e5e1f5133 |
| 1 | HIT-0056 | `NarrowDataType` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded | hw_fa_1x1x256x512x64_c1_s2 |
| 1 | HIT-0057 | `-` | `tl.config_index_bitwidth, tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross, nan-pattern-changed | hw_fa_1x2x512x512x64_c1_s2 |
| 1 | HIT-0064 | `-` | `tl.disable_wgmma` | exact-int-accum:reassociation-excluded, gross | cap_0b881ee73fbdf83c |
| 1 | HIT-0065 | `-` | `tirx.use_async_copy, tl.enable_fast_math` | gross | cap_f0afd623671fbda9 |
| 1 | HIT-0066 | `-` | `tl.disable_wgmma, tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross | cap_ef25e27811080278 |
| 1 | HIT-0067 | `-` | `tl.enable_fast_math, tl.storage_rewrite_detect_inplace` | - | hw_fa_1x1x256x512x64_c0_s2 |
| 1 | HIT-0068 | `-` | `tl.disable_shared_memory_reuse, tl.enable_fast_math` | exact-int-accum:reassociation-excluded | hw_softmax_512x256 |
| 1 | HIT-0070 | `NarrowDataType` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross | cap_aae10cfcad60a975 |
| 1 | HIT-0074 | `-` | `tl.enable_fast_math, tl.storage_rewrite_detect_inplace` | exact-int-accum:reassociation-excluded | hw_fa_2x2x384x384x64_c0_s2 |
| 1 | HIT-0076 | `DecoupleTypeCast` | `tl.enable_fast_math` | gross | cap_6cb27073322886de |
| 1 | HIT-0077 | `-` | `tl.enable_lower_ldgstg` | exact-int-accum:reassociation-excluded, gross, nan-pattern-changed, pass-class:bit-preserving | cap_7fd65a04c9705768 |
| 1 | HIT-0078 | `InjectSoftwarePipeline` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross | cap_5f7b24891a1fad81 |
| 1 | HIT-0079 | `-` | `tl.disable_shared_memory_reuse, tl.enable_fast_math` | - | cap_98a568e72c1ddd35 |
| 1 | HIT-0080 | `RenormalizeSplitPattern` | `tl.enable_fast_math` | - | cap_e98f312c604500e1 |
| 1 | HIT-0081 | `-` | `tl.disable_warp_specialized, tl.enable_fast_math` | exact-int-accum:reassociation-excluded | hw_fa_1x2x512x512x64_c1_s2 |
| 1 | HIT-0082 | `AnnotateWarpGroupRegAlloc` | `tl.enable_fast_math` | exact-int-accum:reassociation-excluded, gross, nan-pattern-changed | hw_fa_1x2x512x512x64_c1_s1 |
