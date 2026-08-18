# TileLang differential fuzzing -- findings

Scope: bit-equivalence only.  A row is: two compilations of the same
PrimFunc, differing only in the pass set, produced different bits on
the same input.  `SMT verdict` is filled in by a reviewing agent.

| id | kernel | culprit | pass_configs | arch | labels | ndiff/total | max ulp | SMT verdict |
|---|---|---|---|---|---|---|---|---|
| HIT-0001 | `cap_081a26caff856329` | `None` | `{"tl.config_index_bitwidth": 64}` | generic | pass-class:bit-preserving, input-mode:captured, exact-int-accum:reassociation-excluded, gross | 3/7 | 2097217536 | TODO |
| HIT-0002 | `cap_0b881ee73fbdf83c` | `None` | `{"tl.disable_wgmma": true}` | hopper-only | pass-class:bit-preserving, kernel-has-gemm, kernel-has-reduce, input-mode:captured, exact-int-accum:reassociation-excluded, gross | 22332/65536 | 16384 | TODO |
| HIT-0003 | `cap_2bae46f31c2f27cd` | `None` | `{"tl.disable_wgmma": true}` | hopper-only | pass-class:bit-preserving, kernel-has-gemm, input-mode:captured, gross | 1047889/1048576 | 3191955456 | TODO |
| HIT-0004 | `cap_adc744f0e1cb07bf` | `None` | `{"tirx.disable_vectorize": true}` | generic | pass-class:bit-preserving, input-mode:captured, exact-int-accum:reassociation-excluded, gross | 1536/1536 | 9222069216401705948 | TODO |
| HIT-0005 | `cap_82592c2e5e1f5133` | `None` | `{"tl.enable_fast_math": true}` | generic | pass-class:numerics-changing, kernel-has-gemm, kernel-has-reduce, input-mode:captured, exact-int-accum:reassociation-excluded, gross | 5627/2107392 | 9 | TODO |
| HIT-0006 | `cap_c84d73cfbb167409` | `InjectSoftwarePipeline` | `{}` | hopper-only | pass-class:bit-preserving, layout-reorders-reduction, kernel-has-gemm, kernel-has-reduce, input-mode:captured, exact-int-accum:reassociation-excluded, few-ulp | 16/524288 | 1 | TODO |
