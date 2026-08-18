# TileLang differential fuzzing -- findings

Scope: bit-equivalence only.  A row is: two compilations of the same
PrimFunc, differing only in the pass set, produced different bits on
the same input.  `SMT verdict` is filled in by a reviewing agent.

| id | kernel | culprit | pass_configs | arch | labels | ndiff/total | max ulp | SMT verdict |
|---|---|---|---|---|---|---|---|---|
| HIT-0001 | `hw_fa_1x1x128x256x64_c0_s2` | `ProducerConsumerWarpSpecialized` | `{}` | hopper-only | pass-class:numerics-changing, layout-reorders-reduction, kernel-has-gemm, kernel-has-reduce, input-mode:normal, exact-int-accum:reassociation-excluded, few-ulp | 2/8192 | 1 | TODO |
| HIT-0002 | `hw_fa_1x1x128x256x64_c0_s2` | `None` | `{"tl.disable_warp_specialized": true}` | hopper-only | pass-class:numerics-changing, kernel-has-gemm, kernel-has-reduce, input-mode:normal, exact-int-accum:reassociation-excluded, few-ulp | 2/8192 | 1 | TODO |
