# TileLang differential fuzzing -- findings

Scope: bit-equivalence only.  A row is: two compilations of the same
PrimFunc, differing only in the pass set, produced different bits on
the same input.  `SMT verdict` is filled in by a reviewing agent.

| id | kernel | culprit | pass_configs | arch | labels | ndiff/total | max ulp | SMT verdict |
|---|---|---|---|---|---|---|---|---|
| HIT-0001 | `cap_081a26caff856329` | `None` | `{"tl.config_index_bitwidth": 64}` | generic | pass-class:bit-preserving, input-mode:captured, exact-int-accum:reassociation-excluded, gross | 3/7 | 2097217536 | TODO |
| HIT-0002 | `cap_0b881ee73fbdf83c` | `None` | `{"tl.disable_wgmma": true}` | hopper-only | pass-class:bit-preserving, kernel-has-gemm, kernel-has-reduce, input-mode:captured, exact-int-accum:reassociation-excluded, gross | 22332/65536 | 16384 | TODO |
