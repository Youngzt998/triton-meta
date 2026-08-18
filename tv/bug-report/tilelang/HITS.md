# TileLang differential fuzzing -- findings

Scope: bit-equivalence only.  A row is: two compilations of the same
PrimFunc, differing only in the pass set, produced different bits on
the same input.  `SMT verdict` is filled in by a reviewing agent.

| id | kernel | culprit | pass_configs | arch | labels | ndiff/total | max ulp | SMT verdict |
|---|---|---|---|---|---|---|---|---|
| HIT-0001 | `cap_081a26caff856329` | `None` | `{"tl.config_index_bitwidth": 64}` | generic | pass-class:bit-preserving, input-mode:captured, exact-int-accum:reassociation-excluded, gross | 3/7 | 2097217536 | TODO |
