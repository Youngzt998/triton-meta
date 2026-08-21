# ttgir-deep hits

Scope of this table: **bit-equivalence only**.  Every row is one case
where two compilations that differ only in the TTGIR pass set produced
different bits on the same input.  No row claims to be a compiler bug;
the columns are labels to filter on.  `int-input control = differs
(reassoc ruled out)` is the strong one: integer inputs keep the fp32
accumulator exact, so the difference cannot be a reordered sum.

| id | title | verdict | arch | SMT verdict | culprit pass | pass-class | ndiff/total | max_abs_diff | int-input control | heap-shift |
|---|---|---|---|---|---|---|---|---|---|---|
| HIT-0001 | coalesce on nested | needs-review | generic | TODO | `coalesce` | ? | ?/? | ? | ? | ? |
| HIT-0002 | coalesce on nested | needs-review | generic | TODO | `coalesce` | ? | ?/? | ? | ? | ? |
| HIT-0003 | coalesce on loop_reduce | needs-review | hopper-only | TODO | `coalesce` | ? | ?/? | ? | ? | ? |
| HIT-0004 | remove-layout-conversions-1 on nested | needs-review | hopper-only | TODO | `remove-layout-conversions-1` | ? | ?/? | ? | ? | ? |
| HIT-0005 | coalesce on nested | needs-review | hopper-only | TODO | `coalesce` | ? | ?/? | ? | ? | ? |
| HIT-0006 | coalesce on loop_reduce | needs-review | hopper-only | TODO | `coalesce` | ? | ?/? | ? | ? | ? |
| HIT-0007 | coalesce on nested | needs-review | generic | TODO | `coalesce` | ? | ?/? | ? | ? | ? |
| HIT-0008 | coalesce on nested | needs-review | generic | TODO | `coalesce` | ? | ?/? | ? | ? | ? |
| HIT-0009 | hopper-warpspec on mm_epilogue | CONFIRMED | hopper-only | TODO | `hopper-warpspec` | ? | ?/? | ? | ? | ? |
| HIT-0010 | remove-layout-conversions-1 on nested | needs-review | hopper-only | TODO | `remove-layout-conversions-1` | ? | ?/? | ? | ? | ? |
| HIT-0011 | coalesce on loop_reduce | needs-review | hopper-only | TODO | `coalesce` | ? | ?/? | ? | ? | ? |
| HIT-0012 | coalesce on nested | needs-review | generic | TODO | `coalesce` | ? | ?/? | ? | ? | ? |
| HIT-0013 | remove-layout-conversions-2,remove-layout-conversions-3 on attn | needs-review | hopper-only | TODO | `remove-layout-conversions-2,remove-layout-conversions-3` | ? | ?/? | ? | ? | ? |
| HIT-0014 | remove-layout-conversions-1,remove-layout-conversions-2,remove-layout-conversions-3 on loop_reduce | needs-review | generic | TODO | `remove-layout-conversions-1,remove-layout-conversions-2,remove-layout-conversions-3` | layout | 22/64 | 0.0001220703125 | equal | 3/3 |
| HIT-0015 | RETRACTED - interleave-tmem,loop-aware-cse-2,optimize-dot-operands-1,pipeline,plan-cta on mm_tma | retracted: reference build is not self-consistent ( | hopper-only | TODO | `interleave-tmem,loop-aware-cse-2,optimize-dot-operands-1,pipeline,plan-cta` | clean,layout,lower,sched | 700/272448 | 421.0 | equal | 2/3 |
| HIT-0016 | RETRACTED - optimize-dot-operands-1 on mm_tma | retracted: reference build is not self-consistent ( | hopper-only | TODO | `optimize-dot-operands-1` | layout | 7424/1048576 | 1.817990202823694e-15 | equal | 1/3 |
| HIT-0017 | remove-layout-conversions-1,remove-layout-conversions-2 on loop_reduce | needs-review | hopper-only | TODO | `remove-layout-conversions-1,remove-layout-conversions-2` | layout | 51/128 | 0.000244140625 | equal | 3/3 |
| HIT-0018 | RETRACTED - optimize-dot-operands-1,schedule-loops on mm_tma | retracted: reference build is not self-consistent ( | hopper-only | TODO | `optimize-dot-operands-1,schedule-loops` | layout,sched | 1/1048576 | 0.0 | equal | 1/3 |
| HIT-0019 | RETRACTED - optimize-dot-operands-1,schedule-loops,triton-licm on mm_tma | retracted: reference build is not self-consistent ( | hopper-only | TODO | `optimize-dot-operands-1,schedule-loops,triton-licm` | layout,sched | 17914/1048576 | 2.491062911502695e-15 | equal | 2/3 |
| HIT-0020 | RETRACTED - optimize-dot-operands-1,triton-licm on mm_tma | retracted: difference is +0.0 vs -0.0 on 1 of 10485 | hopper-only | TODO | `optimize-dot-operands-1,triton-licm` | layout,sched | 1/1048576 | 0.0 | equal | 2/3 |
| HIT-0021 | RETRACTED - canonicalize-a,combine-tensor-select-and-if,optimize-dot-operands-1,remove-layout-conversions-1 on mm_tma | retracted: does not reproduce at its own seed | hopper-only | TODO | `canonicalize-a,combine-tensor-select-and-if,optimize-dot-operands-1,remove-layout-conversions-1` | clean,layout | 4329/272448 | 518.0 | equal | 1/3 |
| HIT-0022 | RETRACTED - combine-tensor-select-and-if,optimize-dot-operands-1 on mm_tma | retracted: reference build disagrees with itself in | hopper-only | TODO | `combine-tensor-select-and-if,optimize-dot-operands-1` | clean,layout | 1273/272448 | 1.3530843112619095e-15 | equal | 1/3 |
| HIT-0023 | remove-layout-conversions-2,remove-layout-conversions-3,schedule-loops on attn | needs-review | hopper-only | TODO | `remove-layout-conversions-2,remove-layout-conversions-3,schedule-loops` | layout,sched | 371/1024 | 1.9073486328125e-06 | equal | 3/3 |
| HIT-0024 | optimize-dot-operands-1,remove-layout-conversions-1,remove-layout-conversions-2,remove-layout-conversions-3 on attn | needs-review | hopper-only | TODO | `optimize-dot-operands-1,remove-layout-conversions-1,remove-layout-conversions-2,remove-layout-conversions-3` | layout | 1/1024 | 1.1920928955078125e-07 | differs (control n/a) | 3/3 |
| HIT-0025 | optimize-dot-operands-1,pipeline,remove-layout-conversions-2,remove-layout-conversions-3 on attn | needs-review | hopper-only | TODO | `optimize-dot-operands-1,pipeline,remove-layout-conversions-2,remove-layout-conversions-3` | layout,sched | 338/1024 | 1.9073486328125e-06 | equal | 3/3 |
| HIT-0026 | pipeline,plan-cta,remove-layout-conversions-2,remove-layout-conversions-3,triton-licm on attn | needs-review | hopper-only | TODO | `pipeline,plan-cta,remove-layout-conversions-2,remove-layout-conversions-3,triton-licm` | layout,sched | 371/1024 | 1.9073486328125e-06 | differs (control n/a) | 3/3 |
| HIT-0027 | assign-latencies,plan-cta,remove-layout-conversions-2,remove-layout-conversions-3 on attn | needs-review | hopper-only | TODO | `assign-latencies,plan-cta,remove-layout-conversions-2,remove-layout-conversions-3` | layout,sched | 378/1024 | 2.86102294921875e-06 | equal | 3/3 |
| HIT-0028 | optimize-thread-locality on nested | needs-review | hopper-only | TODO | `optimize-thread-locality` | numeric | 182/256 | 0.078125 | equal | 3/3 |
| HIT-0029 | optimize-dot-operands-2,remove-layout-conversions-1,remove-layout-conversions-2,remove-layout-conversions-3 on loop_reduce | needs-review | generic | TODO | `optimize-dot-operands-2,remove-layout-conversions-1,remove-layout-conversions-2,remove-layout-conversions-3` | layout | 42/128 | 4096.0 | equal | 3/3 |
| HIT-0030 | optimize-dot-operands-1,remove-layout-conversions-1,remove-layout-conversions-2,remove-layout-conversions-3 on loop_reduce | needs-review | generic | TODO | `optimize-dot-operands-1,remove-layout-conversions-1,remove-layout-conversions-2,remove-layout-conversions-3` | layout | 37/128 | 2048.0 | equal | 3/3 |
