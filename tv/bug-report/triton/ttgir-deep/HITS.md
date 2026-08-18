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
