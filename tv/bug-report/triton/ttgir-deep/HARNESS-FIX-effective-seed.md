# Harness bug found while auditing the heap-shift gate, and what it affected

Adding line `ttgir-broad`'s allocator-perturbation gate immediately dropped two
findings. Checking *why* turned up a bug in this line's triage chain rather than
two fake findings.

## The bug

The sweep tries several inputs per experiment: distribution `dist` at rep 0 and
rep 1, where rep `r` seeds the generator with `seed + 7919*r`. `compare`
recorded **which distribution** failed but the re-checks in `handle_mismatch`
all re-ran at **rep 0**. So a mismatch first seen at rep 1 was re-checked on a
different input:

* the self-consistency check tested the wrong input,
* `bisect` could not reproduce, so it returned the whole drop set instead of the
  single culprit,
* the 3x re-confirm read 0/3,
* the new heap-shift gate saw 0/3 and **discarded the finding as fake**,
* and the repro command printed a seed that does not reproduce.

Fixed by computing the effective seed once (`seed + 7919*rep`) and threading it
through the self-consistency check, the bisect, the re-confirm, the heap-shift
gate, the integer-input control, the shrink step and the repro command. Reports
now also carry `effective-seed` and `rep-offset` labels.

## Audit 1: do the already-published reports reproduce?

Every repro command printed in a committed report was run again, exactly as
published. All of them reproduce:

```
HIT-0001 -> RESULT: MISMATCH      HIT-0009 -> RESULT: MISMATCH
HIT-0003 -> RESULT: MISMATCH      HIT-0013 -> RESULT: MISMATCH
HIT-0004 -> RESULT: MISMATCH      HIT-0014 -> RESULT: MISMATCH
```

So no committed report is affected — they were all rep 0. That is consistent
with the failure mode: a rep-1 finding would have shown 0/3 on the re-confirm or
been thrown away by the gate, so it never reached a report in the first place.

## Audit 2: the two findings the gate dropped

Both were re-tested at rep 0 and rep 1 with the fix in place.

| case | culprit set | dist | rep 0 | rep 1 | verdict |
|---|---|---|---|---|---|
| `mm_epilogue[BK=64,BM=128,BN=128,K=256,M=512,N=512,NS=3,W=4,WS=1,dt=fp16]` | `interleave-tmem, optimize-dot-operands-1` | wide | equal | equal | **correctly dropped** — does not reproduce on either input |
| `attn[BM=64,D=64,M=512,N=512,NS=2,W=4,WS=0,dt=fp16]` | `combine-tensor-select-and-if, hopper-warpspec, optimize-dot-operands-1, remove-layout-conversions-2, remove-layout-conversions-3` | intvalued | equal | **MISMATCH 1/512, max_abs 1.19e-07, heap-shift 3/3** | **falsely dropped**, recovered below |

So the gate itself is sound — one of the two really was unreproducible — but the
seed bug made it throw away a genuine one.

## The recovered finding, recorded here so it is not lost

```
case:            attn[BM=64,D=64,M=512,N=512,NS=2,W=4,WS=0,dt=fp16]
culprit set:     combine-tensor-select-and-if, hopper-warpspec,
                 optimize-dot-operands-1, remove-layout-conversions-2,
                 remove-layout-conversions-3
distribution:    intvalued
effective seed:  827063060   (base 827055141, rep 1)
difference:      1 of 512 elements, max_abs_diff 1.1920928955078125e-07
heap-shift:      3/3 survived
```

Reproduce with:

```bash
source /home/youngzt/fuzz/ttgir-deep/env.sh
python -m harness.repro --family attn \
  --cfg '{"dt":"fp16","M":512,"N":512,"BM":64,"D":64,"NS":2,"WS":0,"W":4}' \
  --drop '["combine-tensor-select-and-if","hopper-warpspec","optimize-dot-operands-1","remove-layout-conversions-2","remove-layout-conversions-3"]' \
  --dist intvalued --seed 827063060
```

Labels, in the PLAN section 4 sense: the culprit set has not been bisected to a
single pass yet (the sweep will redo that now the seed is right); the set is
layout-heavy; `attn` is the one family where the integer-input control does
**not** apply, because `exp2` inside the loop makes the second dot's operands
non-integer, so this is consistent with a legitimate reduction reordering. It is
recorded, not judged.
