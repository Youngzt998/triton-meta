# Worked examples of the rejected reports

The TileLang fuzzing campaign filed **237 reports** across its two rounds. The
review kept **10** of them, rejected **226**, and left **1** parked and
undecided (`r2/HIT-0122`). This directory holds **6** frozen copies of rejected
reports, one per rejection mechanism, so that a reader of `bug-report/tilelang/`
can see *what was filtered out and why* instead of only seeing what survived.

These are illustrative samples, not the record. **The complete set of rejected
reports is in `/home/youngzt/tv/bug-report-rejected/tilelang/`** (round 1) and
`/home/youngzt/tv/bug-report-rejected/tilelang/r2/` (round 2), outside this
repository, and `/home/youngzt/tv/bug-report-rejected/REJECTED.md` lists every
single disposition with its criterion and a one-sentence reason. For round 2,
`r2/HITS.md` and `r2/RECURRENCES.md` in this repository also carry a
`review verdict` column, so all 152 of that round's rows can be read here
without leaving the repo.

**These copies are frozen.** They are the report text as the fuzzing line wrote
it, with only a banner added on top. Nothing in the report body has been
corrected -- in one case (`r2-HIT-0137`) the banner directly contradicts a line
in the report, on purpose. Later passes do not update these copies.

The criteria that decide keep from reject are in
`tv/bug-report/REVIEW-CRITERIA.md`.

## What is here

| example | criterion | the one measurement that settled it |
|---|---|---|
| `r1-HIT-0011` | A2 uninitialised read | the kernel never writes `a_fp32_local` and never reads its input; 2048/2048 elements are garbage registers |
| `r1-HIT-0077` | A2 uninitialised read | same kernel; differs on 19 of 20 draws **including `zeros`** |
| `r1-HIT-0003` | B1 + B2 reassociation | integer-valued inputs (`--mode intvalued`): **no difference** |
| `r1-HIT-0008` | B1 fast math | drop `tl.enable_fast_math`: **no difference**; the two values are `9.99e-41` and `0.0`, a denormal flushed by `-ftz=true` |
| `r2-HIT-0137` | B1 fast math | drop `tl.enable_fast_math`: the one differing draw **stops differing** |
| `r2-HIT-0148` | B3 fma contraction | recompile both arms with `-fmad=false`: **0 of 7 differing draws still differ** |

## What was re-measured while this directory was built

The numbers in the banners are not only quoted from an earlier pass. Four of the
six were re-run on 2026-08-20:

* `r1-HIT-0003` -- reproduced exactly; `--mode intvalued` equal on **3 of 3**
  seeds tried.
* `r1-HIT-0008` -- reproduced exactly; equal once `tl.enable_fast_math` is
  removed.
* `r2-HIT-0137` -- reproduced exactly; equal once `tl.enable_fast_math` is
  removed.
* `r2-HIT-0148` -- reproduced, plus a 12-lowering compile-determinism gate per
  arm and the `-fmad=false` sweep over all 7 differing draws.

The round-1 examples were re-run on the **round-2** harness, which keeps its own
copy of the same corpus, so round 1's frozen directory was not touched.

`r1-HIT-0011` and `r1-HIT-0077` cannot be re-run at all: their kernel was
dropped from both corpora after the third false positive. That is the correct
outcome, and their evidence is the kernel source printed in the report, which
needs no run.

## The whole rejection set, by mechanism

| n | mechanism | criterion |
|---|---|---|
| 168 | a numeric licence, overwhelmingly fast math | B1 (some also B2) |
| 34 | duplicate of a report already kept | B4 |
| 21 | `mul + add -> fma` contraction | B3 |
| 3 | uninitialised read by the test kernel | A2 |
| **226** | **total rejected** (of 237 reports: 10 kept, 1 parked) | |

## Two things the criteria list but this line never produced

`REVIEW-CRITERIA.md` section 1 also names **A3** (the kernel is not
deterministic under a single binary) and **A4** (a write-write race from a
data-dependent store address or an atomic). **The TileLang line produced zero of
each**, so there is no example of them here. That is not luck; it is by
construction. The harness dropped both classes *before* they could become
reports:

* a determinism pre-screen ran the same build twice on the same input, on all 20
  draws, and dropped the kernel if it disagreed with itself;
* an IR scan (`tlfz/irscan.py`) dropped every kernel with a scatter store (a
  store address derived from loaded data) or an atomic.

So all three Group-A rejections that did get through are the same mechanism -- a
read of a never-written local fragment, which neither gate can see -- and all
three are the same corpus kernel, `cap_7fd65a04c9705768`. Two of the three are
here; the third, `HIT-0047`, is the same kernel again with a third pass knob.
After the third one the line added an uninitialised-local scan on the reference
side and dropped the kernel from the corpus.

## The fast-math class was inflated by a harness defect

This matters for anyone reading the quarantine, so it is written out here rather
than only in the reports.

`tl.enable_fast_math` was supposed to be a **shared** axis -- set identically on
both arms, so that it can never by itself explain a difference. For most of the
campaign it leaked onto the **candidate arm only**, while the report printed
*"fast math was OFF on both arms"*. **110 of 135 reports written before the fix
carry that false line.** The fix landed on 2026-08-19 at about 22:10.

Two consequences a reader should know:

1. **A pre-fix report's `## Fast-math control` section cannot be trusted.**
   Check the *minimal variant* printed a few lines above it instead: if it
   contains `tl.enable_fast_math`, fast math was on one arm only. `r2-HIT-0137`
   in this directory is deliberately such a report, so the defect is visible and
   not merely described. Every pre-fix rejection in `REJECTED.md` had its
   ablation re-run by hand by a reviewer before the verdict was written.
2. **After the fix the class disappears completely.** Of the 12 rejections
   written after 22:10 -- the entire post-fix population, not a sample -- **none
   is a fast-math rejection**: 4 are B3 contraction and 8 are B4 duplicates of
   bugs already kept. So the fast-math differences were real numeric
   differences, but they were measuring the flag rather than the pass under
   test.

## Two more habits these examples are meant to teach

**Size is not evidence, in either direction.** Section 3 of the criteria says
"the difference is tiny" is never on its own a reason to reject -- `r1-HIT-0008`
is 2 elements of 30720 and was rejected on its ablation, not its size. The
mirror also holds and is easier to get wrong: `r1-HIT-0003` differs on
**1047889 of 1048576** elements at up to 3191955456 ulp and is entirely
licensed. "Almost everything is wrong by a lot" is not a reason to keep.

**The same finding gets filed more than once.** The sweep bisects each hit down
to a minimal variant, and unrelated starting variants often bisect to the same
place. `r2-HIT-0148` is one of **three** reports (`0148`, `0149`, `0150`) that
are the same instance with the same output bits, reached through three unrelated
toggles; `r2/HIT-0151` and `r2/HIT-0152` are one compiled pair recorded twice.
`RECURRENCES.md` in each round's directory counts how often each signature
recurred across the whole sweep, which no individual report records.
