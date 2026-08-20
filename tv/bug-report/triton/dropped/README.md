# Worked examples of the rejected reports

The Triton fuzzing campaign filed **562 reports** across its two rounds and
seven lines. The review kept **31**, rejected **512**, and left **19** parked and
undecided (listed in `tv/bug-report/PARKED.md`). This directory holds **6**
frozen copies of rejected reports, one per rejection mechanism, so that a reader
of `bug-report/triton/` can see *what was filtered out and why* instead of only
seeing what survived.

These are illustrative samples, not the record. **The complete set of rejected
reports is in `/home/youngzt/tv/bug-report-rejected/triton/`**, outside this
repository, and `/home/youngzt/tv/bug-report-rejected/REJECTED.md` lists every
single disposition with its criterion and a full reason -- for most reports a
paragraph, not a sentence.

**These copies are frozen.** They are the report text as the fuzzing line wrote
it, with only a banner added on top. Nothing in the report body has been
corrected -- in two cases (`ttgir-broad-HIT-0063`, `r2-inductor-HIT-0179`) the
banner directly contradicts a label in the report, on purpose, because that
contradiction is the lesson. Later passes do not update these copies.

The criteria that decide keep from reject are in
`tv/bug-report/REVIEW-CRITERIA.md`.

## What is here

| example | criterion | the one measurement that settled it |
|---|---|---|
| `r2-inductor-HIT-0174` | A4 write-write race | the store index evaluated by hand over the launch domain: `ks0 = 63` against a 128-long row, so **947 of 1073 addresses have 2-3 writers** with different values |
| `r2-corpus-HIT-0110` | A2 out-of-bounds store | measured write map: both arms write **1,021,643 fp16 words into the kernel's own input buffer**, at indices 4..16,717,826, exactly the predicted overflow |
| `ttgir-deep-HIT-0020` | A3 non-deterministic under one binary | the **reference disagrees with itself more often than with the candidate**: ref-vs-ref 6/20, ref-vs-cand 3/20, cand-vs-cand 0/20 |
| `ttgir-broad-HIT-0063` | B1 reassociation | the plain integer control is invalid here (`exp`), so a valid one was built: with all inputs equal `exp(0) = 1.0` exactly and the two arms go **bit-identical** |
| `r2-inductor-HIT-0179` | B3 fma contraction | override the draw to `enable_fp_fusion = False`: the two arms are **equal**. The kernel has **no `tt.reduce` at all**, so B1 is not even available |
| `ttgir-broad-HIT-0083` | B1 dot+add accumulator fusion | ptxas `-O0` with `fma.rn.f32` unchanged **1024 vs 1024** and only the trailing `add.f32` going **32 -> 0** -- the fusion itself, not contraction |

Each directory holds the bannered report, the IR pair and diff under `ir/`, and
the report's `facts.json` / `repro.json` / `spec.json` (or `meta.json` for the
`ttgir-deep` line) at the top level.

## The whole rejection set, by mechanism

Counted from the reviewer ledger over the 476 rejections it records. Reports are
filed under the first criterion that settles them, so a report rejected as "B1,
and B2 confirms it" is counted once, under B1.

| n | mechanism | criterion |
|---|---|---|
| 272 | a numeric licence: reduction reorder, accumulator fusion, precision | B1 |
| 97 | duplicate of a report already kept | B4 |
| 27 | `mul + add -> fma` contraction | B3 |
| 25 | the difference disappears under exact-integer inputs | B2 |
| 17 | write-write race from a non-injective store index or an atomic | A4 |
| 14 | not a defect -- a documented pass-order precondition, mostly `PlanCTA` | -- |
| 10 | the report's own repro command does not reproduce | A1 |
| 6 | not deterministic under a single binary | A3 |
| 6 | out-of-bounds or uninitialised read | A2 |
| 2 | the two IRs are the same program | A5 |
| **476** | **total, as recorded in the ledger** | |

A bookkeeping note so the numbers can be checked: there are **512** rejected
report files on disk and the ledger records **476** of them as `reject`. The
other 36 were moved out before ledger tracking began and have no ledger entry.
Every ledger rejection is present on disk; none is missing.

Unlike the TileLang side, this line produced real examples of **all three** UB
classes, so all three are represented here. The A3 rejections split into two
groups: **4 of the 6** are one family in round 1 -- warp-specialized `mm_tma`,
`ttgir-deep/HIT-0018`, `HIT-0019`, `HIT-0020`, `HIT-0022`, with `HIT-0015` a
fifth member filed under A1 first -- and the other **2** are round-2
(`r2-inductor/HIT-0007`, `HIT-0014`), both A3 **and** A4 at once, where a patched
shape made the store index non-injective *and* the arm went non-deterministic as
a result.

## Two lessons this side of the campaign produced about its own method

These are the most instructive things the Triton lines found out about
themselves, so they are written here rather than only inside individual reports.

### 1. The harness's own shape or scalar draw was the largest single source of false findings

The fuzzer does not only pick a kernel and a pass list. It also **patches the
kernel's scalar arguments and shapes**. When a patched scalar breaks a stride
relationship the kernel's author took for granted, the kernel becomes undefined
behaviour -- and then the two builds are allowed to disagree, for reasons that
have nothing to do with the compiler. This is what most A2 and A4 rejections are.

`r2-inductor-HIT-0174` in this directory is the case worth studying, because
**it survived every automatic control**:

* the all-inputs-equal exactness control -- **still differs** (16 elements,
  `max_abs` 63.0);
* `DISABLE_PTXAS_OPT=1` -- **still differs**.

On the printed labels that is the exact fingerprint of the two ptxas classes the
review **did** keep. The only thing that separated it from them was working out
the store index by hand: `ks0 = 63` against a 128-long row means consecutive rows
overlap, giving 947 of 1073 addresses two or three writers with different values.

The automatic store-injectivity screen `fz/storeidx.py` returned **`unknown`**
here ("nested for block"). The lesson is narrow and practical: **an `unknown`
from a screen is not a pass.** It is a request for manual work, and this report is
what a skipped request looks like. The same screen returned `unknown` on the two
`HIT-0099` occurrences appended in this pass, and there the hand check came back
clean -- so `unknown` really is uninformative in both directions.

### 2. `reassociation_excluded: true` has been a false positive twice

The harness prints a label meaning "this difference survives the exact-integer
control, so a reordered fp32 sum does not explain it". REVIEW-CRITERIA §2 treats
that as a **keep** signal. Twice it was wrong, for two different reasons:

* **`r2-oracle/HIT-0019` -- the real cause was UB.** The label is *correct* that
  a reordered sum does not explain the difference. It is just not the whole
  story: the kernel's store reaches element 2,097,151 while its fp32 slot holds
  131,072, so it is 1,966,080 elements out of bounds, past the whole pool
  (`task_oob_check -> in_bounds=False`). Rejected A2. **A true "not
  reassociation" label says nothing about whether the program is defined.**
* **`r2-corpus/HIT-0131` -- neither control is valid.** The combine is
  `logaddexp` (`log1p(exp(min-max)) + max`), which is irrational under
  whole-number fills *and* under all-equal fills (`log 2`). So **both** exactness
  controls are meaningless on this kernel and the label built on them carries no
  information. It was settled instead with a float64 model of the same scan: both
  arms are within 1.3e-7 relative of the fp64 result, and the reference is the
  closer one on 59 of the 108 differing elements against the candidate's 49 --
  neither arm is wrong. Rejected B1.

`ttgir-broad-HIT-0063` in this directory is the same trap caught one step
earlier, and is included because it shows the repair as well as the failure: the
plain integer control is invalid because of `exp`, so a **valid** exact control
was constructed (make every input equal, then `exp(0) = 1.0` and every partial
sum is a small integer) and it goes bit-identical. The general rule: **any
transcendental, division or square root in the chain invalidates the plain
whole-number control.** Do not read its output; replace the control.

## Two smaller notes for anyone re-reading these

**Say which checkout a line number is in.** The compiler under test is
`/home/youngzt/fuzz/triton-public` @ `3277063a6`, which is *not* the repository
this file lives in. `CombineDotAddPattern` is at `Combine.cpp:250` there and at
`Combine.cpp:234` in `/home/youngzt/tv/triton`, and each line number looks wrong
from the other tree. A pointer without its checkout is worse than no pointer:
this exact confusion was walked into once during the wrap-up pass.

**Dot+add accumulator fusion is not a round-2 discovery.** It is worth saying
plainly because the opposite was believed at one point. Round 1 hit
`CombineDotAddPattern` at least **17** times -- 16 in `ttgir-broad` and once in
`ttir-broad` -- and `ttir-broad/HIT-0007` is the **kept** representative of the
class, with the round-1 rejections recorded against it as B4 duplicates. The
example chosen here, `ttgir-broad-HIT-0083`, is deliberately a round-1 report for
that reason. What round 2 added was volume (22 more) and a second kernel family
(conv-transpose and conv3d rather than plain matmul).
