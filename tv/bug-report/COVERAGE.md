# Campaign coverage

What the Triton / TileLang compiler-fuzzing campaign actually covered, so a
reader can tell what a "no finding" result is worth.

This document is required by `/home/youngzt/fuzz/PLAN.md` §10b, amended by
`/home/youngzt/fuzz/PLAN-R2.md` §6 (which adds the shape-space and config-space
columns and says to report the two rounds separately).

## How to read it

* **Every number here is measured.** It is copied from a line's own
  `STATUS.md`, `FINAL_SUMMARY.md`, `checkpoint.json` / `state.json`, its source
  code, or it was computed by streaming that line's `results.jsonl` end to end.
  Where a number was computed from a log, the field name and the record count
  are given. Where a number is not in any file, the cell says **not measured** —
  nothing is estimated or back-derived.
* **Where two files disagree, both are given.** There is a section for that.
* **The two rounds are reported separately and are not merged.** They swept
  different axes; the comparison between them is the most useful thing here.
* **Round 2's five lines are not comparable to each other.** They measured
  different units. Each line's own unit is named in its row.

Frozen at: round 1 stopped 2026-08-19 ~14:40; round 2 stopped 2026-08-20
12:43–12:47. Review-side numbers (kept / rejected / parked) were read on
2026-08-20 13:36 PDT.

---

# Round 1 — four lines, `2026-08-17 → 2026-08-19`

Budget was 72 h per line. All four were stopped early, at **45.3–46.8 h**.

Round 1 pinned two axes on purpose (and did not know it): **input shape was
fixed per kernel** — whatever shape the corpus source used — and **compile
config was fixed** (`ttir-broad`'s corpus holds 526 kernels whose `num_warps`
histogram is `{4: 526}` with `num_stages` unset on all of them; `ttgir-broad`
defaults `num_warps=4`). That is why round 2 exists.

## Corpus

| line | corpus | usable after pre-screen | drops, by reason |
|---|---|---|---|
| `ttir-broad` | 526 specs (387 bench + 139 inductor) | **466** | badspec 37, nondeterministic 8, out-of-bounds 8, ub-race 6, crash 1 |
| `ttgir-broad` | 607 kernels (184 inductor + 423 bench) | **523** | unrunnable 31, ub-race 24, nondeterministic 17, suspect-unstable 12, no-output 4, too-slow 1 |
| `ttgir-deep` | 926 **generated** cases (not corpus kernels — a parameter grid over 8 hand-written kernel templates) | **926** prescreened | 1 blocklisted (non-deterministic / broken) |
| `fuzz-tilelang` | 607 usable + 165 dropped by the line's own count | **607** | ub-race 53, load_error 52, ref_compile_error 40, ref_run_error 15, input_error 2, nondeterministic 2, suspect-unstable 1 |

`ttgir-deep`'s 8 kernel families: `mm_tma`, `mm_tma_persist`, `attn`, `mm_ptr`,
`loop_reduce`, `nested`, `tma_copy`, `mm_epilogue`.

## Sweep volume and pass space

| line | full passes over the corpus | kernels swept | pass universe | single-pass toggles tried | distinct pass sets tried |
|---|---|---|---|---|---|
| `ttir-broad` | round **319** | 148,857 | **22** flags (tier1 11 + tier2 11; 19 distinct pass names, 3 are `--canonicalize` option variants) | **not measured** | **not measured** |
| `ttgir-broad` | cycle **84** | not measured | **56** base passes (scraped from this build's `triton-opt --help`; the classification table has 56 entries) | **82** distinct rendered flags / 56 base passes | **217,613** ordered candidate sets (153,445 as unordered sets) |
| `ttgir-deep` | not a cycle counter | not measured | **35** droppable of **40** pipeline steps on `cuda:90`; 21 named "focus" passes weighted 8×, of which 20 are droppable | **35** (every droppable pass, alone, at least once) | **51,818** distinct drop sets |
| `fuzz-tilelang` | cycle **56** | 23,713 jobs | **63** single toggles = 30 droppable passes + 33 pass-config keys, out of a 64-step pipeline | **63** — all of them | **327,944** distinct (skip-set, config) variants; 31,655 distinct skip-sets alone |

`ttir-broad`'s pass-set counts are **not recoverable**: its `results.jsonl`
holds only hits, instances and discards (1,270 records in total), and its
checkpoint keeps only aggregate `jobs` / `compares` / `noop`. What it does
record: **38,377** pipelines run (armA 36,544 + armB 1,833).

The three measured pass-set counts were computed by streaming each log
line by line and collecting a `set()`:
`ttgir-broad/run/results.jsonl` field `added`, 1,064,424 records;
`ttgir-deep/logs/results.jsonl` field `drop` (sorted first), 655,654 records;
`fuzz-tilelang/run/results.jsonl` field `v` = `{skip, cfg}`, 792,142 records
(744,486 of them `variant` records).

For `ttgir-broad` and the round-2 Triton lines, only the *candidate delta* is
logged, not the reference prefix, so "distinct pass sets" means distinct
candidate sets, not distinct (reference, candidate) pipeline pairs.

## Shape space and config space

Round 1 has **no shape-space or config-space counters** — that is the whole
point of round 2. What is on record:

* `ttir-broad`: **1 shape per kernel** and **1 config** — `num_warps` is 4 on
  all 526 specs, `num_stages` unset on all 526.
* `ttgir-broad`: `num_warps` defaults to 4 (`fz/specs.py:156`, `fz/corpus.py:51`).
* `ttgir-deep`: shape and config are part of the enumerated case grid (dtype ×
  BM/BN/BK × num_stages × M/N/K × warp-spec × num_warps), so the 926 cases
  *are* the shape/config space — but it is a fixed grid, not a draw.
* `fuzz-tilelang`: **not measured**.

## Comparisons

| line | unit | total | no-op skips | mismatches | errors | input draws per compiled pair |
|---|---|---|---|---|---|---|
| `ttir-broad` | compiled pair × input draw | **721,231** | 985,148 (pass set left the IR or the cubin unchanged) | see outcomes | 0 opt failures | **20** |
| `ttgir-broad` | task = one compiled pair | **1,064,123** | 635,140 `ir-identical` | 1,330 | 179,761 (+3 timeouts) | **20** |
| `ttgir-deep` | experiment = one compiled pair | **640,916** | 266,696 vacuous (pass did nothing) | 7,169 | 14 compile + 176 run | **20** |
| `fuzz-tilelang` | variant | **743,260** | 531,529 pruned (identical lowered CUDA, nvcc skipped) | 1,194 DIFF | 5,799 (+503 crashes, 6 timeouts) | **20** |

Only `ttir-broad` counts comparisons at the input-draw level. The other three
count compiled pairs; each pair then saw 20 draws, but the product is not
recorded anywhere, so it is not given here.

## Outcomes

| line | raw mismatches | distinct findings after the line's own dedup | reports written | kept | rejected | parked |
|---|---|---|---|---|---|---|
| `ttir-broad` | 1,206 hit instances | **4** classes | 4 | **1** | 3 | 0 |
| `ttgir-broad` | 1,330 | **58** bit differences | 59 files | **2** | 66 | 0 |
| `ttgir-deep` | 7,169 (of which 7,000 were repeat sightings of an already-reported finding) | **30** | 30 | **1** | 29 | 0 |
| `fuzz-tilelang` | 1,194 DIFF | **82** signatures | 82 (+1 `CODEGEN` filed by the reviewer) | **6** | 77 | 0 |
| **round 1 total** | | | **175 files** | **10** | **175** | **0** |

Gate drops that never became reports:
`ttir-broad` nondeterministic 5, flaky 4, ub-race on rescan 6;
`ttgir-deep` heap-shift 61, non-deterministic 60, signed-zero-only 3, throttled
on a suspect-unstable case 15;
`ttgir-broad` heap-shift 0.

`ttgir-broad` has **68 review dispositions but only 59 report files**. The
extra 9 ids are an early block of reports that were retracted during the run
(`tv/bug-report/triton/ttgir-broad/RETRACTED.md`; the hit-id counter was also
reset once during setup, so a few numbers were reused). The 59 files that
survive are `HIT-0035` … `HIT-0093`.

---

# Round 2 — five lines, `2026-08-19 → 2026-08-20`

Budget was 72 h per line. All five were stopped at **~20 h** (19.7–20.1 h).

Round 2 opened the two axes round 1 had pinned — shape (including **base
pointer alignment**, which no round-1 line varied) and compile config — and
added two oracles that can see a bug in the *default* pipeline.

🔴 **The five lines measured different things and their totals must not be put
in one column.** `r2-compile` never runs a kernel: its unit is a
(kernel × shape × config × arch) compile point. `r2-oracle` runs two oracles
with two more units. The unit is named in every row.

## Corpus

| line | corpus | usable after pre-screen | drops, by reason |
|---|---|---|---|
| **R1 `r2-corpus`** | 607 kernels (the round-1 corpus) | **474** | suspect-unstable 37, too-slow 25, unrunnable 25, nondeterministic 21, ub-race 17, no-output 8. Plus 17 kernels pinned back to their base shape after faults. |
| **R2 `r2-inductor`** | generated: 7,460 random torch programs (7,347 compiled, 113 failed) → 16,689 raw Triton kernels seen → **10,136 harvested** (deduped by structure) → 10,405 in the driver | **9,838** | nondeterministic 393, unrunnable 88, ub-race 66, no-output 13, too-slow 4, suspect-unstable 3 |
| **R3 `r2-oracle`** | 607 kernels | **478**; of those the interpreter can run **304** and only **139** are strict-comparison capable | too-slow 51, unrunnable 31, ub-race 20, nondeterministic 18, no-output 9 |
| **R4 `r2-compile`** | 607 kernels | 607 (compile-only, no pre-screen needed) | — |
| **R5 `fuzz-tilelang-r2`** | captured 476 usable + 143 dropped, **plus** generated instances from 14 families | **476** captured | ub-race 53, ref_run_error 50, ref_compile_error 39, heap_dependent 1 |

R3's `139 of 607` is the single most limiting number in round 2: the
interpreter oracle can make a **strict** bit-equality demand on only 139 of the
607 corpus kernels. Everywhere else it records a difference with a label and
proves nothing.

## Sweep volume and pass space

| line | full passes over the corpus | pass universe it could draw from | single-pass toggles tried | distinct pass sets tried |
|---|---|---|---|---|
| `r2-corpus` | cycle **13** | **78** = 56 TTGIR passes + 22 TTIR flags; **92** distinct rendered flags actually used | **92** | **57,705** distinct candidate pass sets |
| `r2-inductor` | cycle **5** | **56** TTGIR passes; **97** distinct rendered flags actually used | **94** | **536,315** distinct full pipelines by the driver's own counter; **145,101** distinct candidate deltas measured from the log — see below |
| `r2-oracle` | cycle **32** | **56** TTGIR passes (O2 tests one pass at a time for idempotence) | **not measured** | **not measured** |
| `r2-compile` | round **668** | not a sampled space: the IR-validity job **replays the production pipeline pass by pass**. **785,391** single-pass `triton-opt` replays. Distinct pass count: **not measured** | n/a | n/a |
| `fuzz-tilelang-r2` | cycle **19** | **30** droppable passes + **27** pass-config keys observed in use | **30** (all) | **146,320** distinct (skip-set, config) variants |

**Why `r2-inductor` has two pass-set numbers, and why they are both right.**
Its driver counter (`cov_passsets.txt`) hashes `ref_flags + candidate_delta`,
i.e. the whole pipeline, and gets **536,315**. Streaming its
`run/results.jsonl` (634,799 records with an `added` field) and hashing only
the candidate delta gets **145,101**. They count different objects. For a
like-for-like comparison against `r2-corpus`'s 57,705, use **145,101**.

Pass-set counts computed from logs, with the record counts scanned:
`r2-corpus/run/results.jsonl` field `added`, 263,030 records;
`r2-inductor/run/results.jsonl` field `added`, 634,799 records;
`fuzz-tilelang-r2/run/results.jsonl` field `v` = `{skip, cfg}`, 276,877 variant
records.

## Shape space and config space — the round-2 amendment

These are **what was actually compiled**, not what was possible.

| line | distinct shapes compiled | distinct config tuples compiled |
|---|---|---|
| `r2-corpus` | **5,682** distinct (kernel, shape) points | **12,242** |
| `r2-inductor` | **25,987** | **7,041** |
| `r2-oracle` | **71,957** shapes drawn | **17,338** configs drawn |
| `r2-compile` | **not measured** — it keeps no coverage set; each of its 1,231,940 points draws a fresh shape and config | **not measured** |
| `fuzz-tilelang-r2` | **2,930** | **3,168** |

The config axes that were drawn from (PLAN-R2 §2b, verified against this
build's `CUDAOptions`): `num_warps` ∈ {1,2,4,8,16,32}, `num_stages` ∈ 1…8,
`num_ctas` ∈ {1,2,4} (`r2-compile` also 8), `maxnreg` ∈ {None,128,96,64,32},
`default_dot_input_precision` ∈ {tf32, tf32x3, ieee, bf16x3, bf16x6},
`launch_cooperative_grid`, `launch_pdl`, `sanitize_overflow`.
`enable_fp_fusion` is held **False** for the main sweep and True on about 10%
of draws, which are labelled.

Shape draws thrown away because the kernel's store index map is not injective
(a write-write race at that shape, so undefined behaviour and out of scope):
`r2-corpus` **8** (0 kernels skipped for a round), `r2-inductor` **753**
(4 kernels skipped for a round).

## Comparisons — one row per line, each with its own unit

| line | unit | total | no-op skips | mismatches | errors | draws per pair |
|---|---|---|---|---|---|---|
| `r2-corpus` | task = one compiled pair | **263,208** (ttir 18,624 / ttgir 33,948 groups) | 188,521 `ir-identical` | 972 | 29,839 (+114 timeouts) | **20** |
| `r2-inductor` | task = one compiled pair | **634,152** | 355,673 `ir-identical` | 2,593 | 136,329 (+15 timeouts) | **18** |
| `r2-oracle` O1 | compiled kernel vs `TRITON_INTERPRET=1` | **71,680**, of which only **13,212 strict** | 3,873 skipped | **121** (strict only) | — | **8** per kernel per cycle |
| `r2-oracle` O2 | pass `P` vs `P;P` | **64,492** | 50,495 `ir-same` | **58** | 7,468 | **6** |
| `r2-compile` | **point = kernel × shape × config × arch** — nothing is run | **1,231,940** (sm80 419,625 / sm89 406,078 / sm90 406,237) | n/a | n/a — it hunts crashes, not bit differences | see the outcome table below | n/a |
| `fuzz-tilelang-r2` | variant | **276,061** | 187,679 pruned (identical lowered CUDA) | 1,176 DIFF | 2,252 (+56 crashes, +56 timeouts) | **20** |

`r2-inductor` uses **18** draws per pair, not 20. PLAN §10b's sentence "each
pair sees 20 input draws" is true of every line except that one, and except
`r2-oracle`, whose two oracles use 8 and 6.

Extra counters worth having:
`r2-inductor` recorded **110** compile asymmetries (one arm built, the other did
not — a finding under PLAN-R2 §2b) and **18** `triton-opt` crashes;
`fuzz-tilelang-r2` recorded **87** one-sided compile failures;
`r2-oracle` had **196** timeouts.

`r2-compile`'s outcome split over its 1,231,940 points:

| outcome | count | counted as a finding? |
|---|---|---|
| ok | 1,086,427 | no |
| frontend-reject | 127,362 | no |
| crash-assert | 4,405 | yes |
| timeout | 4,198 | yes |
| config-reject | 3,110 | no |
| invalid-ir | 2,976 | yes |
| invalid-ir-frontend | 2,671 | yes |
| resource-limit | 547 | no |
| crash-ptxas | 154 | yes |
| crash-python | 51 | yes |
| crash-signal | 39 | yes |
| **findings total** | **14,494** | |
| **noise total** | **1,217,446** | |

## Outcomes

| line | raw signal | distinct findings after the line's own dedup | reports written | kept | rejected | parked | never reviewed |
|---|---|---|---|---|---|---|---|
| `r2-corpus` | 972 mismatches | **143** bit differences + **22** compiler crashes | 165 | **4** | 155 (50 of them B4 duplicates) | **6** | 0 |
| `r2-inductor` | 2,593 mismatches | **471** bit differences | 400 | **1** | 187 (18 B4) | **12** | **200** |
| `r2-oracle` | 179 mismatches (O1 121 + O2 58) | 19 | 19 files, ids `HIT-0001`…`HIT-0023` | **2** (incl. `HIT-INTERP-BF16`, filed by the reviewer) | 21 (4 B4) | **1** | 0 |
| `r2-compile` | 14,494 findings | **75** distinct crash classes (by error-string signature) | 71 | **20** | 51 (33 B4) | 0 | 0 |
| `fuzz-tilelang-r2` | 1,176 DIFF | **152** signatures | 152 + 1 `CODEGEN` (+1 `HIT-COMPILE-NONDET` filed by the reviewer) | **4** | 149 (33 B4) | **1** | 0 |
| **round 2 total** | | | **808 files** | **31** | **563** | **20** | **200** |

Gate drops that never became reports: `r2-corpus` dropped **110** differences as
non-deterministic at the drawn shape/config and 0 as heap-shift artifacts;
`r2-inductor` dropped 3 as heap-shift artifacts; `r2-oracle` dropped 10 as
artifacts (5 of them because the kernel indexed outside the slot it was given);
`fuzz-tilelang-r2` dropped **51** points because an arm does not compile to the
same CUDA twice (49 on the reference at the point gate, 2 on the candidate at
the hit gate) and labelled 2 weak-oracle events.

### Round 2's raw report count overstates the number of distinct findings

Two measured examples, both from the review:

* **TileLang.** `r2/HIT-0148`, `r2/HIT-0149` and `r2/HIT-0150` are **one**
  `gemv` instance (`gen_gemv_4da8042d09cfec`) with the same output bits,
  reached through three unrelated toggles and filed three times. Separately,
  `r2/HIT-0151` and `r2/HIT-0152` are one compiled pair recorded twice.
* **`r2-compile`.** `HIT-0047`, `HIT-0060`, `HIT-0065`, `HIT-0067`, `HIT-0068`
  and `HIT-0071` are **one ptxas crash wearing six ids**. The line
  de-duplicates on the ptxas repro command line, which contains the random
  temporary file name `/tmp/tmpXXXXXXXX.ptx`; the normaliser masks digits but
  not letters, so every run's name became a new signature. (`HIT-0067` and
  `HIT-0068` are even the same kernel.) The same shape of defect made the
  `Insufficient registers (N) … in function <name>` signature produce **eleven**
  ids for one non-defect. The reviewer's own rollup
  (`tv/bug-report/triton/r2-compile/REVIEW.md`) regroups by root cause and folds
  22 further ids into 12 of the kept reports.

So `r2-compile`'s "75 distinct crash classes" is a count of **error strings**,
not of root causes, and it is an over-count by a known margin.

---

# Review outcome, whole campaign

Kept reports are the ones **tracked in git** under `tv/bug-report/`, excluding
`tv/bug-report/tilelang/dropped/` — that directory holds 6 frozen copies of
*rejected* reports, kept on purpose as worked examples, and counting them as
keeps would be wrong. Untracked report files under `tv/bug-report/` are parked,
not kept.

| | round 1 | round 2 | campaign |
|---|---|---|---|
| report files written by the fuzzing lines | 175 | 808 | 983 |
| ids given a review disposition | 185 | 614 | 799 |
| **kept** | **10** | **31** | **41** |
| **rejected** | **175** | **563** | **738** |
| **parked** (decided by nobody yet) | 0 | **20** | **20** |
| never reviewed | 0 | 200 (all `r2-inductor`) | 200 |

The 738 rejections match `/home/youngzt/tv/bug-report-rejected/REJECTED.md`
exactly (738 rows) and the quarantine on disk exactly (512 Triton report `.md`
files + 226 TileLang report directories = 738). Of the 738, **168** were
rejected as criterion **B4** — a duplicate of a report already kept.

The 41 kept reports:

| line | kept ids |
|---|---|
| `ttir-broad` | HIT-0007 |
| `ttgir-broad` | HIT-0008, HIT-0071 |
| `ttgir-deep` | HIT-0009 |
| `tilelang` (round 1) | CODEGEN-0001, HIT-0001, HIT-0004, HIT-0009, HIT-0040, HIT-0043 |
| `r2-corpus` | HIT-0017, HIT-0019, HIT-0024, HIT-0037 |
| `r2-inductor` | HIT-0099 |
| `r2-oracle` | HIT-0001, HIT-INTERP-BF16 |
| `r2-compile` | HIT-0001…HIT-0009, HIT-0016, HIT-0019, HIT-0038, HIT-0045, HIT-0047, HIT-0048, HIT-0049, HIT-0050, HIT-0055, HIT-0063, HIT-0066 |
| `tilelang/r2` | CODEGEN-0001, HIT-0008, HIT-0043, HIT-COMPILE-NONDET |

## Parked — 20 reports, neither kept nor rejected

These need their own cell. They are all "the difference has **zero value
magnitude**" cases — NaN payload bits only, or the sign of zero only, with
`max_abs_diff 0.0` and `max_ulp 0`. No owner ruling has been made, so they are
left in place, untracked and uncommitted.

| line | parked ids | count |
|---|---|---|
| `r2-corpus` | HIT-0071, 0104, 0142, 0160, 0164, 0165 | 6 |
| `r2-inductor` | HIT-0026, 0095, 0098, 0103, 0114, 0116, 0140, 0146, 0149, 0150, 0176, 0186 | 12 |
| `r2-oracle` | HIT-0009 | 1 |
| `tilelang/r2` | HIT-0122 (sign of zero, bfloat16 `fa`) | 1 |
| **total** | | **20** |

This set was **18** until the last review shard: `r2-inductor/HIT-0176` and
`HIT-0186` had been parked in an earlier round for the same reason and were
only folded into the count on 2026-08-20 at 13:00
(`/home/youngzt/fuzz/reviewer-triton/shard-inductor.json`). Ledger merges were
still in flight when this table was read, so the number can move again.

## 200 `r2-inductor` reports were never reviewed

`r2-inductor` wrote **400** report files. 200 of them have a disposition (187
rejected + 1 kept + 12 parked). **The other 200 were never reviewed at all**:
`HIT-0203` … `HIT-0400` (198 of them), plus `HIT-0013` and `HIT-0141`. They
are not findings and they are not non-findings; nobody looked. The last
`r2-inductor` review shard reported "no fifth wrong-answer mechanism" in the
batch it did read, and eight of its nine rejections were the same licensed
reduction reorder, so the prior is that the unread 200 are more of the same —
but that is a prior, not a measurement.

---

# Counters that changed meaning during a run

Several gates were added while the lines were running, so the counters they
feed do **not** cover the whole run. Where a counter starts part-way through,
this is when.

| line | gate added | when | what it means for the numbers |
|---|---|---|---|
| **R5 `fuzz-tilelang-r2`** | compile-determinism gate (point gate + hit gate) | **2026-08-19 19:40**, about 3.0 h into a 20.1 h run | The "compile-unstable points dropped 51" counter only covers the last ~17 h. Differences recorded before it may include points where an arm does not compile to the same CUDA twice (TileLang's `ThreadSync` lottery — that is kept report `HIT-0043`). Measured gate strength on the HIT-0043 point, 40 invocations each: point gate fires 26/40 (65 %), hit gate 30/40 (75 %) per arm. |
| **R5 `fuzz-tilelang-r2`** | degenerate-`fa` shape gate + `reference-mostly-nan` label (fires at ≥ 0.80 NaN) | with the mid-run harness fixes of **2026-08-19 17:45** | Only 2 weak-oracle events are counted (1 re-drawn shape, 1 labelled difference); earlier `fa` draws were not screened for a nearly all-NaN reference. |
| **R5 `fuzz-tilelang-r2`** | `tl.enable_fast_math` made a *shared* axis | **2026-08-19 ~22:10** | Before the fix the flag leaked onto the candidate arm only while the report text claimed it was off on both. **110 of the 135 reports written before the fix carry that false line.** Their `## Fast-math control` section cannot be trusted; every pre-fix rejection had its ablation re-run by hand. After the fix the fast-math rejection class disappears completely: of the 12 post-fix rejections, 0 are fast-math. |
| **R3 `r2-oracle`** | store-injectivity condition and the buffer-bounds ("out-of-slot") control | `fz/storeidx.py` last changed **2026-08-20 10:49**, driver **10:54** — under 2 h before shutdown | The bounds gate is why 5 artifacts were dropped for out-of-slot indexing and 0 for a non-injective store map. Those two counters cover only the tail of the run. |
| **R1 `r2-corpus`, R2 `r2-inductor`** | store-injectivity screen on the drawn shape (`fz/storeidx.py`) | file written **2026-08-19 18:37**; first firing in `r2-corpus` at t = 7,744.9 s ≈ 2.15 h into the run | The "shape draws dropped, store index not injective" counters (8 and 753) start from then, not from t = 0. |
| **R1, R2, R3** | second exactness control ("all inputs equal", which keeps a mean / variance / Welford accumulator exact where the plain integer control does not) | switched on by `touch ONESISH_CONTROL` at **2026-08-19 21:45:38** on all three lines | This is a **label**, never a filter, so it changes no drop count. But findings written before 21:45 carry only one exactness control, and findings after carry two. Do not compare the label sets across that boundary. |

## Counters known to be unreliable

* **`r2-compile`'s "distinct crash classes: 75"** — it groups by error-string
  signature, not root cause, and the signature keeps the random `/tmp` file
  name, the kernel name and register counts. Six ids for one ptxas crash and
  eleven for one register-pressure non-defect are measured, not suspected.
  Treat 75 as an upper bound on the number of distinct compiler defects.
* **`ttgir-broad`'s "report files: 59"** and its id range disagree — ids run to
  `HIT-0093` and 68 ids were reviewed. The id counter was reset once during
  setup and 8 reports were retracted in one block, so ids are not a count of
  reports.
* **`r2-oracle`'s "report files: 19"** counts the files sitting in `HITS/` at
  the moment of the refresh, not the number of reports the line ever wrote:
  ids `HIT-0001` … `HIT-0023` all exist and all were dispositioned.
* **The `FINAL_SUMMARY.md` of `r2-oracle` and `fuzz-tilelang-r2` is a
  per-process summary, not a campaign total.** Both lines were restarted during
  the run, and their last `FINAL_SUMMARY.md` was written by an earlier process.
  Their `STATUS.md` carries the cumulative counters and is what this document
  uses.

## Where two sources disagree

Every case found while building this table. No source was silently picked.

| what | source A | source B | used here |
|---|---|---|---|
| `ttir-broad` comparisons / pipelines / no-op skips | `STATUS.md`: 721,231 / 38,377 / 985,148 | `state/checkpoint.json`: 721,711 / 38,401 / 986,026 | STATUS (the checkpoint has a few more, written after the last STATUS refresh) |
| `ttgir-broad` tasks | `STATUS.md`: 1,064,123 | `checkpoint.json` and `FINAL_SUMMARY.md`: 1,064,291 | STATUS in the table; both given here |
| `fuzz-tilelang` (round 1) variants / pruned / same / diff | `STATUS.md`: 743,260 / 531,529 / 204,738 / 1,194 | `FINAL_SUMMARY.md` and `run/state.json`: 743,560 / 531,770 / 204,795 / 1,195 | STATUS in the table; both given here |
| `fuzz-tilelang` (round 1) corpus size | `STATUS.md`: 607 usable + 165 dropped = 772 | `run/state.json` `prescreen` map: 806 entries, 739 `ok`; `corpus/` directory: 720 entries | STATUS. The three do not reconcile and no file explains the gap. |
| `ttgir-broad` pre-screen breakdown | the dict `{ok 523, suspect-unstable 12, too-slow 1, unrunnable 31, nondeterministic 17, no-output 4, ub-race 24}` | sums to **612**, but the corpus is **607** | Both given. 5 kernels are counted under two states, presumably re-screened. |
| `ttgir-deep` corpus size | `README.md:95`: "818 cases" | `STATUS.md` and `state.json`: **926** | 926. The README is stale. |
| round-2 numbers in `/home/youngzt/fuzz/final-status-round2/` | the snapshot was copied at 12:45 | the live line directories were written at 12:47, after the last kernels finished | the live directories. Example: `r2-compile` points are 1,220,565 in the snapshot `STATUS.md`, 1,229,414 in the snapshot `checkpoint.json`, 1,131,246 in the snapshot `FINAL_SUMMARY.md`, and **1,231,940** in all three live files. |
| `r2-oracle` pre-screen | `FINAL_SUMMARY.md` (11:18): ok 481, too-slow 48 | `STATUS.md` (12:42): ok 478, too-slow 51 | STATUS. Three kernels moved from `ok` to `too-slow` in the last 1.5 h. |
| `r2-inductor` distinct pass sets | driver counter: 536,315 (full pipelines) | log scan: 145,101 (candidate deltas) | both, with the definitions spelled out above. Not a contradiction. |
| how many reports were dispositioned | `REJECTED.md` + the git-tracked keeps + the untracked parks: `r2-inductor` **200**, `r2-corpus` **165**, `r2-oracle` **24**, `ttgir-broad` **68** | the reviewer ledgers `/home/youngzt/fuzz/reviewer-triton/*.json`: `r2-inductor` **173**, `r2-corpus` **158**, `r2-oracle` **22**, `ttgir-broad` **68** | the files. `REJECTED.md` and the quarantine agree with each other exactly (738 = 738); the ledgers are behind by 27 / 7 / 2 ids. Merges into `reviewed.json` were still running when this was read, and `shard-corpus-a`, `shard-corpus-b` and `shard-inductor` had not been merged in yet. |

---

# What this does not cover

Read this before treating any "no difference found" as evidence of anything.

### 1. The sweep samples the pass space; it never exhausts it

Every line drew random pass sets. The fraction of the space covered is
astronomically small and the numbers say so:

* `ttgir-broad` tried **217,613** distinct pass sets out of a universe of 56
  passes. The number of subsets of 56 passes is 2⁵⁶ ≈ 7.2 × 10¹⁶, and the sweep
  also varies **order**, which makes the true space far larger still.
* `r2-corpus` tried **57,705** distinct candidate sets from a 78-flag universe;
  `r2-inductor` **145,101** from 56 passes; `ttgir-deep` **51,818** drop sets
  from 35 droppable passes (2³⁵ ≈ 3.4 × 10¹⁰); `fuzz-tilelang-r2` **146,320**
  (skip-set, config) variants from 30 passes and 27 config keys.
* What *is* exhaustive: the **single-pass** layer. `ttgir-deep` toggled all 35
  of its droppable passes alone, and round-1 `fuzz-tilelang` all 63 of its
  single toggles. Only that layer is complete.
* `ttir-broad` did not record its pass sets at all, so for that line even the
  sampling rate is unknown.

### 2. Tile sizes and grids are small, deliberately

PLAN-R2 §2a says "keep shapes small enough that a comparison stays cheap — the
point is variety, not size", and that is what was done. Measured:

* `r2-corpus` re-draws block-size constexprs from **16 … 1024** and scales the
  grid **0.5× … 3×** around the corpus kernel's own grid; the last shape draw
  recorded in its `STATUS.md` is `grid=(1, 1, 1)`.
* `r2-compile`'s size ladder tops out at 1,048,576 for plain sizes, but every
  block-like constexpr is forced to a power of two, and nothing is ever run.
* Round 1 used **one** shape per kernel — whatever the corpus source happened
  to use — so everything round 1 measured (721,231 draw-level comparisons on
  `ttir-broad`, 1,064,123 pairs on `ttgir-broad`, 640,916 on `ttgir-deep`,
  743,260 variants on TileLang) sits at a single point in shape space per
  kernel.
* Nothing in the campaign compiled a production-size GEMM or attention tile
  under a real launch grid.

### 3. Each pair sees a handful of input draws, so "no difference" is not a proof

The oracle is: run both arms and compare bits. That is a **test**, not a proof.
The sample size per compiled pair is:

* **20** draws for `ttir-broad`, `ttgir-broad`, `ttgir-deep`, both TileLang
  rounds and `r2-corpus`;
* **18** for `r2-inductor`;
* **8** per kernel per cycle for `r2-oracle` O1 and **6** for O2;
* **0** for `r2-compile` — it never runs anything, so it can only find crashes
  and invalid IR, never a wrong answer.

Twenty draws from a handful of fill modes over an input space of size 2³² per
fp32 element bounds nothing. On a related problem in this project, a
*deliberately wrong* recipe still produced byte-identical output on 91.7 % of
randomly drawn inputs. Surviving 20 draws is the normal case for a wrong
program, not evidence that it is right.

The strict lane is narrower still: `r2-oracle` could make an exact demand on
only **13,212** of its 71,680 O1 comparisons, on **139** of 607 kernels.

### 4. Blackwell paths were compile-only, and mostly not even that

The box is 8 × **H100 (sm_90)**. Nothing Blackwell-only can be executed.

* `r2-compile` compiled for **sm_80, sm_89 and sm_90a only** — 419,625 +
  406,078 + 406,237 points. It compiled **0** points for sm_100/sm_103, on
  purpose: the `ptxas` in this environment is version-skewed (13.1.80 against
  the pinned 13.3.33), so a Blackwell result there would not be trustworthy.
* The Blackwell-only passes (TMEM allocation and layouts, `tcgen05` lowering,
  CLC) are in the round-1 and round-2 TTGIR pass pools but are **weighted 0.25**
  against 3.0 for a bit-preserving pass, and they run on sm_90 input IR, so
  what was tested is the pass refusing or no-op-ing, not the Blackwell code
  path.
* Every `blackwell-only` arch label in a report is therefore a **static** claim
  read off the IR diff, never a runtime observation.

### 5. The corpus is what it is

* The shared Triton corpus is **607 kernels**: 184 TorchInductor-generated and
  423 hand-written (FlagGems / FLA / TritonBench / torchao). The lines that used
  it kept **474–523** after pre-screen (`ttir-broad` built its own 526-spec
  version of the same two sources and kept 466). It is not a sample of
  anything; it is what was to hand.
* `r2-inductor` broke that ceiling by generating kernels — 7,460 random torch
  programs → 10,136 harvested kernels — but they are **small random torch
  programs** (elementwise chains, reductions, broadcasts, views/permutes, the
  occasional matmul), not real models.
* `ttgir-deep`'s 926 "cases" are not corpus kernels at all. They are a
  parameter grid over **8 hand-written templates**. Its result says something
  about those 8 shapes of program and nothing about the other 599 corpus
  kernels.
* Kernels that are undefined behaviour were dropped on purpose and are **not**
  covered by any result here: across the campaign the pre-screens dropped
  atomics and scatter stores (`ub-race`: 6 / 24 / 17 / 66 / 20 / 53 / 53
  across the lines), non-deterministic kernels (8 / 17 / 21 / 393 / 18 / 2),
  and kernels that could not be run at all (`unrunnable`, `too-slow`,
  `no-output`, `load_error`). A data race is undefined behaviour, so the
  compiler is allowed to do anything; that is why they were dropped, but it
  does mean this campaign says nothing about them.
* Both rounds were stopped early: round 1 at **45.3–46.8 h** of 72 h, round 2
  at **19.7–20.1 h** of 72 h. Round 2 in particular ran only about 28 % of its
  budget.

### 6. Two more things nothing here covers

* **The default pipeline.** Round 1's oracle is "pass on versus pass off", and
  both arms run the default pipeline, so it is structurally blind to any bug
  that lives in the default pipeline. Only `r2-oracle` O1 (the interpreter
  oracle) can see one, and it made a strict demand on 139 kernels and 13,212
  comparisons.
* **Anything below the IR.** `r2-compile` finds ptxas crashes, but no line in
  this campaign can tell an IR-level bug from a codegen, register-allocation or
  hardware bug by itself; that judgement is made per report, in the report.
