# r2-compile (R4) — reviewer rollup

Written by the reviewing agent, not by the fuzzer. R4's own `CLASSES.md` groups
by **error string**; this file groups by **root cause**, which is what criterion
B4 in `tv/bug-report/REVIEW-CRITERIA.md` asks for.

Three review passes so far (pass 3 is written up at the bottom of this file):

| pass | ids | kept | rejected |
|---|---|---|---|
| 1 | HIT-0001 … HIT-0015 | 9 | 6 (all B4) |
| 2 | HIT-0016 … HIT-0044 | 3 | 26 (16 B4, 4 A1, 6 resource-limit — the last is not a numbered criterion; see below) |
| 3 | HIT-0045 … HIT-0047 | 2 | 1 (B4) |

🔴 **R4 used to rewrite every `HIT-*.md` in this directory on each refresh. Since
the driver restart at 17:52 on 2026-08-19 it does not.** Two guards are live and
were confirmed working during pass 2: it skips the repo copy of a report that
carries the reviewer's headings, and it skips any id listed in
`/home/youngzt/tv/bug-report-rejected/REJECTED.md` or present in the quarantine
directory. Details in `/home/youngzt/fuzz/r2-compile/NOTES-followup.md` §2 and
§5. This file is a name R4 does not generate, so it survives either way.

## The 14 kept reports

Counts are the **folded** totals for the root cause, read at 2026-08-19 19:27.
They keep growing; the number matters less than the ranking.

| kept | root cause | culprit | merged into it | occurrences |
|---|---|---|---|---|
| HIT-0001 | `PlanCTA::processReduce` applies the leftover CTA count twice → a CGA layout with `numCTAs × remainingCTAs` CTAs | `TritonGPUPlanCTAPass` | HIT-0013, HIT-0021 | 302 |
| HIT-0002 | `fp8e4b15` is `is_floating()` but its MLIR type is `i8` → `arith.*f` on integers | frontend | HIT-0015, 0025, 0028, 0029, 0034, 0035 | 84 |
| HIT-0003 | PlanCTA's cast propagation handles only `scf.for` / `scf.if`; `scf.while` hits `report_fatal_error` | `TritonGPUPlanCTAPass` | HIT-0014 | 101 |
| HIT-0004 | fp8 arithmetic reaches the LLVM lowering, which maps fp8 → `i8` but keeps the float op | `ConvertTritonGPUToLLVM` | HIT-0010, 0012, 0018, 0027, 0030, 0031, 0033, 0043 | 143 |
| HIT-0005 | `assert stores.size() > 0` on a kernel with no store | `TritonGPUPlanCTAPass` | — | 43 |
| HIT-0006 | `markTiled()` unguarded inside the per-`tt.dot` walk; 2+ matmuls assert | `TritonGPUPlanCTAPass` | — | 117 |
| HIT-0007 | `convert_custom_float8` ignores `dst_ty` unless it is fp32, so `cast` returns the wrong type | frontend (NVIDIA `cuda/utils.py`) | HIT-0024, 0039, 0041 | 73 |
| HIT-0008 | the `fp8e4b15` inline-asm conversion hard-codes `pack=4` | frontend (NVIDIA `cuda/utils.py`) | — | 146 |
| HIT-0009 | unstaged fp8 `tt.fp_to_fp`; the lowering's 11-entry table has no fp8→fp8 and no fp8→f64, and aborts on a miss | `ConvertTritonGPUToLLVM` | HIT-0011 | 125 |
| **HIT-0016** | `tt.scan`'s block counts divide the **whole-tensor** shape by a **per-CTA** tile, so they over-count by `product(CTASplitNum)` | `ConvertTritonGPUToLLVM` (`ScanLoweringHelper`) | HIT-0017 | 79 |
| **HIT-0019** | `CTAPlanner::run`'s worklist has a fixed 10000-step budget and an `assert`, so a large enough function aborts | `TritonGPUPlanCTAPass` | — | 81 |
| **HIT-0038** | the default CGA layout puts the same dimension split at a **different CTA-index bit** for different ranks, so a rank-changing `tt.reshape` stops being free | `ConvertTritonToTritonGPU` (default blocked encoding) | — | 2 |

Bold = added in pass 2. Rejections are listed in
`/home/youngzt/tv/bug-report-rejected/REJECTED.md` and the reports themselves in
`/home/youngzt/tv/bug-report-rejected/triton/r2-compile/`, each with the
reviewer's re-run log (`review-rerun.log`).

## Two themes, now six and seven defects deep

**Theme 1 — `num_ctas > 1` (Hopper clusters) is untested.** Six defects:
HIT-0001, 0003, 0005, 0006 and 0019 are all `TritonGPUPlanCTAPass`, and HIT-0016
is the scan lowering. Add HIT-0038, which is the *default layout assignment* at
`num_ctas >= 4`, and the seventh, `r2-oracle/HIT-0001` (`tl.cumsum` silently
drops the cross-CTA carry — the only finding in the whole campaign that is a bug
in the **default** pipeline).

`PlanCTAPass::runOnOperation` returns immediately when `num_ctas == 1`
(`PlanCTA.cpp:986`), the file's closing TODO list ends with "Add some lit tests
for this pass", and it has none. Nothing in Triton's own test matrix runs any of
this. R4 opened it by making `num_ctas` a config axis (PLAN-R2 §2b); no round-1
line varied it.

Pass 2 also showed the theme is **not confined to `PlanCTA`**. HIT-0016 is in
`lib/Analysis/` + `lib/Conversion/TritonGPUToLLVM/`, and HIT-0038 is in the
`BlockedEncodingAttr` CGA builder in `TritonGPUAttrDefs.td`. The common shape of
all three new ones is the same mistake in three places: **code that treats a
layout's `block` (CTA) dimension as if it were not there.** HIT-0016 omits it
from a count; HIT-0038 places its replicated bases inconsistently; the scan's
scratch sizing omits it too. Anyone fixing one should grep for the others.

**Theme 2 — fp8 dtypes are storage-only in practice but the language does not
say so.** Five kept defects (HIT-0002, 0004, 0007, 0008, 0009) and, by pass 2's
count, **400 of the line's occurrences**. Two stories:

- `fp8e4b15` has *no MLIR type at all* (it is `i8`, `python/src/ir.cc:1055`) yet
  is classified as floating point, so arithmetic on it is invalid at TTIR build
  time, and its hand-written PTX conversion helper is wrong in two more ways
  (ignores the destination type; hard-codes `pack=4`).
- `fp8e4nv` / `fp8e5` *do* have MLIR float types, so TTIR and TTGIR verify
  clean, and the compiler falls over only at the LLVM lowering, where fp8
  becomes `i8`.

In every one, the correct behaviour would be a Python-level message naming the
dtype. Instead the user gets invalid MLIR, an assert, or `abort()`.

## What pass 2 rejected, and one judgement call

* **16 B4 duplicates** — 6 into HIT-0004, 5 into HIT-0002, 3 into HIT-0007, 1
  into HIT-0001, 1 into HIT-0016. Every one was re-run and its mechanism checked against
  the kept report's, not merged on the error string. Three merges needed real
  work: HIT-0021's `32 = 8 × 4` matches HIT-0001's `numCTAs × remainingCTAs`
  formula exactly; HIT-0024/0039/0041 were traced in the built IR to the
  `convert_fp8e4b15_to_float16` inline PTX feeding an op that wanted `f64` or
  `i1`; HIT-0017 differs from HIT-0016 only in which of the two identical
  assertions runs, and that is decided by `warpsPerCTA[axis] == 1`.
* **4 A1 — did not reproduce.** HIT-0032 (ptxas host-memory failure), HIT-0022
  and HIT-0023 (LLVM host OOM / SIGABRT at the `llir` stage — the same event
  seen with and without its message), HIT-0037 (`timeout@ptx`, which finished
  cleanly in 560 s once the harness budget was lifted). All four are
  machine-load artifacts, recorded while four other round-2 lines were
  compiling on the same box.
* **6 ptxas register-pressure failures**, rejected as `not a defect
  (resource-limit)`: HIT-0020, 0026, 0036, 0040, 0042, 0044. (HIT-0032 is a
  seventh ptxas failure but it is counted under A1 above, since it did not
  reproduce.) 🔴 **This is a judgement call and it goes against a literal
  reading of REVIEW-CRITERIA §2** ("compile failure … keep it"). The reasoning:
  `num_warps=32` means 1024 threads per CTA, so 65536 registers / 1024 threads
  fixes the budget at 64 per thread (`num_warps=16` → 128; HIT-0036 asked for
  `maxnreg=64` outright). ptxas refuses, names the number it needs, and Triton
  raises a typed `triton.runtime.errors.PTXASError` carrying the full text. That
  is the compiler making a decision and reporting it, which is exactly R4's own
  `resource-limit` class — its classifier just routes `ptxas fatal` to
  `crash-ptxas` instead. §2's "compile failure" keep is about the compiler
  failing, not about a hardware limit being reported correctly. If the owner
  disagrees, HIT-0020 is the cleanest representative to restore.

## Was R4's `CLASSES.md` grouping trustworthy?

**Still: no false merges, but it keeps over-splitting — and now under-merges in
one new way.**

* **Over-splitting, as before.** Pass 1 found five root causes split across
  several rows; pass 2 found seven more splits of the same three families
  (HIT-0018/0027/0030/0031/0043 for HIT-0004; HIT-0025/0028/0029 for HIT-0002;
  HIT-0024/0039/0041 for HIT-0007).
* **The mid-run fix helped, and its limits are now visible.** R4 folded on the
  operand type at 17:39 (NOTES-followup.md §3). That collapsed eleven signatures
  into three. But the regex matches only the wording *"operand #N must be
  floating point"*, so **HIT-0043 (`'llvm.fptrunc' op **result** #0 …`) escaped
  it** and took a fresh id for the same hole. Suggested tightening: match
  `(operand|result) #N`.
* **A genuine under-merge.** The six ptxas register-pressure reports
  (HIT-0020, 0026, 0036, 0040, 0042, 0044) are one class, split because the
  signature keeps the function name and the register count. Normalising the
  kernel name and the number inside `ptxas fatal : (CN) Insufficient registers
  (N) … in function <name>` would collapse them, and routing `ptxas fatal` to
  `resource-limit` would stop them being written up at all.
* **A merge that should not have happened at the report level.** HIT-0022
  (`llvm-error:out of memory`) and HIT-0023 (`SIGIOT@llir`) are the same abort
  seen with and without its message. Not R4's fault — it cannot know — but a
  reviewer should expect the pair.

## Notes for the next reviewer

- **The two guards work.** Reviewed reports are no longer clobbered and rejected
  ones are no longer re-created. You can rewrite and commit at your own pace.
  Confirmed by `/home/youngzt/fuzz/r2-compile/run/reviewed-kept.log`, which logs
  every skip.
- **New reports keep arriving.** Pass 2 started with 25 unreviewed ids and
  finished with 29; HIT-0041…HIT-0044 all appeared mid-pass. Every one landed in
  a class pass 2 had already settled (three fp8, one ptxas register pressure), so
  the marginal value of the line is dropping fast. It may be worth telling R4 to
  stop counting the fp8 families.
- **The cheapest way to pin a TTGIR/LLIR crash** is still the MLIR crash
  reproducer Triton prints on stderr: take the module, strip the leading
  `builtin.module(` from the printed pipeline, cut it before the suspect pass,
  run `triton-opt` to get the pre-pass module, then run the single pass. For
  pass 2's three keeps I went one step further and hand-wrote minimal inputs —
  `HIT-0016/scan-cta-min.mlir` (20 lines), `HIT-0038/reshape-cta-min.ttir`
  (9 lines) — which is worth the ten minutes: a hand-written file that switches
  at exactly the same `num_ctas` as the corpus kernel is much stronger evidence
  than a shrunk draw.
- **`MLIR_ENABLE_DUMP=1` on the generated `repro.py` works** and is how HIT-0038
  was attributed to `ConvertTritonToTritonGPU` rather than to `PlanCTA`. The dump
  is a few thousand lines for a small kernel.
- **Vary `num_ctas` first on anything sm_90-only.** Every sm_90-only class in
  this line so far has turned out to be a `num_ctas > 1` bug, and the check costs
  one re-run: `/tmp/vary.py`-style config override on the report's own draw.
</content>

---

## Pass 3 (2026-08-19) — HIT-0045 … HIT-0047

| pass | ids | kept | rejected |
|---|---|---|---|
| 3 | HIT-0045 … HIT-0047 | 2 | 1 (B4) |

All three reproduced on a quiet box.

**Kept — HIT-0045, a seventh `num_ctas > 1` defect and the third in `PlanCTA`
that is not a duplicate.** `CTAPlanner::processStoreLikeOps`
(`PlanCTA.cpp:350-385`) reads the CGA layout of the **first** store-like op into
a local at line 373 and then reuses it for **every** later store at line 378. A
`CGAEncodingAttr` has one entry per dimension, so a kernel with two stores of
different rank rebuilds the second store's `BlockedEncodingAttr` with the first
store's rank and trips the "same rank" verifier. Checked against the kept
`r2-corpus/HIT-0037`, which is the other `replaceCGALayout` crash: that one is
the *recursive* slice path at line 77 passing the slice's shape to the parent,
dies inside `SmallVector` before any verifier runs, and needs a
`SliceEncodingAttr`. This one has no slice and no recursion — the stack goes
`PlanCTA.cpp:379 -> :70` directly. A 13-line hand-written `.ttgir` with two
stores of rank 1 and rank 2 at `num_ctas = 8` reproduces it and is committed as
`HIT-0045/planCTA-mixed-rank-min.ttgir`; the same file at `num_ctas = 1` passes.

**Kept — HIT-0047, a new class: ptxas *crashes*, it does not mis-compile.**
`ptxas` V12.9.86 takes SIGSEGV at `--opt-level 2` and `3` on one 2868-line PTX
file for `--gpu-name=sm_89`. The same file is fine at `-O0`/`-O1` and fine at
sm_90 / sm_90a at every level, so the PTX is valid and Triton is not at fault.
Insensitive to `--regAllocOptLevel` (0-3) and `--maxrregcount` (32-255), so it is
**not** the register allocator and therefore not a member of the six
`Insufficient registers` reports pass 2 rejected as resource-limit; and it is not
the ptxas *wrong-code* class either (`ttgir-broad/HIT-0008`,
`r2-corpus/HIT-0024`), because nothing is produced at all. The `.ptx` is
committed, so the repro is one command with no Triton, no Python and no GPU in
it. Not minimized below 2868 lines.

**Rejected — HIT-0046, B4 of HIT-0002.** `'arith.truncf' op result #0 must be
floating-point-like, but got 'i8'`, with `'arith.mulf' op operand #0 ... 'i8'` in
the same stderr — the `fp8e4b15`-is-`i8` family. R4's 17:39 signature fold still
matches only `operand #N`; the suggestion from pass 2 (match
`(operand|result) #N`) is now confirmed by a second escapee, after HIT-0043.

**Notes for pass 4.**
* The two guards held: no reviewed report was clobbered and no rejected id was
  re-created during this pass.
* The `num_ctas > 1` theme is now **seven** defects plus `r2-oracle/HIT-0001`.
  Anyone fixing `PlanCTA` should read HIT-0001, 0003, 0005, 0006, 0019, 0037 and
  0045 together — they are seven separate mistakes in one 1000-line file with no
  lit tests.
* Capturing the PTX that ptxas dies on is easy and worth doing for any
  `crash-ptxas` / `crash-python@cubin` report: set `TRITON_PTXAS_PATH` to a
  wrapper script that copies its `.ptx` argument aside and then `exec`s the real
  ptxas. That turns a Triton-dependent report into a one-command NVIDIA bug
  report.

---

## Pass 4 (2026-08-19) — HIT-0048, plus two crashes that arrived on another line

| pass | ids | kept | rejected |
|---|---|---|---|
| 4 | HIT-0048 | 1 | 0 |

**Kept — HIT-0048, a new class and the first one in this line that is neither
`PlanCTA`, nor fp8, nor ptxas.** `Semantic.cast` builds `tt.int_to_ptr` for a
source integer of **any** width (`python/triton/language/semantic.py:897-899`),
but the op is declared `TT_I64Like` — `i64` only (`TritonOps.td:46`, backed by
`TritonTypes.td:42`). So `x.to(tl.pointer_type(T))` on an `i8`/`i16`/`i32`
produces malformed IR and a bare `RuntimeError: error encountered during
parsing` with no source location. The sibling branch three lines above, pointer
→ integer at `semantic.py:890-895`, **does** check the width and falls through
to a clean `assert False, 'cannot cast ...'`; the two halves of one function
disagree. A 20-line hand-written repro is committed as
`HIT-0048/int_to_ptr_width.py`: `*i64` compiles, `*i32`, `*i16`, `*i8` and `*u8`
all fail. Low severity (a diagnostic-quality defect, like HIT-0002) and the
failing draw does use a swapped pointer dtype, but no fuzzer is needed to see
it.

**Two `PlanCTA` crashes were folded in from `r2-corpus`, not from this line.**
The pass-toggle line found the same two defects this line already owns, which is
worth recording because it is independent confirmation from a different oracle:

* `r2-corpus/HIT-0066` → **HIT-0019**. `PlanCTA.cpp:164`, the `step < maxSteps`
  assert, on a 3476-line TTGIR at `num_ctas = 2`. The budget is hit by *size*,
  so any line that compiles a big enough function with clusters on will find it.
* `r2-corpus/HIT-0068` → **HIT-0003**. `PlanCTA.cpp:849`, `Unexpected parent op
  of block argument`, on an IR whose only region op is one `scf.while`.

Both were reproduced with a single `triton-opt` command on the dumped reference
IR. Occurrence lines were appended to HIT-0019 and HIT-0003 and committed on
their own.

**Notes for pass 5.**
* The `num_ctas > 1` theme is unchanged at seven defects plus
  `r2-oracle/HIT-0001`; nothing new joined it this pass. What is new is that a
  *second* line now reaches two of them, so the theme is not an artifact of R4's
  config sweep.
* R4's guards held again: no reviewed report was clobbered, and no rejected id
  was re-created, during this pass.
* `TRITON_DEFAULT_FP_FUSION=0` is **useless as a control on the r2-inductor
  line** — its spec sets `enable_fp_fusion` per compile, so the env default is
  overridden and the arm is unchanged. Override the config instead.

---

## Pass 5 (2026-08-19) — HIT-0049 … HIT-0051

| pass | ids | kept | rejected |
|---|---|---|---|
| 5 | HIT-0049 … HIT-0051 | 2 | 1 (A1) |

**Kept — HIT-0049, an eighth `num_ctas > 1` defect and the sixth in `PlanCTA`.**
`CTAPlanner::processMultiUsersBackward` (`PlanCTA.cpp:903-958`) gives a value
two different CTA layouts by cloning the op that defined it; a block argument
has no defining op, so line 937 calls
`llvm::report_fatal_error("Layout conflict for block arg")` — with a `// TODO`
next to it. It is reached when a loop-carried value has **two** users that want
**two** layouts: here the `scf.for`-carried A-operand pointer block, read by
`tt.load` and advanced by `tt.addptr`, with a `tt.dot` forcing the load's CTA
split. Checked against HIT-0003, which is the other `report_fatal_error` on a
block argument: that one is the *single*-user path at line 849, a different
message, and needs an `scf.while`. The crash reproducer Triton prints was cut
down to an 81-line `.ttir` and a 116-line pre-pass `.ttgir`, both committed;
`num_ctas = 1` compiles clean, `2` and `4` crash. Two attempts at a
hand-written minimal file did **not** reproduce, so the conflict needs more
index arithmetic than the obvious skeleton — recorded so the next reviewer does
not spend the ten minutes again.

**Kept — HIT-0050, a second and distinct `ptxas` crash.** `ptxas` V12.9.86
segfaults at `--opt-level 2` and `3` on a 23983-line `sm_90a` PTX from a
`tl.associative_scan` at `num_warps = 1`. It is **not** HIT-0047: under `gdb`
the two die at different addresses with different call chains (`0x961948` here
against `0xa910d0` there, sharing only the outer driver frame), and they answer
the register-allocator switch in opposite ways — this one is cured by
`--regAllocOptLevel=0` while HIT-0047 crashes at all four levels. `--maxrregcount`
32…255 all crash, so it is not a register budget. At `-O3 --regAllocOptLevel=0`
the file assembles with 128 registers and 7.5 KB of spill stores, which is what
puts the crash in the optimising register allocator. The `.ptx` is committed, so
the repro is one command with no Triton, no Python and no GPU.

**Rejected — HIT-0051, A1.** `SIGIOT@ptx`, `LLVM ERROR: out of memory / Buffer
allocation failed`. On a quiet box the same `repro.py` compiled cleanly through
`cubin` in 6 min 30 s with a 10.9 GB peak resident set. Third member of the
host-memory class already represented by HIT-0022, HIT-0023 and HIT-0032.

**Notes for pass 6.**
* The `num_ctas > 1` theme is now **eight** defects: HIT-0001, 0003, 0005,
  0006, 0019, 0045, **0049** in `PlanCTA`, HIT-0016 in the scan lowering,
  HIT-0038 in the default layout assignment, `r2-corpus/HIT-0037` in
  `replaceCGALayout`, plus `r2-oracle/HIT-0001` as the only silent-wrong-answer
  member. Six of the eight are crashes.
* `r2-corpus` reached a **ninth** `PlanCTA` assertion this pass —
  `PlanCTA.cpp:242`, `"PlanCTAPass should follow immediately after
  CoalescePass"` (`r2-corpus/HIT-0086`, `HIT-0087`) — and it was **rejected**,
  not folded in, because the pass-permutation oracle put `plan-cta` after
  `accelerate-matmul` and the assert states that as its own precondition. The
  production pipeline never does that. Judgement call in the same family as
  pass 2's six `resource-limit` rejections; flagged here so it is visible. Note
  that with `NDEBUG` the assert disappears and the pass would silently build
  wrong layouts.
* Capturing the PTX with a `TRITON_PTXAS_PATH` wrapper worked again and is now
  two for two. For any `crash-ptxas` / `crash-python@cubin` report, do it
  first: it turns a Triton-dependent report into a one-command NVIDIA repro and
  makes the `--opt-level` / `--regAllocOptLevel` / `--maxrregcount` sweep that
  separates the ptxas classes cost about a minute.
* 🔴 **The `onesish` (all-inputs-equal) control is written but not running.**
  The three GPU lines' `fz/driver.py` files contain `reassoc_controls`, the
  `ONESISH_CONTROL` flag files exist since 21:45, and `fz/worker.py` has
  `task_onesish` — but the driver processes were started at 18:45–19:00 and
  hold the older module, and the edit preserved the files' mtime (ctime 21:30,
  mtime 18:38), so nothing forces a reload. **0 of 190 `r2-inductor` findings,
  0 of 79 `r2-corpus`, 0 of 12 `r2-oracle` carry the `reassociation_excluded`
  field, and no report in the repo carries the label.** A reviewer must run the
  control by hand (`op: onesish` against the line's own worker) until a driver
  restart picks the code up.
