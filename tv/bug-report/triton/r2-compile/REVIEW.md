# r2-compile (R4) — reviewer rollup

Written by the reviewing agent, not by the fuzzer. R4's own `CLASSES.md` groups
by **error string**; this file groups by **root cause**, which is what criterion
B4 in `tv/bug-report/REVIEW-CRITERIA.md` asks for.

Two review passes so far:

| pass | ids | kept | rejected |
|---|---|---|---|
| 1 | HIT-0001 … HIT-0015 | 9 | 6 (all B4) |
| 2 | HIT-0016 … HIT-0044 | 3 | 26 (18 B4, 4 A1, 7 resource-limit — the last is not a numbered criterion; see below) |

🔴 **R4 used to rewrite every `HIT-*.md` in this directory on each refresh. Since
the driver restart at 17:52 on 2026-08-19 it does not.** Two guards are live and
were confirmed working during pass 2: it skips the repo copy of a report that
carries the reviewer's headings, and it skips any id listed in
`/home/youngzt/tv/bug-report-rejected/REJECTED.md` or present in the quarantine
directory. Details in `/home/youngzt/fuzz/r2-compile/NOTES-followup.md` §2 and
§5. This file is a name R4 does not generate, so it survives either way.

## The 12 kept reports

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

* **18 B4 duplicates.** Every one was re-run and its mechanism checked against
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
* **7 ptxas register-pressure failures**, rejected as `not a defect
  (resource-limit)`: HIT-0020, 0026, 0036, 0040, 0042, 0044 (and HIT-0032, which
  is also A1). 🔴 **This is a judgement call and it goes against a literal
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
* **A genuine under-merge.** The seven ptxas register-pressure reports
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
