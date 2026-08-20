# r2-compile (R4) — reviewer rollup

Written by the reviewing agent, not by the fuzzer. R4's own `CLASSES.md` groups
by **error string**; this file groups by **root cause**, which is what criterion
B4 in `tv/bug-report/REVIEW-CRITERIA.md` asks for.

🔴 **R4 rewrites every `HIT-*.md` in this directory on each refresh.** The
reviewed text of each kept report lives in git history (one commit per report);
the working-tree copy may be overwritten by the fuzzer at any moment, and
`git status` will then show it as modified. This file is a name R4 does not
generate, so it survives. When R4 finishes, restore the reviewed reports from
their commits.

## Reviewed in this pass: HIT-0001 … HIT-0015

All 15 reproduce. HIT-0016 and later arrived while the review was running and
are left for the next pass.

**9 kept, 6 rejected — every rejection is B4 (duplicate of a kept class).**
Nothing was rejected as fake: REVIEW-CRITERIA §2 makes a compile failure, crash,
or invalid generated code a keep, and the Group A / Group B criteria are written
about bit differences and do not apply to a crash.

| kept | root cause | culprit | merged into it | occurrences |
|---|---|---|---|---|
| HIT-0001 | `PlanCTA::processReduce` applies the leftover CTA count twice → a CGA layout with `num_ctas²` CTAs | `TritonGPUPlanCTAPass` | HIT-0013 | 23 |
| HIT-0002 | `fp8e4b15` is `is_floating()` but its MLIR type is `i8` → `arith.addf` / `arith.cmpf` on integers | frontend | HIT-0015 | 6 |
| HIT-0003 | PlanCTA's cast propagation handles only `scf.for` / `scf.if`; `scf.while` hits `report_fatal_error` | `TritonGPUPlanCTAPass` | HIT-0014 | 14 |
| HIT-0004 | fp8 arithmetic reaches the LLVM lowering, which maps fp8 → `i8` but keeps the float op | `ConvertTritonGPUToLLVM` | HIT-0010, HIT-0012 | 16 |
| HIT-0005 | `assert stores.size() > 0` on a kernel with no store | `TritonGPUPlanCTAPass` | — | 6 |
| HIT-0006 | `markTiled()` unguarded inside the per-`tt.dot` walk; 2+ matmuls assert | `TritonGPUPlanCTAPass` | — | 12 |
| HIT-0007 | `convert_custom_float8` ignores `dst_ty` unless it is fp32, so `cast` returns the wrong type | frontend (NVIDIA `cuda/utils.py`) | — | 6 |
| HIT-0008 | the `fp8e4b15` inline-asm conversion hard-codes `pack=4` | frontend (NVIDIA `cuda/utils.py`) | — | 18 |
| HIT-0009 | unstaged fp8 `tt.fp_to_fp`; the lowering's 11-entry table has no fp8→fp8 and no fp8→f64, and aborts on a miss | `ConvertTritonGPUToLLVM` | HIT-0011 | 13 |

Rejections are listed in `/home/youngzt/tv/bug-report-rejected/REJECTED.md` and
the reports themselves are in
`/home/youngzt/tv/bug-report-rejected/triton/r2-compile/`.

## Two themes, not nine unrelated bugs

**Theme 1 — `num_ctas > 1` (Hopper clusters) is untested.** HIT-0001, 0003,
0005 and 0006 are all `TritonGPUPlanCTAPass`, which
`PlanCTAPass::runOnOperation` skips entirely when `num_ctas == 1`
(`PlanCTA.cpp:986`). Four independent defects in one pass, found within minutes,
because nothing in Triton's own test matrix ever runs it. The file's closing
TODO list ends with "Add some lit tests for this pass", and it has none. R4
opened this by making `num_ctas` a config axis (PLAN-R2 §2b); no round-1 line
varied it.

**Theme 2 — fp8 dtypes are storage-only in practice but the language does not
say so.** HIT-0002, 0004, 0007, 0008 and 0009 are all fp8. Two separate stories:

- `fp8e4b15` has *no MLIR type at all* (it is `i8`, `python/src/ir.cc:1055`) yet
  is classified as floating point, so arithmetic on it is invalid at TTIR build
  time, and its hand-written PTX conversion helper is wrong in two more ways
  (ignores the destination type; hard-codes `pack=4`).
- `fp8e4nv` / `fp8e5` *do* have MLIR float types, so TTIR and TTGIR verify
  clean, and the compiler falls over only at the LLVM lowering, where fp8
  becomes `i8`.

In every one of the five, the correct behaviour would be a Python-level message
naming the dtype. Instead the user gets invalid MLIR, an assert, or `abort()`.

## Was R4's `CLASSES.md` grouping trustworthy?

**Partly.** It never merged two different root causes — no false merge was found.
But it split one root cause across several rows five times, because its
signature is the compiler's error string and one defect produces a different
string for every op that happens to trip it. The clearest case is HIT-0004:
`llvm.fcmp`, `llvm.fmul` and `packLLElements` are three rows for one hole, and
while this review was running R4 filed a fourth (`HIT-0018`, `llvm.fsub`) for the
same hole. That row will keep splitting once per arithmetic op.

**Suggestion for R4, if it is ever changed:** when the failing stage is `llir`
and the message names an `llvm.f*` op with a non-float operand, fold on the
*operand type* rather than on the op name.

## Notes for the next reviewer

- The generated `repro.py` was briefly invalid Python (bare JSON `null` / `false`
  inside a dict literal). R4 fixed it mid-run to `json.loads(r"""...""")`. If it
  comes back, run it under a wrapper that sets
  `builtins.null = None; true = True; false = False`.
- To pin the culprit pass for a TTGIR crash, use the MLIR crash reproducer Triton
  already prints: take the module from stderr, strip the leading
  `builtin.module(` from the printed pipeline, cut the pipeline just before the
  suspect pass, run `triton-opt` to get the pre-pass module, then run the single
  pass on it. That is how the four PlanCTA reports were pinned, and it produced
  the small `.mlir` artifacts committed with them.
