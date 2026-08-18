# Bug-report review criteria

**This file is the single source of truth for the reviewer agents.** They re-read
it at the start of every run, so editing this file changes their behaviour
immediately — no restart needed.

Scope: the reports under `tv/bug-report/triton/<line>/` and
`tv/bug-report/tilelang/`. They are produced by an automated fuzzing campaign
that compiles the same kernel twice, differing only in the compilation pass set,
runs both on identical inputs, and compares the output bit for bit.

A report records a **bit difference**. It does **not** claim to be a bug. Deciding
that is this review's job.

---

## 1. Reject (fake) — move out of the repo

Rejected reports are **moved**, never deleted outright, to
`/home/youngzt/tv/bug-report-rejected/<triton|tilelang>/` — outside the git repo,
so the repo stays clean but the decision can still be spot-checked. **Do not
renumber anything**; the remaining reports keep their original ids, and gaps in
the numbering are expected and fine.

Append one line per rejection to
`/home/youngzt/tv/bug-report-rejected/REJECTED.md`: id, which criterion below
fired, and one sentence of reason.

### Group A — not a real difference at all (measurement artifact)

| # | Reject when |
|---|---|
| A1 | Re-running the report's own repro command does not reproduce the difference. |
| A2 | The difference disappears when the GPU allocator is perturbed (heap shift) — an out-of-bounds or uninitialised read by the test harness, not the compiler. |
| A3 | The kernel is not deterministic under a *single* binary (same build, same input, different bits) — that is UB. |
| A4 | The kernel's store address is derived from loaded data, or it uses atomics — a write-write race, which is UB, so the compiler may legally do anything. |
| A5 | The reference and candidate IR are in fact identical — the test was vacuous. |

### Group B — a deliberate, performance-motivated numeric change

| # | Reject when |
|---|---|
| B1 | The culprit pass's **source** explicitly does one of: reassociate/reorder a reduction, fuse into an accumulator (e.g. dot+add), lower precision (tf32, bf16x3, fp8), or select a hardware instruction with different rounding — **and** the difference is consistent with that licence. |
| B2 | The consistency test for B1: the difference **disappears under exact-integer inputs** (integer values whose fp32 accumulator provably stays below 2^24, so no rounding occurs and evaluation order cannot matter). |
| B3 | Same for FP contraction (`mul+add → fma`): that is a licence held by the whole compiler, not by any one pass, so *any* pass that merely changes IR shape can trigger it. Same integer-input test applies. |
| B4 | The report has the same root cause (same culprit pass **and** same mechanism) as a report already kept. Keep one, reject the duplicates, and note the occurrence count on the kept one. |

## 2. Keep — never reject these

- The difference **survives** the exact-integer control ⇒ it is not reassociation,
  contraction or precision.
- An **integer or boolean** output differs — there is no floating-point excuse.
- **NaN or Inf appears on one side only.** There is exactly **one** exception, the
  fast-math one below. In particular, **a licensed reassociation that crosses an
  overflow boundary still counts — keep it.** If a reduction reorder makes an
  intermediate partial sum overflow to `±inf` so that a later `(+inf) + (−inf)`
  yields `NaN`, that is a **bug**, even though the reorder itself is licensed and
  B2's exact-integer test passes. **Owner ruling, do not re-litigate:** turning a
  finite or `±inf` result into `NaN` is a change of kind, not of precision, and
  `NaN` poisons everything downstream. `triton/ttgir-broad/HIT-0071` is the
  reference case. Reports whose root cause is the same reorder *without* the
  NaN/Inf outcome remain ordinary B1/B2 rejections — the NaN/Inf outcome is what
  makes the difference.

  *Fast-math exception* — the difference disappears once fast-math is removed: Fast math implies flush-to-zero (`-ftz=true`), and
  `inf × denormal → inf × 0 = NaN` manufactures a NaN out of finite inputs; that
  is the documented behaviour of the flag, not a bug. So: re-run with fast math
  off (TileLang: drop `tl.enable_fast_math`; Triton: the equivalent flag), and if
  the difference goes away, reject it under B1 instead of keeping it.
  **Exception, pinned by the project owner:** `tilelang/HIT-0009` **stays in the
  repo** as the documented representative of this class — it explains why the
  pattern is not a bug. Do not reject or move it, on this or any later pass, even
  though the rule above would otherwise catch it.
- The culprit pass has **no numeric licence anywhere in its source**, and the
  difference is not explained by downstream fma contraction either.
- **Compile failure, crash, or invalid generated code** (e.g. TileLang
  `CODEGEN-0001`). Different shape from a bit difference — keep it.

## 3. 🔴 Never a reason to reject, on its own

**"The difference is tiny" — few elements, few ULP, small magnitude.**

Magnitude alone is not evidence. A real bug can be small. Reject only when a
specific criterion above *explains* the difference. If nothing explains it, keep
it, even if it is one element at one ULP.

---

## 4. For every report that is kept: the deep analysis

Rewrite the report so it contains all six of these, then commit it (one commit
per report, message `tv bug-report: <line> <ID> <short title>`, explicit
`git add` of that report's paths only — never `-A`, never push).

1. **Root cause in the source.** Read the actual compiler code and say *where*
   it goes wrong — file and line, the transformation it performs, and why that
   is incorrect. Not "the pass mishandles X" but the specific step.
2. **Reproduction.** A command that works start to finish, copy-pasteable, with
   the exact input draw/seed that fails. Verify it by running it.
3. **Where it reproduces.** Which machines: H100/sm90 only, any NVIDIA GPU,
   any backend? Say what the judgement rests on (does the failing IR contain
   Hopper-only constructs — wgmma, TMA, mbarrier, cluster, warp specialization —
   or is it generic?).
4. **A plain-language explanation.** What the compiler got wrong, in ordinary
   words, understandable without knowing the pass. Short.
5. **Can an SMT solver model this?** Assume no such tool exists yet and none of
   its current limits apply — the question is whether the *semantics* at the root
   of this bug is capturable in principle by an SMT equivalence check. Decide
   between: a deterministic single-instance value-semantics difference (yes, in
   principle); needs whole-grid modelling; needs a concurrency/memory-ordering
   model; needs bit-exact FP; below the IR level and therefore out of reach of
   any IR-level checker; or expressible but computationally impractical. Justify
   it. A well-argued "no, and here is why" is more valuable than an optimistic
   yes.
6. **Confidence.** How sure are you this is a genuine compiler bug rather than
   something we have misunderstood, and what would settle the remaining doubt?

Be willing to conclude "I cannot determine the root cause" — say so explicitly
rather than inventing a mechanism.
