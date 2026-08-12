# KB — How Alive2 handles loops

> **This is a knowledge base entry, NOT a plan.** Nothing here is an adopted
> decision for `tv/`. It records how another translation validator works, so we
> can decide later with evidence instead of guesswork. Anything we actually adopt
> must be written into `roadmap.md` / `tile-smt-design.md` / `CLAUDE.md` as an
> explicit decision.

**Subject.** [Alive2](https://github.com/AliveToolkit/alive2) — a bounded
translation validator for LLVM IR (Lopes, Lee, Hur, Liu, Regehr; PLDI 2021,
"Alive2: **Bounded** Translation Validation for LLVM"). It is the closest
existing tool to `tv/`: same job (prove two IR functions equivalent with an SMT
solver), different IR (LLVM vs Triton TTIR).

**Provenance.** Everything below with a `file:line` was read directly from the
local clone at `/home/youngzt/tv/alive2` (HEAD `a68009c9`, 2026-07-23) and
re-verified by hand. The PLDI 2021 paper itself could **not** be read (403 from
this network), so "what the authors say" is inferred from the title, the code,
and `TODO.md` — treat it as unverified. Two items lean on arXiv preprints and
are marked.

---

## 1. The mechanism: static unrolling only

Loops are handled by exactly one thing — cloning the loop body a fixed number of
times:

```cpp
// ir/function.cpp
void Function::unroll(unsigned k) {
  if (k == 0)
    return;
  LoopAnalysis la(*this);
  ...
```
called exactly twice, at the end of `Transform::preprocess()`:
```cpp
// tools/transform.cpp:1979-1980
src.unroll(config::src_unroll_cnt);
tgt.unroll(config::tgt_unroll_cnt);
```

- **The bound is typed by hand.** Flags `-src-unroll` / `-tgt-unroll` (as
  `-tv-src-unroll` / `-tv-tgt-unroll` under the opt/clang plugin). Src and tgt
  can differ.
- **Default is 0** (`util/config.cpp`: `unsigned src_unroll_cnt = 0;
  unsigned tgt_unroll_cnt = 0;`), and `unroll(0)` is a hard no-op. Unchanged
  since 2020-10.
- **No trip-count analysis of any kind.** A repo-wide grep for
  `tripcount|trip_count|ScalarEvolution|getSmallConstant|invariant|fixpoint`
  over `ir/ tools/ util/ smt/ llvm_util/ tv/` returns **zero hits**. It never
  asks LLVM for a trip count. After ~7 years there is no invariant inference and
  no loop summarization, and `TODO.md` lists no plan to add either.
- Symbolic / unknown trip counts get **no special treatment** — they are
  truncated at whatever bound the user typed, silently.

## 2. The cut is in the encoder, not the unroller

What actually makes a loop terminate is separate from unrolling:

```cpp
// ir/state.cpp — any jump back to an already-executed block goes to #sink
auto dst = &dst0;
if (seen_bbs.count(dst)) {
  dst = &f.getSinkBB();
}
```
`#sink` is a real block that is never symbolically executed, so a path that
reaches it never reaches a `ret`. Because this lives in the encoder, a loop
degrades to "body encoded once, rest dropped" **even at bound 0** — it never
hangs or crashes.

## 3. There is NO CBMC-style unwinding assertion

A repo-wide grep for `unwind` hits only LLVM's unrelated `dead_on_unwind`
attribute. Over-bound paths are **assumed away**, not asserted about. The UB
query even subtracts them explicitly:

```cpp
// tools/transform.cpp
// avoid false-positives in refinement query 1 due to bounded unrolling
expr pre_old = pre;
pre &= !tgt_state.sinkDomain(true);
```

This is deliberate: without it, bounded unrolling makes the source look "more
defined" than the target and produces spurious failures. (Contrast CBMC, where
the unwinding *assertion* is opt-in via `--unwinding-assertions`; its default is
also an unwinding *assumption*.)

## 4. Reporting: this is Alive2's weakest point

- The **only** bound-related message fires when **no path at all** reaches a
  return:
  ```cpp
  // tools/transform.cpp
  if (!sink_src.isFalse() &&
      check_expr(axioms_expr && !sink_src, "return_src").isUnsat()) {
    errs.add("The source program doesn't reach a return instruction.\n"
             "Consider increasing the unroll factor if it has loops", false);
  ```
  The guard is `isUnsat()` on `!sink_src` — i.e. it needs **every** execution to
  sink. Partial truncation (the normal case for a symbolic trip count) is
  completely silent.
- A truncated run prints the **same string** as a full proof:
  `"Transformation seems to be correct!"` (`llvm_util/compare.cpp`). The unroll
  factor is not printed, not even in the debug dump.
- Bounded unrolling is **not** registered in Alive2's own "Approximations done:"
  channel (`State::doesApproximation` has 7 call sites; none is loop truncation).
- It *does* stay honest about **solver** failure: three verdicts CORRECT /
  UNSOUND / FAILED_TO_PROVE, and a timeout never becomes "correct".

*(Secondary source, medium confidence: arXiv 2503.19449v3 reports a case — test
`s481` — where a wrong transformation passed Alive2 because the bug needed ≥2
iterations and every such path sank. A verifier re-derived the mechanism from the
source rather than trusting the paper.)*

## 5. In practice, tiny bounds — and they still find bugs

- Every loop test names its own factor by hand. Histogram of `-src-unroll=`
  across `tests/`: **1×5, 2×29, 3×30, 4×8, 5×3, 7×1, 8×1, 32×2** — 75 of 80 are
  between 1 and 5; the max is 32.
- Rationale stated in the tests, e.g. `pr36437-bounded.srctgt.ll`: *"Bound the #
  of iterations to 2, because the problematic execution happens at the second
  iteration."*
- Despite that, `BugList.md` (125 entries) contains ~15 real **loop**
  optimization bugs found this way — LoopReroll, loop peeling, LoopUnroll,
  SimpleLoopUnswitch, LoopVectorize.

## 6. A second, different bounding style for loop-shaped libcalls

`strlen` / `memcmp` / `bcmp` use `LoopLikeFunctionApproximator` with a
hard-coded count (8, raised to 10) and **no flag**. It bounds by **assumption**:
```cpp
// ir/instr.cpp
bool is_last = i >= unroll_cnt - 1;
...
if (is_last) s.addPre(prefix().implies(!continue_i));
```
Difference that matters: a **sink** drops one *path* (other paths for the same
input are still checked); an **`addPre` assumption** removes the whole *input*
from the checked space. There is an escape hatch: for constant inputs it keeps
unrolling up to 512.

## 7. Unrolling is not the cheap option

- `Function::unroll` is ~170 lines of clone + jump-patch + phi repair, with a
  ~20-commit bug tail (nested-loop exit values, aggregates, multi-entry/multi-exit
  phis, dominator-tree recomputation, *"handle complex SSA with load/stores — not
  pretty, but works for now"*). 71 loop test files exist largely to pin those fixes.
- Loop detection is Tarjan-Havlak; the source says **"Irreducible loops are
  partially supported."**
- There is an off-by-one hack: the header is cloned once more so the last
  iteration can exit, *"Here we assume the header is an exit, as that's the
  common case."* So "k-th backedge → sink" is exact only for single-block loops.
- Unrolling **widens the memory encoding**: it runs before
  `calculateAndInitConstants`, so cloned `alloca`s raise `num_locals`, which
  raises `bits_for_bid` — the block-id field of *every* pointer in the query.
- *(Secondary, medium confidence: arXiv 2406.04693v1 reports 82/125 pairs
  inconclusive with plain Alive2 due to timeout/memory-out.)*

---

## 8. Possible implications for `tv/` — observations, not decisions

Listed so we don't lose them; each still needs its own decision.

1. **Separating "unroll" from "cut" may be worth copying.** The cut prevents
   false alarms from truncation, and makes every reported counterexample a path
   that really completed within the bound. Alive2 had to add the sink subtraction
   to the UB query by hand *after* getting spurious failures.
2. **Alive2's reporting gap is cheap to avoid.** It already computes
   `sinkDomain` and already asks the solver about it — it just asks "did *every*
   path sink?" instead of "did *any* path sink?". A three-way verdict
   (proved / proved-up-to-k / unknown) plus printing `k` would close it.
3. **Bounded unrolling with tiny bounds does find real bugs** — evidence that an
   unroll-first path can produce value before any summarization exists.
4. **But the cost profile differs for us.** Alive2's loop bodies are a few
   scalars; ours are tiles. Our blowup is `trip_count × tile_width`, and Triton
   trip counts are small (tens) while tile widths are not.
5. **Assume-style vs sink-style bounding are not interchangeable** — one drops a
   path, the other drops an input.
6. **Nobody has done summarization here.** After ~7 years Alive2 still has none,
   and no plan for one. That is a warning about cost, not proof that it is wrong
   — Alive2's loops (data-dependent exits, heap, irreducible CFGs) are much
   harder than `scf.for`.

## 9. Open questions this KB could not answer

- What the PLDI 2021 paper actually *says* about loops (paper unreadable: 403).
- Whether the authors ever considered and rejected invariants/summarization, and
  why (GitHub blocked, so issues/discussions unread). This is the single most
  useful missing piece.
- Why the cheap reporting fix was never made — is there a reason, e.g. the flag
  would fire on nearly every real loop and become noise?
