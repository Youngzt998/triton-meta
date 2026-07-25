# tv development roadmap — M / T / V numbering

**Stable numbering (use from 2026-07-24 onward in commits, docs, discussion):**
- **M<n>** — major (development) work.
- **T<n>** — testing work.
- **V<n>** — validation experiments.

These IDs are stable and reused long-term. This file is the numbered execution
plan; `tile-smt-goals.md` gives the *why* + success criteria, `tile-smt-design.md`
the interfaces. (This supersedes the old M0–M4 list that used to live in
tile-smt-goals.md.)

Starting point (pre-M0): today's **working, Triton-coupled** validator in
`tv/semantics/` (validates TTIR add/softmax; eval suite green).

---

## Major work (M)

### M0 — migrate existing results onto the new plan
Move today's Triton-coupled implementation onto the tile-smt architecture. **Two
sides:**
- **SMT side** — extract `tile-smt` (core) out of `tv/semantics/`: MLIR-free,
  own `DType`/`Shape`/`MemId`, owning Memory + AbstractFp + tile ops +
  equivalence (witness); **standalone build + unit tests with only Z3**.
- **Triton side** — turn the current op handlers + `State`/`Env` into a thin
  **Triton adapter** that walks TTIR and calls the tile-smt builder.
- *Done when:* the eval suite (`run_eval.py all`) is green through the adapter,
  and `tile-smt` has zero MLIR includes/symbols.

### M1 — extend tile-smt to model (almost) all of Triton TTIR
Grow the tile-smt semantic model, driven by TTIR's needs, until it cleanly models
the vast majority of TTIR semantics. **Two sub-phases:**
- **M1-MVP** — common/simple kernels (elementwise, softmax-class, basic reduce).
- **M1-complete** — complex kernels (e.g. **flash attention**): control flow
  (`scf.for`/`if`), `tt.dot`, multi-dim reduce, and whatever else FA needs.

### M2 — model tile-gpu-smt (TTGIR / GPU layer)
Start the GPU layer: layouts / `convert_layout`, shared memory, warp/lane, async
(TMA/mbarrier), warp specialization. Structured like M1 (**M2-MVP** →
**M2-complete**).

### M3 — go beyond Triton
Add a second frontend. **Next target: TileLang** (a non-MLIR frontend — proves
framework independence). Its own adapter drives the same tile-smt / tile-gpu-smt.
(Later targets, e.g. Pallas/Helion "≈ free", and accelerator hardware, come after.)

---

## Testing work (T)
`T<n>` = testing work: tile-smt unit tests (Z3-only), adapter tests, the eval
gates (pairs / inequal / compile-options), regression. Concrete `T<n>` items are
defined as the matching `M<n>` lands; each M must keep its tests green.
(Guiding rule: every migration step in M0 keeps `run_eval.py all` + unit tests
green.)

## Validation experiments (V)
`V<n>` = validation experiments: running the tool at scale to learn/prove
something — hunting real Triton miscompiles (the near-term north star), FP-mode
comparisons (a/b/c), solver-cost studies, later cross-language equivalence.
Concrete `V<n>` items TBD. (e.g. a `V<n>` = permutation campaign on complex
kernels to find/reproduce a real Triton bug.)

---

## Notes
- Near-term north star (see goals): a complete Triton implementation that
  **finds & reproduces a real Triton compilation bug** — realistically needs
  M1-complete and/or M2 (bugs cluster in complex kernels & TTGIR passes).
- Long-term (awareness, not scheduled here): whole-kernel-launch semantics +
  functional correctness (kernel-vs-spec); own repo; cross-language equivalence.
