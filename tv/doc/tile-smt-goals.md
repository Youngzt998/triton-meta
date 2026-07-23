# tile-smt — project goals (refined)

High-level goals/vision for the refactor. Interface details live in
`tv/doc/tile-smt-design.md`; cross-language evidence in
`tv/doc/tensor-languages-survey.md`. This is a **draft** — revise as we discuss.

## 1. What tile-smt is

A **standalone, pure SMT abstract semantic model** for tile/tensor kernel
programs, plus a **builder API**. It is independent of:
- any **source language** (Triton, TileLang, Pallas, …),
- any **compiler IR framework** — MLIR, TVM TensorIR, a custom IR, or a Python
  AST can all drive it,
- (as far as possible) any **hardware**.

A language models its semantics by writing a thin **adapter** that walks its own
IR and calls the tile-smt builder. tile-smt encodes that into Z3 and answers
semantic questions — primarily **equivalence** (translation validation).

## 2. Non-negotiables — what "MLIR-independent" means concretely

1. tile-smt source includes **no MLIR / Triton / TVM headers** — only Z3 + the
   C++ standard library.
2. tile-smt **builds and its unit tests run standalone**, with **only Z3** as a
   dependency (its own CMake target; tests do not need MLIR or Triton).
   *Checkable:* no `mlir::` symbols in tile-smt objects; it does not link MLIR.
3. Every builder parameter is a **neutral type** — `DType`, `Shape`, `MemId`, Z3
   handles, combine callbacks — **never** `mlir::Value/Type/Operation` or any
   framework type. (This is why pointer provenance is an opaque `MemId`, not an
   `mlir::Value`.)
4. All language/framework coupling lives in the **adapter**; for the Triton
   adapter, that is the only place that includes MLIR.

## 3. Layered architecture

- **tile-smt** (core): the model + builder + equivalence. Hardware-neutral.
- **tile-gpu-smt**: GPU/SIMT extensions (layouts/`convert_layout`, shared memory,
  warp/lane, async TMA/mbarrier, warp specialization).
- **tile-accel-smt** (future): non-GPU hardware (TPU/Mosaic, Trainium/NKI).
- **adapters** (one per language, living in that language's own tree): Triton
  adapter first; others later. An adapter is thin: IR walk → builder calls.

## 4. Capabilities tile-smt provides

- value/type model: `Scalar`/`Tensor`/`Ptr`; `DType`/`Shape`.
- tensor ops: elementwise map, reduce/scan (combine), dot/contract, structural
  (iota/splat/broadcast/reshape), `program_id`.
- memory: abstract address space + masked windowed load/store (default = linear
  byte-heap + pointer), functional `z3::lambda` update.
- FP semantics: Abstract now; Real/IntegerRange/FPA later — all language-neutral.
- equivalence: witness-address check over output memories → equivalent /
  not-equivalent (+counterexample) / unknown.
- control-flow merge helpers (if→ite, for→unroll); the adapter supplies bodies.

## 5. Verification goals (what we can ask it)

1. **Translation validation** (primary, now): the same kernel before vs after a
   compiler pass/pipeline is equivalent → find miscompiles.
2. **Cross-language equivalence** (enabled by language-neutrality; longer term):
   a Triton kernel vs a TileLang/Pallas kernel meant to compute the same thing
   both encode into the *same* tile-smt model, so they become comparable.
3. **Soundness self-checks**: it must catch genuine differences (the `inequal`
   gate), not just confirm equivalences.

## 6. Milestones (scope now vs later)

- **M0 (today):** Triton-coupled implementation in `tv/semantics/` (works: add,
  softmax; eval suite green).
- **M1:** extract **tile-smt** (core) — MLIR-free, standalone build + tests;
  Triton becomes an adapter that calls the builder; eval stays green. (migration
  steps: `tile-smt-design.md`)
- **M2:** **tile-gpu-smt** when TTGIR support lands.
- **M3:** a second adapter to *prove* independence — Helion / Pallas-GPU are
  ~free (they emit Triton IR); a **non-MLIR** frontend (TileLang/TVM) proves
  framework independence.
- **M4:** cross-language equivalence; **tile-accel-smt** (TPU/Trainium); more FP
  modes.

## 7. Success criteria (checkable)

- tile-smt has **zero MLIR includes/symbols** and links only Z3.
- tile-smt unit tests run **without** MLIR/Triton.
- The Triton adapter is the only MLIR-including part; the existing eval
  (add/softmax gates + permutation campaign) is **green** after the refactor.
- Adding a Triton-IR-emitting frontend (Helion) needs **no new tile-smt code**.

## 8. Non-goals (for now)

- Not a code generator and not an optimizer — only **semantics + equivalence**.
- Layout/placement/scheduling are **not** semantics; they are attributes the
  equivalence check ignores.
- Cross-hardware *numerics* are a separate concern (FP-mode question), not the
  access model.
- Closed/absent IRs (Mojo low-level IR, cuTile) are out of scope until openable.
