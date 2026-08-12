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
- *Detailed split plan:* `tv/doc/m0-plan.md`.

### M1 — extend tile-smt to model (almost) all of Triton TTIR
Grow the tile-smt semantic model, driven by TTIR's needs, until it cleanly models
the vast majority of TTIR semantics.

**Two kernel corpora** drive it (they are complementary, not overlapping in role):
- `tv/benchmark/benchmark_kernels.py` — 636 **hand-written** kernels (FlagGems,
  FLA, TritonBench, torchao). Complex: attention, linear attention, quantized
  GEMM, nested loops. Source of the *hard* features → drives M1-complete.
- `tv/benchmark/inductor_kernels.py` — **TorchInductor-generated** kernels
  (`torch.compile`). Regular: fused pointwise + reduction, template-produced,
  and generatable in unlimited quantity. Source of *breadth* → drives M1-MVP.
  Also the highest-value bug target: it is the most-executed Triton code there
  is.

**Measured starting point** (AST scan of the 636-kernel corpus, 2026-08):
**15.7%** (100/636) are fully modelable today. The blockers are NOT what we
assumed — `tt.dot` blocks only 37 kernels and ranks 9th. The real blockers are
dtype casts (`.to()`, 266), **for loops** (197), `tl.where`/select (185),
data-dependent `if` (121), `maximum`/`minimum`/`abs` (100/60/49). Greedy unlock
order: data-dependent if (+45) → `where` (+24) → casts (+34) → for loop (+24) →
`dot` (+17), reaching ~47% cumulative.

**Sub-phases, defined by measurable coverage (not by a hand-picked kernel list):**
- **M1-MVP** — data-dependent `scf.if`, `select`, dtype casts, `maximum`/
  `minimum`/`abs`, the common `math.*`, and loops in `unroll` mode.
  *Done when:* **≥40%** of the hand-written corpus and **≥80%** of the inductor
  corpus are modelable.
- **M1-complete** — `tt.dot`, multi-dim reduce, `make_block_ptr`, and loops in
  `summarize` mode. *Done when:* **≥60%** of the hand-written corpus is
  modelable **and flash attention validates**.

**Explicitly out of scope: atomics** (`tt.atomic_rmw`, `tt.atomic_cas`,
`tl.atomic_add` — the 3rd-largest blocker at 24 kernels). Not because they are
hard, but because **the question is ill-posed**: atomics are non-deterministic
across program instances, and since FP addition is not associative, the result
itself is not deterministic — so "bit-exact equivalence" has no meaning for
them. Such kernels must report UNSUPPORTED, never a verdict.

### M2 — model tile-gpu-smt (TTGIR / GPU layer)
Start the GPU layer: layouts / `convert_layout`, shared memory, warp/lane, async
(TMA/mbarrier), warp specialization. Structured like M1 (**M2-MVP** →
**M2-complete**).

### M3 — go beyond Triton
Add a second frontend. **Next target: TileLang** (a non-MLIR frontend — proves
framework independence). Its own adapter drives the same tile-smt / tile-gpu-smt.
(Later targets, e.g. Pallas/Helion "≈ free", and accelerator hardware, come after.)

---

## Testing work (T) — dynamic differential testing (independent bug hunt)

An **independent, dynamic** line that does **not** use tile-smt. It is the M
line's implicit control / ground truth, and its main aim is to **brute-force find
real compiler bugs** (may also hit crash bugs).

**Method:** for every kernel in the benchmark kernels, toggle compilation passes
so the **only variable is whether a given pass is on**; compile the kernel with
vs without that pass, run **both on large amounts of randomly generated inputs**,
and check the outputs are **bit-identical**. (Dynamic → needs real kernel
execution / GPU.)
- **Exclude precision-trading passes**: some passes intentionally trade accuracy
  for performance — they would mismatch legitimately, so drop them.
- **Refinement (later):** beyond pure-random inputs, craft inputs that **trigger
  a given optimization's effect** (more likely to expose bugs). Start with random.

**Sub-lines (independent of each other):**
- **T1** = TTIR passes. **T2** = TTGIR passes.
- While M is still Triton-only, T1/T2 scope is **Triton-only**.
- Fully independent of the M/V lines — can run today.

**T helps M and V:** when T finds a real bug, **reconstruct the two IRs** (pass
on vs off) and feed them to **M's validator (tile-smt)** — does it also catch the
bug? That directly tests how strong/complete our semantic modeling is, and gives
V/M **ground-truth bug cases**.

(Note: M's own unit tests + eval gates keeping green is part of each M's
definition-of-done, not a T item — T is this dynamic differential line.)

## Validation experiments (V) — static equivalence via tile-smt

Exercises **tile-smt's own power** and is the real use of the tool to hunt
compiler bugs — but **static**: the SMT solver proves IR equivalence, with **no
tensor launch** (the key contrast with T's dynamic runs).

**Method:** like T, compare an IR with a pass on vs off — but **statically with
our SMT solver** (today's `permute_passes.py` is the seed of this line).

**Sub-lines (mirror T):**
- **V1** = TTIR-level static validation; starts once **M1** (TTIR modeling) is
  done. **V2** = TTGIR-level; needs **M2**. (V1↔T1↔TTIR, V2↔T2↔TTGIR — confirmed.)
- Precision sensitivity is set by the **FP encoding + axiom profile** (two knobs;
  design §"FP axiom profiles"), not by excluding passes. Two profiles drive V:
  - **Exact / bit-to-bit** (FPA, or Abstract reassoc-off) — proves a pass is
    bit-for-bit; a precision-trading pass correctly shows as NOT equivalent.
  - **Reassoc-allowed** (Abstract + associativity axioms, or Real) — proves a
    precision-trading pass did only *legal* FP reordering (tree reduce, FMA
    fusion).
  Run a suspect pass under both: exact-SAT + reassoc-UNSAT = benign perf reorder;
  reassoc-SAT = real bug. (Both to be added; code not started.)

---

## Notes
- Near-term north star (see goals): a complete Triton implementation that
  **finds & reproduces a real Triton compilation bug** — realistically needs
  M1-complete and/or M2 (bugs cluster in complex kernels & TTGIR passes).
- Long-term (awareness, not scheduled here): whole-kernel-launch semantics +
  functional correctness (kernel-vs-spec); own repo; cross-language equivalence.
