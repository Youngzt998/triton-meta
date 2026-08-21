# Confirmed compiler bugs — index

41 confirmed bugs found by differential fuzzing of **Triton** (31) and
**TileLang** (10), over two campaigns.

**The convention that matters: a report is confirmed if and only if it is
tracked in git.** Untracked files in these directories are either unreviewed or
parked. 983 reports were written; 41 survived review, 738 were rejected, 20 are
parked, 200 were never reviewed. See `COVERAGE.md` for what the campaign
actually covered and `PARKED.md` for the undecided ones.

Rejected reports are **not** in this repo — they are quarantined at
`/home/youngzt/tv/bug-report-rejected/` with `REJECTED.md` giving every
disposition. That directory does not travel with a clone. `triton/dropped/` and
`tilelang/dropped/` keep six worked examples each so a reader can judge the
filtering without it.

---

## Part 1 — useful for the SMT equivalence-check work

### 1a. These two contradict the current design. Read them first.

| report | why it matters |
|---|---|
| [`triton/r2-inductor/HIT-0099`](triton/r2-inductor/HIT-0099.md) | A `tt.reduce` result is **not the same in every lane** its layout calls a copy. Under the model `tv/doc/tile-smt-design.md` sketches — reduce as a function of the tile, one value per output index — the two arms are **equal by construction**, so a checker built that way reports EQUIVALENT and is wrong. Seeing it needs layouts as `(register, lane, warp, block) -> tensor index` maps **and** copies allowed to hold different values. Three instances; it can present with no NaN, no integer output, and even as signed-zero-only. |
| [`triton/r2-oracle/HIT-0001`](triton/r2-oracle/HIT-0001.md) | `tl.cumsum` drops the cross-CTA carry at `num_ctas > 1`. **Both the TTIR and the TTGIR are correct** — `tt.scan` still means "prefix sum over the axis" and the CGA layout is legal. The fault is in the TTGIR→LLVM lowering, so a **TTIR↔TTGIR check — the boundary a tile-level validator attacks first — cannot see it.** |

### 1b. Value-semantics equivalence should catch these

| report | capability it needs |
|---|---|
| [`triton/ttir-broad/HIT-0007`](triton/ttir-broad/HIT-0007.md) | bit-exact FP with `inf`/`NaN` — invisible under the reassoc-allowed profile |
| [`triton/ttgir-broad/HIT-0071`](triton/ttgir-broad/HIT-0071.md) | same; this pair is the discriminator between the two FP axiom profiles |
| [`tilelang/HIT-0001`](tilelang/HIT-0001/) | pure integer bitvector + heap compare at a witness address — the easiest case there is |
| [`tilelang/HIT-0043`](tilelang/HIT-0043/) | a memory model that distinguishes "written" from "anything" (uninitialised-read check) |
| [`tilelang/r2/HIT-0008`](tilelang/r2/HIT-0008/) | same, on a register fragment |
| [`triton/ttgir-deep/HIT-0009`](triton/ttgir-deep/HIT-0009.md) | symbolic addressing + `ttg.warp_specialize` as composition of its regions (M2) |
| [`triton/r2-oracle/HIT-INTERP-BF16`](triton/r2-oracle/HIT-INTERP-BF16.md) | reframe from pass-vs-pass to **builder-vs-builder**, plus bit-exact FP; the model must keep a value's *storage* separate from its *meaning* |

### 1c. Not an equivalence query — a one-program obligation

Stronger than per-instance equivalence: these are universally quantified
preconditions, provable once over all shapes and layouts, in the same layout
vocabulary the GPU layer already wants.

| report | the obligation |
|---|---|
| [`triton/r2-compile/HIT-0016`](triton/r2-compile/HIT-0016.md) | `forall L, S: getAxisNumBlocks(L,S) * getNonAxisNumBlocks(L,S) == ...` |
| [`triton/r2-compile/HIT-0038`](triton/r2-compile/HIT-0038.md) | a rank-changing `tt.reshape` is free only if `flatten(L_src(b,w,l,r), S_src) == flatten(L_dst(b,w,l,r), S_dst)` |
| [`tilelang/r2/CODEGEN-0001`](tilelang/r2/CODEGEN-0001/) | the TMA shared destination address must be `0 mod 128` |

⚠️ Five more look like the same shape but their own reports say "not applicable,
a compile-time crash" and do **not** make the argument:
`r2-compile/HIT-0045`, `0048`, `0055`, `0063`, `0066`. Two of those (`0048`,
`0055`) are type well-formedness; two (`0045`, `0063`) are layout
well-formedness. Check before relying on it — this reading is not the reports'.

### 1d. Boundary markers — what is provably out of reach

Useful precisely because they are negative results.

| report | boundary |
|---|---|
| [`triton/ttgir-broad/HIT-0008`](triton/ttgir-broad/HIT-0008.md) | The IR pair **is** equivalent; ptxas miscompiles it. Corollary worth building: "checker says equal **and** the binaries differ" is evidence the fault is below the IR. |
| [`triton/r2-corpus/HIT-0024`](triton/r2-corpus/HIT-0024.md) | Same class, wrong values instead of wrong addresses. Cleanest exhibit is its occurrence `HIT-0159`: an exact integer count, so no FP licence exists at all. |
| [`triton/r2-compile/HIT-0019`](triton/r2-compile/HIT-0019.md) | Sharpest boundary in the set. `HIT-0016` is a false belief **about the IR**, which a solver can refute; this is a false belief the compiler holds **about itself** ("my worklist finishes in 10000 steps"). Nothing about the input program is wrong, so there is no obligation to state. |
| [`triton/r2-corpus/HIT-0017`](triton/r2-corpus/HIT-0017.md) | Needs a concurrency and memory-ordering model — a dropped cluster barrier. |
| [`tilelang/HIT-0040`](tilelang/HIT-0040/) | Same. Cheaper reformulation in the report: check the allocator's own contract, `forall i≠j: overlap_in_time => disjoint_in_space` — small linear arithmetic. |
| [`tilelang/r2/HIT-0043`](tilelang/r2/HIT-0043/) | Same; the stock pipeline randomly drops two shared-memory barriers. |

---

## Part 2 — the rest

### Triton compile-time crashes (`triton/r2-compile/`)

Two themes account for most of them.

**`num_ctas > 1` is untested** — `TritonGPUPlanCTAPass` is skipped entirely at
`num_ctas == 1`, has no lit tests, and its own file ends with a TODO asking for
them: `HIT-0001` (CGA layout with `num_ctas²` CTAs) · `HIT-0003` (`scf.while`) ·
`HIT-0005` (kernel with no store) · `HIT-0006` (two matmuls) · `HIT-0045` (two
stores of different rank) · `HIT-0049` (loop-carried pointer needing two
layouts) · `HIT-0063` (first one outside PlanCTA — `AccelerateMatmul.cpp:192`).

**fp8 is storage-only in practice but the language does not say so** —
`HIT-0002` · `HIT-0004` · `HIT-0007` · `HIT-0008` · `HIT-0009`. Every one ends
in invalid MLIR, an assert, or `abort()` where the right behaviour is a
Python-level message naming the dtype.

Others: `HIT-0047` and `HIT-0050` (two **distinct** ptxas 12.9 segfaults —
separated by crash address and backtrace, not by symptom) · `HIT-0048` and
`HIT-0055` (frontend type holes, each with a correct sibling branch a few lines
away) · `HIT-0066` (`AxisInfo` constant folder divides by zero on the host).

### Triton, other lines

[`triton/r2-corpus/HIT-0019`](triton/r2-corpus/HIT-0019.md) and
[`HIT-0037`](triton/r2-corpus/HIT-0037.md) — two more `PlanCTA` crash sites,
reached only with a pass order production never uses. Kept because `HIT-0037`'s
is an **unchecked out-of-bounds read**, silent under `NDEBUG`.

### TileLang

[`CODEGEN-0001`](tilelang/CODEGEN-0001/) (invalid CUDA from vectorized signed
floordiv) · [`HIT-0004`](tilelang/HIT-0004/) (vectorizing an RNG loop swaps in a
curand API with a different stream step — note its root-cause table was
corrected: only the fp64-uniform substitution actually differs) ·
[`HIT-0009`](tilelang/HIT-0009/) (**kept as the documented not-a-bug**: fast
math's `-ftz` turning `inf * denormal` into NaN) ·
[`r2/HIT-COMPILE-NONDET`](tilelang/r2/HIT-COMPILE-NONDET.md) (the compiler emits
different CUDA for the same input — up to 33 sources from 48 lowerings, at least
two causes; the `MergeSharedMemoryAllocations` one is a *(offset, name_hint)*
sort that ties when six buffers are all named `workspace`).

---

## Reading the `Arch:` line

Every report has one; all 41 checked. But the **strength** varies and the label
alone does not show it:

1. **Measured on several architectures** — only `r2-compile` can do this (it is
   compile-only): `HIT-0002`, `0007`, `0008`, `0055`, `0066` were built for
   sm_80, sm_89 and sm_90.
2. **Proved architecture-dependent** — `HIT-0047` (sm_89 only; the same PTX is
   clean at sm_90) and `HIT-0050` (sm_90a only).
3. **Structural** — "`num_ctas > 1` is rejected below sm_90, so this path is
   unreachable there". Reliable, but by definition rather than by measurement.
4. **Inferred** — "no wgmma/TMA/mbarrier/cluster in the IR, no target check in
   the pass". This covers most of the 18 "generic" labels, and **every one of
   them was only ever run on one H100.** Compile-time evidence, never a runtime
   proof.

Exactly one bug has its architecture-independence *proved*:
`r2-oracle/HIT-INTERP-BF16`, which reproduces with `CUDA_VISIBLE_DEVICES=""`.

Two reports say `unknown` (`r2-corpus/HIT-0019`, `HIT-0037`) — that is honesty,
not an omission.

One label is on the wrong axis: `ttgir-broad/HIT-0008` reads "any NVIDIA target
built with CUDA 12.9 ptxas", which is a **ptxas version** dependency, not an
architecture one. There is no field for that.

---

## Reproducing any of these

The IR, generated CUDA, minimal test cases and `repro.py` are committed with
each report, so **what** the bug is travels with a clone. **Running** the repro
does not: most repro commands begin with `source /home/youngzt/fuzz/<line>/env.sh`,
which is not in this repo, and only 13 of 41 reports pin the compiler commit
(`3277063a6` for the public Triton build; TileLang 0.1.12).

`env.sh` sets `TRITON_OPT`, `TRITON_CACHE_DIR`, `TRITON_ALWAYS_COMPILE=1`,
`MLIR_DISABLE_MULTITHREADING=1` and thread caps. On another machine, point
`$TRITON_OPT` at your own `triton-opt` and set `TRITON_ALWAYS_COMPILE=1`;
everything else is performance or isolation, not correctness.
