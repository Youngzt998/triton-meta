# CLAUDE.md — `tv/` Translation Validator

> **Scope.** Guidance for working on the translation validator under `tv/`.
> Repo-wide Triton guidance is in the root `CLAUDE.md`.
> **Language.** Chat/explanations in Chinese; keep **all code, comments, commit
> messages, and docs (including this file) in English**.
> **History.** The old, Triton-only version of this file is
> `tv/CLAUDE.legacy.md` (SUPERSEDED — do not follow it; some of its claims are
> stale). The refactor direction is in `tv/doc/tensor-languages-survey.md`.

---

## 1. What `tv/` is (current state)

A working **SMT translation validator**: `triton-tv a.ttir b.ttir` proves two
MLIR functions semantically equivalent with Z3.
Exit codes: `0` = EQUIVALENT (UNSAT), `1` = NOT EQUIVALENT (SAT, prints a
counterexample), `2` = UNKNOWN. Today it validates **Triton TTIR** (add_kernel,
softmax) at realistic sizes.

**Current design facts (accurate — trust these over the legacy file):**
- **Memory = Option B**: one byte-addressable Z3 array `Array(BV64,BV8)` per
  pointer argument (Triton args don't alias). `Memory::store` builds a **single
  `z3::lambda`** heap update; `checkEquivalence` compares at a **symbolic witness
  address** (`select(m1,w) != select(m2,w)`), NOT array extensionality.
  (`tv/semantics/Memory.{h,cpp}`, `tv/semantics/State.cpp`)
- **FP = Abstract mode only**: each FP value is an opaque `BitVec` id; ops are
  uninterpreted Z3 functions + minimal axioms (add/mul/max commutativity,
  neg-involution, distinct reserved consts). FP encoding is designed as a
  **pluggable mode** — (a) Abstract [only one implemented], (b) Real, (c) FPA,
  (d) int-approx/interval later; first focus a/b/c (see
  `tv/doc/tile-smt-design.md` §"FP encoding modes"). (`tv/semantics/AbstractFp.{h,cpp}`)
- **Engine = `State`** (`env` + per-arg `ptrMems`), pure
  `interpretOp`/`interpretBlock` walking a `tt.func`. Ops modeled: `arith`/`math`
  elementwise (incl. `divf`/`maxnumf`/`math.exp`), `tt.make_range`/`splat`/
  `addptr` (tile **and** scalar)/`load`/`store` (masked), `tt.reduce`
  (combine-region fold). **No control flow yet** (`scf.if/for/while` are
  `llvm_unreachable`), no `tt.call` (rely on `-inline`), no `tt.dot`, no
  multi-dim reduce.
- **Unit tests**: header-only harness `tv/test/validator/SimpleTest.h` (no
  GoogleTest).
- **Practical note**: proving EQUIVALENCE (UNSAT) is fast even at 1024 elems;
  proving NON-equivalence (SAT) is much harder — use **small tiles** for
  inequality cases.

## 2. Macro goal — decouple tensor semantics (the direction we're moving to)

Today `tv/semantics/` is coupled to Triton. The goal is to factor the SMT
modeling into a **pure, MLIR-independent SMT semantic model** that builds with
**only Z3** (no MLIR/Triton/TVM headers), driven by any language through a
builder API. Full goals + success criteria: `tv/doc/tile-smt-goals.md`.

**Near-term north star:** first ship a complete impl on **Triton** and use it to
**find & reproduce a real Triton compilation bug**; design for extension, but
Triton is the first language.

The libraries and Triton-as-client:
- **`tile-smt`** — hardware-neutral core: logical tensor values; `map` /
  `reduce(axis)` / `contract(K)`; **affine windowed access + optional predicate**;
  memory as an address space; correctness = final-memory equality. **No
  pointers/warps/layouts baked in.**
- **`tile-gpu-smt`** — GPU/SIMT layer: pointer+mask lowering, register layouts
  / `convert_layout`, shared memory, warp/lane, async (TMA/mbarrier), warp
  specialization.
- **(future) `tile-accel-smt`** — non-GPU hardware (TPU/Mosaic, Trainium/NKI):
  DMA+semaphores, scratchpad placement, systolic staging.

A thin per-language **adapter** walks that language's IR and calls the lib's
builder API to encode + check equivalence. Triton's own **TTIR ≈ generic /
TTGIR ≈ GPU** split is the intended library cut; cross-language evidence (TileLang,
Pallas/Mosaic, IREE Linalg, Hidet, Helion, NKI) is in
`tv/doc/tensor-languages-survey.md`.

**Status:** survey done; **interface draft written** —
`tv/doc/tile-smt-design.md`. Locked decisions: **Builder API** (adapter calls
the lib; no neutral IR); **incremental access model** (core keeps the linear
byte-heap + pointer `Memory`, abstract enough to later swap for memref/TPU);
**`program_id` lives in the core**; core holds no `mlir::Value` — pointer
provenance is an opaque `MemId` the adapter maps. Implementation NOT started;
`tv/semantics/` stays the current Triton-coupled impl until migration Step 1.

## 3. Development rules

- All `tv/` work stays **inside `tv/`**; push only to branch **`tv`** or
  **`tv-trials`**.
- Any change to a design component **must update** the tests in
  `tv/test/validator/`.
- **Chinese** for chat; **English** for code/comments/commits/docs.
- Commit policy: may **auto-commit new changes** (short one-line msg, no Claude
  author line); **ask before** history rewrites (squash/amend/rebase).

## 4. Build (offline recipe on this machine)

C++ changes require a rebuild. Full recipe + rationale: memory
`tv-env-and-remote-state`. Essentials:

```bash
cd /home/youngzt/tv/triton && source .venv/bin/activate
export TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF \
       LLVM_SYSPATH=/home/youngzt/triton-llvm/build-0729a74e \
       JSON_SYSPATH=/usr TRITON_BUILD_WITH_CLANG_LLD=1 MAX_JOBS=100
pip install -e . --no-build-isolation          # configure the CMake build
ninja -C build/cmake.linux-x86_64-cpython-3.12 triton-tv tv-validator-tests
```

## 5. Testing

**Unit tests** (C++): build `tv-validator-tests`, then run the exes under
`build/.../tv/test/validator/` or `.venv/bin/ctest --test-dir <build> -R TestTritonTV`.

**Routine testing — validator gates + optimization-permutation campaign** (drives
the built `triton-tv`; the SMT solver is the verifier):

```bash
python tv/eval/run_eval.py all       # gates: pairs + inequal + compile-options, then timing
python tv/eval/permute_passes.py     # bug hunt on add_kernel (all 1/2/3-pass TTIR perms)
python tv/eval/permute_passes.py \
  --baseline tv/eval/compile-options/softmax_kernel/standard.ttir \
  --out tv/eval/compile-options/softmax_kernel     # bug hunt on softmax
```

- **pairs** — curated equivalent/non-equivalent pairs (verdict must match tag).
- **inequal** — genuinely non-equivalent pairs; must ALL be caught as NEQ
  (soundness). Use small tiles (NEQ is a SAT search, cheap only when small).
- **compile-options** — pass variants vs the unoptimized standard; must be EQUIV.
- **permute_passes.py** — applies every 1/2/3-pass TTIR-optimization permutation
  to an unoptimized kernel and validates each vs the reference. `EQUIVALENT` for
  all = healthy; any `NOT EQUIVALENT` = a potential Triton miscompile (it saves
  `BUG_neq.ttir` and stops — re-check soundness before filing).

Last runs: add & softmax each 259 permutations all EQUIVALENT; no miscompile
found.

## 6. Where things are

- `tv/semantics/` — `Memory`, `AbstractFp`, `Env`, `State` + `mlir/` op handlers
  (the Triton-coupled implementation to be refactored into the libs).
- `tv/eval/` — evaluation suite: `pairs/`, `inequal/`, `compile-options/`,
  `solver-cost/`, `run_eval.py`, `permute_passes.py`.
- `tv/test/validator/` — C++ unit tests (`SimpleTest.h` harness).
- `tv/doc/` — `plan.md`, `ttir.md`, `ttgir.md`, **`tensor-languages-survey.md`**.
- `tv/paper/noticable.md` — scaling-issue log (e.g. the store-blowup fix).
- `tv/CLAUDE.legacy.md` — old Triton-only guidance (SUPERSEDED; historical).

## 7. Known limitations / next work

- No control flow (`scf.if/for/while`), no `tt.call` (rely on `-inline`), no
  `tt.dot`, no multi-dim reduce; only Abstract FP mode.
- Non-equivalence (SAT) is slow at large sizes.
- The `tile-smt` / `tile-gpu-smt` refactor is **not started** — design
  discussion pending. Keep new modeling code factorable along the generic-vs-GPU
  line so the split stays cheap.
- (Long-term awareness) modeling is **single program-instance** today;
  whole-**kernel-launch** semantics + functional correctness (kernel-vs-spec:
  does GEMM really compute GEMM, FA really FA) is a long-term goal, not
  near-term. Don't design anything that blocks it.
