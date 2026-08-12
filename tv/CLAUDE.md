# CLAUDE.md — `tv/` Translation Validator

> **Scope.** Guidance for working on the translation validator under `tv/`.
> Repo-wide Triton guidance is in the root `CLAUDE.md`.
> **Language.** Chat/explanations in Chinese; keep **all code, comments, commit
> messages, and docs (including this file) in English**.
> **History.** The old, Triton-only version of this file is
> `tv/CLAUDE.legacy.md` (SUPERSEDED — do not follow it).

---

## 1. What `tv/` is (current state)

A working **SMT translation validator**: `triton-tv a.ttir b.ttir` proves two
MLIR functions semantically equivalent with Z3.
Exit codes: `0` = EQUIVALENT (UNSAT), `1` = NOT EQUIVALENT (SAT, prints a
counterexample), `2` = UNKNOWN. Today it validates **Triton TTIR** (add_kernel,
softmax) at realistic sizes.

**M0 is DONE.** The code is split into an MLIR-free core + per-language builders
(§6). Verified: core has zero MLIR, `libtile-smt.a` links only Z3, all unit tests
and the eval suite green.

**Current design facts (accurate — trust these over any older doc):**
- **Memory = Option B**: one byte-addressable Z3 array `Array(BV64,BV8)` per
  pointer argument (Triton args don't alias). `Memory::store` builds a **single
  `z3::lambda`** heap update; `checkEquivalence` compares at a **symbolic witness
  address** (`select(m1,w) != select(m2,w)`), NOT array extensionality — the
  latter returns `unknown`. (`semantics/Memory.*`, `semantics/Equivalence.*`)
- **FP = Abstract mode only**: each FP value is an opaque `BitVec` id; ops are
  uninterpreted Z3 functions with **only 5 axioms** (5 reserved consts distinct;
  add/mul/max commutative; neg involutive). Deliberately no associativity, no
  NaN propagation, no zero identities. `lt`/`le` throw. (`semantics/AbstractFp.*`)
- **Ops modeled**: `arith` constant/addi/subi/muli/andi/extsi/cmpi(10 preds)/
  addf/subf/mulf/divf/maxnumf; `math.exp`; `tt.` get_program_id/make_range/splat/
  addptr(scalar **and** tile)/load/store (both **require a mask**)/reduce
  (combine-region fold, **1-D single-operand only**).
- **Not modeled**: control flow (`scf.if/for/while` are `llvm_unreachable`),
  `tt.call` (rely on `-inline`), `tt.dot`, multi-dim reduce, `arith.cmpf`,
  `arith.select`, most casts, all `math.*` except `exp`. An unmodeled op is a
  **hard crash**, not a graceful "unsupported" verdict (to fix in M1).
- **Practical note**: proving EQUIVALENCE (UNSAT) is fast even at 1024 elems;
  proving NON-equivalence (SAT) is much harder — use **small tiles** for
  inequality cases.

## 2. Macro goal — abstract tensor semantics, mapped per language

**The whole pipeline = abstract tensor semantics (core) + per-language mapping
(builder).** Those two together are what turns a GPU kernel into SMT semantics:

- The **core** owns a *reasonably complete but abstract* set of **tensor
  operations** — elementwise map, reduce/scan, contraction, structural
  (iota/splat/broadcast/reshape/transpose), gather/scatter, masked windowed
  memory access, loop combinators, program identity. They are defined by their
  **semantics**, not by any language's op names, and depend on **no language**.
- Each **builder** has exactly one job: **map its own language's tensor
  operations onto that abstract set**. The mapping is not 1:1 — one core op
  serves many surface syntaxes (Triton `tt.load(ptrs,mask,other)`, TileLang
  `T.copy`, Pallas `pl.load(ref,idx)` are all "masked windowed read"), and one
  language op may expand into several core ops.
- **Completeness test:** adding a new language should need **no new core ops**.

⚠️ **Known gap:** today's `Context` API was lifted from the Triton handlers, so
parts of it are still Triton-shaped — e.g. `addPtr`/`splatPtr` are *pointer-level*
addressing, not abstract tensor ops (memref/TPU languages don't address by
pointer). Generalizing the op set along the lines above is M1 work. See
`tv/doc/tile-smt-design.md` §"Abstract tensor operation set".

**Near-term north star:** ship a complete implementation on **Triton** and use it
to **find & reproduce a real Triton compilation bug**.

The library split:
- **`tile-smt`** — hardware-neutral core (the abstract tensor semantics above).
- **`tile-gpu-smt`** — GPU/SIMT layer (M2): layouts/`convert_layout`, shared
  memory, warp/lane, async TMA/mbarrier, warp specialization.
- **(future) `tile-accel-smt`** — TPU/Mosaic + Trainium/NKI; **one layer covers
  both**.

Full goals + success criteria: `tv/doc/tile-smt-goals.md`. Interfaces:
`tv/doc/tile-smt-design.md`. Numbered plan: `tv/doc/roadmap.md`.

**Locked decisions:** Builder API (adapter calls the lib; no neutral IR);
incremental access model (linear byte-heap + pointer, abstract enough to swap for
memref/TPU later); `program_id` in the core; core holds no `mlir::Value` —
pointer provenance is an opaque `MemId`; **FP axiom profiles** (exact bit-to-bit
vs reassoc-allowed); **loop model** `--loop-model=unroll|summarize|auto` (§7).

## 3. Development rules

- 🔴 **NEVER change any Triton file to serve tv** — not even the repo root
  `CLAUDE.md`. tv's *only* hook into Triton is the single pre-existing line
  `add_subdirectory(tv)` in the root `CMakeLists.txt`. Everything tv adds lives
  under `tv/`. Gate: `git diff --name-only <base> HEAD -- ':(exclude)tv/'` must
  be empty. (Out-of-tree is infeasible: Triton's core is CMake OBJECT libraries
  with no exported package — `triton-tv` links ~305 raw `.o` files.)
- **This project does not run `pre-commit`** (the repo root CLAUDE.md asks for
  it; tv is exempt, and we do not edit that file to say so).
- Push only to branch **`tv`** or **`tv-trials`**.
- Any change to a design component **must update** the tests.
- **Chinese** for chat; **English** for code/comments/commits/docs.
- Commit policy: may **auto-commit new changes** (short one-line msg, no Claude
  author line); **ask before** history rewrites (squash/amend/rebase).

## 4. Build (offline recipe on this machine)

C++ changes require a rebuild. Full recipe + rationale: memory
`tv-env-and-remote-state`. **Do not run `pip install -e .`** while other sessions
are working — it relinks the shared `libtriton.so`. Use ninja:

```bash
cd /home/youngzt/tv/triton && source .venv/bin/activate
export TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF \
       LLVM_SYSPATH=/home/youngzt/triton-llvm/build-0729a74e \
       JSON_SYSPATH=/usr TRITON_BUILD_WITH_CLANG_LLD=1 MAX_JOBS=100
BD=build/cmake.linux-x86_64-cpython-3.12
ninja -C $BD tile-smt tile-smt-tests        # core only — fast, no MLIR
ninja -C $BD triton-tv tv-validator-tests   # full; the triton-tv link is slow
```

## 5. Testing

**Unit tests.** Two groups:
- **core (Z3-only, no MLIR)** — `ctest --test-dir $BD -R TileSmt` (5 tests:
  Types, AbstractFp, Memory, Context, Equivalence).
- **builder (needs MLIR)** — `ctest --test-dir $BD -R TestTritonTV` (Env, State).

**Routine testing — validator gates + optimization-permutation campaign:**

```bash
python tv/eval/run_eval.py all       # gates: pairs + inequal + compile-options, then timing
python tv/eval/permute_passes.py     # bug hunt on add_kernel (all 1/2/3-pass TTIR perms)
python tv/eval/permute_passes.py \
  --baseline tv/eval/compile-options/softmax_kernel/standard.ttir \
  --out tv/eval/compile-options/softmax_kernel     # bug hunt on softmax
```

- **pairs** — curated equivalent/non-equivalent pairs (verdict must match tag).
- **inequal** — genuinely non-equivalent pairs; must ALL be caught as NEQ
  (soundness). Use small tiles.
- **compile-options** — pass variants vs the unoptimized standard; must be EQUIV.
- **permute_passes.py** — every 1/2/3-pass TTIR-optimization permutation vs the
  reference. Any `NOT EQUIVALENT` = a potential miscompile (saves `BUG_neq.ttir`
  and stops — re-check soundness before filing).

Last runs: pairs 4/4, inequal 6/6, compile-options 9/9; add & softmax each 259
permutations all EQUIVALENT; no miscompile found yet.

⚠️ `tv/eval/compile-options/generate.py` picks the target from the **live GPU**
if one exists. On a different machine that silently changes which architecture's
IR you are validating — pin it explicitly when it matters.

## 6. Where things are

```
tv/
  semantics/     CORE `tile-smt` — MLIR-free, links ONLY z3 (namespace tile_smt)
                 Types · Value · AbstractFp · Memory · Context · Equivalence
    test/        Z3-only unit tests (SimpleTest.h harness)
  builder/       per-language builders — the ONLY place that includes MLIR
    mlir/        shared by all MLIR languages: DTypeOf · Env · State (walk +
                 dispatch + control-flow stubs) · ArithOps
    triton/      Triton-specific: TritonOps (tt.*)
  bin/           triton-tv.cpp — validator main
  test/validator/  builder-level C++ tests (need MLIR)
  eval/          pairs/ inequal/ compile-options/ solver-cost/ run_eval.py
                 permute_passes.py
  benchmark/     benchmark_kernels.py — 420+ collected @triton.jit kernels
  doc/           roadmap.md · tile-smt-goals.md · tile-smt-design.md ·
                 m0-plan.md · code-navigation.md · tensor-languages-survey.md
    kb/          knowledge base on external tools — REFERENCE, NOT plans
                 (alive2-loops.md). Nothing in kb/ is an adopted decision.
  paper/         noticable.md — scaling-issue log (e.g. the store-blowup fix)
```

**Start here to read the code:** `tv/doc/code-navigation.md` (layer map,
recommended reading order, key invariants, the 3-edit recipe for adding an op).

## 7. Known limitations / next work

**M1 is next** — extend the core to (almost) all of TTIR. Coverage today:
7 of ~48 `tt.*` ops, ~12 of ~30 `arith.*`, 1 of ~14 `math.*`, **0 of 7**
control-flow ops.

**Decided:**
- **Loops** — `--loop-model=unroll | summarize | auto`. `summarize` lifts a loop
  to a combinator over `k ∈ [0,N)`; its enabling analysis is **affine recurrence
  recognition** (turn `ptr += stride` into `base + k*stride`), decomposed by
  **SCC of the carried-value dependence graph**. `scf.while` is out of scope
  (measured: 2.4% of kernels, all irregular).
- **Generalize the core op set** away from its Triton shape (§2).

**Measured fact (not a decision):** solver blowup is `trip_count × tile_width`,
and today **both** are statically unfolded — `Context::reduce` folds
element-wise, `Memory::store` unfolds lanes.

**Open — proposals only, nothing decided:** how to report a bounded/truncated
result; unmodeled-op behavior (hard crash today vs an `UNSUPPORTED` verdict); a
solver `--timeout` (none inside the binary today); pointer aliasing (may-alias
vs today's no-alias assumption); UB policy; `program_id` range constraint;
`tt.dot` fidelity; reduce ordering; M1's target-kernel list.

**Long-term awareness:** modeling is **single program-instance**;
whole-**kernel-launch** semantics + functional correctness (does GEMM really
compute GEMM) is a long-term goal — don't design anything that blocks it.
Equivalence assumes **race-free**; a possible tie-in is the Triton Sanitizer for
cross-instance data-race detection (⚠️ unverified).
