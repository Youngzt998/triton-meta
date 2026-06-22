# bitequiv — Project Design Doc

> **This is the design doc for the whole `bitequiv` project** (bitwise equivalence &
> constraint-aware autotuning), written for humans. It is the single reference for:
> the problem and motivation, the architecture, the major sources of inequivalence,
> the data-layout theory (the thread↔data map and how it is encoded in TTGIR and
> PTX), the algorithm that decides whether two layouts/kernels are equivalent at
> each IR level, and the public interface.
>
> **When asked to extend `bitequiv`, follow this doc — and keep it in sync** when the
> design, algorithm, or interface changes. Terse machine-readable state lives in
> `bitequiv/PROGRESS.md`; the project guide/conventions live in `bitequiv/CLAUDE.md`.

Status: M1 (2026-06-15), restructured as the whole-project doc (2026-06-22). Built:
an MLIR-native **TTGIR reduction-equivalence checker** over distributed encodings
(`toLinearLayout`), single- and multi-operand `tt.reduce`, plus the **PTX/FMA
backstop** (§4.6). In flight / planned: M2 reduction layout optimization, M3 GEMM/MMA
accumulation-order equivalence. Audience: team + XFN.

**TL;DR.** Two kernels are *bitwise equivalent* if they produce the same output down
to every bit; because FP add/mul is non-associative, this means "the FP operations
happen in the same order." An autotuning config is both a *performance point* and a
*numerics point* — it compiles to a particular reduction/accumulation order, hence to
particular bits. `bitequiv` makes the Triton autotuner **constraint-aware**: it
statically reads the compiled IR, computes a per-config order *signature*, and prunes
configs that are not bitwise-equivalent to a chosen reference — recovering tuning
freedom that is otherwise thrown away by freezing layouts and disabling tuning. The
core machinery is Triton's `LinearLayout` (the thread↔data map), checked at **TTGIR**
(where layout first becomes explicit) with a **PTX backstop** for FMA contraction.

---

## 1. Background & problem

### 1.1 What "bitwise equivalence" means

Two kernels are **bitwise-equivalent** if, for the same input, they produce the same
output down to every bit. The hard part is floating point: FP addition and
multiplication are **non-associative** —

```
(a + b) + c   ≠   a + (b + c)        # in IEEE-754, due to rounding
```

so equivalence reduces to a structural property: **the FP ops must happen in the same
order.** Concretely, the *shape of the reduction/accumulation tree* — which value is
combined with which, and in what sequence — must match.

A crucial framing: **"different bits ≠ incorrect."** A different summation order
yields a different *valid* answer. "Correct" in this project means *matches a chosen
reference bit-for-bit* (determinism with respect to a reference), not "closest to the
true real-number result." The only genuine "bugs" are cases like a tensor-memory load
that corrupts values — those are separate from order-induced differences.

### 1.2 Why it matters

- **Skip the GPU ladder.** Customers validate model changes by re-running large
  evaluation ladders; bitwise-exact numerics let them skip thousands-of-GPU,
  week-long re-runs.
- **RL, batch-invariant inference, LLM training** all need reproducible numerics
  across batch sizes, hardware, and recompiles.
- **Real production signal.** The `STABLE_REDUCTION` workaround in layer_norm
  (D104785121) — a manual sequential reduction — was added because an ordinary
  reduction-order change caused a **5.24% NE (numerical error) gap**. That is the
  cost of *not* controlling order.
- **The current price.** Today equivalence is bought by **freezing layouts and
  disabling autotuning**, which leaves performance on the table. The goal of
  `bitequiv` is to *prove/enforce* equivalence and **recover the tuning freedom**.

### 1.3 The core tension

An autotuning **config** (`BLOCK_SIZE`, `num_warps`, layout, num_stages, …) is
simultaneously:

- a **performance point** — it determines occupancy, vectorization, pipelining; and
- a **numerics point** — it determines the layout, hence the reduction tree, hence
  the exact bits.

So "stay bitwise-equivalent while tuning" converts the unconstrained
`argmax(perf)` into a **constrained maximization over a safe set** of configs that
all reproduce one chosen reference order.

### 1.4 Where the bits get decided (the pipeline)

```
Python → TTIR (layout-free) → TTGIR (layout assigned) → LLVM IR → PTX → (ptxas) → SASS
```

The thread↔data map — and therefore the reduction tree — first becomes **explicit at
TTGIR**. TTIR is layout-free (so it cannot decide bits on its own); below PTX is the
`ptxas` black box (which can still contract/reorder). This is why `bitequiv` checks
primarily at **TTGIR**, with PTX as a targeted backstop (§5.1).

---

## 2. High-level design

### 2.1 The four verbs

1. **Prune** configs that are non-equivalent to the reference (and genuinely buggy
   ones) out of the tuning search.
2. **Enforce / enlarge** the safe set by carrying an *order-intent constraint*
   (e.g. an ordered reduction) so more configs become equivalent.
3. **Optimize** runtime *within* the safe set (M2: pick the best layout given the
   constraint).
4. **Verify** that the result still matches the reference bit-for-bit.

### 2.2 The design change

Today, equivalence = "freeze the layout." The `bitequiv` direction is to **carry an
order-intent constraint** (not a frozen layout) through TTGIR→PTX, so compiler passes
are free to optimize codegen *given* the constraint, paired with an **independent
verifier** and a **PTX-flag backstop** (ptxas lives outside MLIR, so it needs flag-
level control like `--fmad`). The constraint is the contract; the layout is free to
vary as long as it satisfies the contract.

### 2.3 The autotuner socket

There is a **single** hook in Triton core:

```python
ir_config_prune(config, asm, metadata, reference) -> bool   # True = KEEP
```

registered via `prune_configs_by`, and run **after** a `warmup=True` compile (compile-
only, **no GPU launch**), reusing the already-compiled artifact (`asm["ttir"|"ttgir"|
"llir"|"ptx"]` + `metadata`). Equivalence checkers are **injected** from `bitequiv`,
so Triton core never imports `bitequiv` (`python/triton/runtime/autotuner.py`).

**The goal of the feature, in use** — turn on equivalence pruning by registering one
predicate; the autotuner keeps only configs whose reduction order matches the
reference (the first config by default):

```python
import triton, triton.language as tl
from bitequiv.equivalence import reduction_equivalence_prune

prune = reduction_equivalence_prune(level="ttgir")   # keep a handle for introspection

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 4096}, num_warps=nw) for nw in (2, 4, 8)],
    key=["N"],
    prune_configs_by={"ir_config_prune": prune},
)
@triton.jit
def sum_kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0))

# after a tuning run, introspect what happened:
prune.classes   # {signature: [Config, ...]}  — the equivalence classes seen
prune.pruned    # {Config: "not-equivalent-to-reference"}  — what was dropped
```

Without `bitequiv` the autotuner picks the fastest of `nw ∈ {2,4,8}` — silently
landing on whatever bits that config happens to produce. With the predicate, configs
that would change the reduction order are pruned, so the winner is both fast *and*
bit-identical to the reference.

### 2.4 Static, not runtime

Equivalence is **proven from the compiled IR**, never by launching the kernel and
comparing outputs. (The earlier runtime correctness hook was removed on 2026-06-10 —
output comparison is out of scope.) This makes the check cheap enough to run per
config inside the autotuner.

### 2.5 The reference model

Equivalence is always **relative to a reference**: "match this config's order."
The reference defaults to the first config in the tuning set, or can be an **external
anchor** not in the set — a pre-compiled golden kernel, an `asm` dict, raw IR text,
or (future) cuBLAS.

### 2.6 Milestone roadmap

| Milestone | Focus | Status |
|-----------|-------|--------|
| **Starter** | Autotuner `ir_config_prune` socket; IR/PTX pruning examples; onboarding tutorials for numerics-modifying passes | Done |
| **M1** | Reduction equivalence **detect + enforce**: static TTGIR checker + autotuner integration; PTX/FMA backstop | Done |
| **M2** | Reduction **layout optimization** pass: pick the best layout *given* the ordered-reduction constraint; cross-config experiment framework + NCU profiling | Planned |
| **M3** | **GEMM/MMA** equivalence: model MMA/wgmma/tcgen05 accumulation order + precision; persist an MMA constraint; match **cuBLAS** on ~5 shapes (H100 + B200) | Planned |
| **M4–M7** | Stretch: TC-instruction opt · PyTorch/`torch.compile` fusion · AMD (MFMA) · GEMM+LayerNorm fusion | Stretch |

### 2.7 The guardrail (non-negotiable)

**Correctness gates performance — always.** Soundness is non-negotiable: a relation
must **never** declare two configs equivalent when their bits could differ. When
unsure, be **conservative** (over-split, never optimistic). Re-run the equivalence
tests after any perf- or codegen-affecting change. This matters especially for AI-
assisted work, where an agent optimizing for speed may silently undo an ordering
constraint and "succeed" while breaking correctness.

---

## 3. Major sources of kernel inequivalence (in scope)

For each: what it is, why it changes bits, what IR level it lives at, and how the
project handles it.

### 3.1 Reduction order (primary)

A reduction distributes `S` elements across threads and combines them in three
phases (`lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`):

1. **Within-thread** — each thread folds the elements it holds.
2. **Within-warp** — a butterfly (`shfl.sync.bfly.b32`) shuffle across lanes.
3. **Cross-warp** — warp leaders write partials to shared memory; one warp reduces
   them.

The order is set by two things: **(a) the layout** — which lane/warp holds which
element — and **(b) the `reduction_ordering` attribute**:

- `unordered`: within-thread **left fold** in index order; warp shuffle **count-down**
  offsets (`16,8,4,2,1`).
- `inner_tree`: within-thread **balanced** pairwise tree; warp shuffle **count-up**
  offsets (`1,2,…`). `inner_tree` is **layout-invariant** by construction (one fixed
  tree over the original element indices), so all `inner_tree` configs with the same
  axis extent reduce identically.

This is the dominant in-scope source, fully modeled by the TTGIR checker (§4.6).

### 3.2 FMA contraction

Whether a `mul` feeding a reduction **fuses** with the following `add` into one
rounded `fma` — or stays two separately-rounded ops — is decided **below TTGIR**, in
the LLVM NVPTX backend / `ptxas` (gated by `enable_fp_fusion` / `--fmad`):

```
fma.rn.f32 r, a, b, c      ⇔   round(a*b + c)              # one rounding
mul.rn.f32 t, a, b ; add.rn.f32 r, t, c  ⇔  round(round(a*b) + c)   # two roundings
```

Two configs can compile to **byte-identical TTGIR yet bit-different PTX**. This is
exactly the gap the **PTX backstop** closes (§4.6). `DotOpToLLVM/FMA.cpp`.

### 3.3 MMA / tensor-core precision

MMA/wgmma instructions accumulate with mode-dependent precision: **TF32 vs IEEE** vs
the **TF32x3 / BF16x3** decompositions, and the accumulator precision itself. Each
mode rounds differently → different bits (reproducible *within* a mode). The K-axis
accumulation **order** has no `tt.reduce` to parse and is the **M3** target; until
then it is covered by a conservative guard (§4.6). `F32DotTC.cpp`, NVIDIA `WGMMA.cpp`.

### 3.4 Out of scope / nondeterministic (excluded by construction)

Not modeled; stated here so they are not silently assumed equivalent:

- **Multi-CTA atomic merges** (`MultiCTAReduction.cpp`) — partitioning across CTAs
  changes the reduction tree and explicitly breaks bitwise reproducibility.
- **Float `atomic_rmw` contention** — the result depends on runtime execution order.
- **Fast vs precise `tl.math`** (`sqrt`/`exp`/`div` approximations) — a per-call
  choice, not an order property.
- **Compiler-internal nondeterminism** — e.g. the historical `DenseMap`-iteration
  partition-assignment bug; addressed in the compiler, not by this checker.

---

## 4. Data layout — the thread↔data map and its equivalence

This is the technical heart: general theory → how the map is encoded in TTGIR and PTX
→ the algorithm that decides equivalence at each IR level.

### 4.1 What a layout is

A **layout** is the map from **hardware coordinates** `(register, lane, warp, block)`
to **logical tensor coordinates** `(dim0, dim1, …)` — i.e. "which thread's which
register holds which tensor element." Naively this is an `O(S)` lookup table; in
practice the map is **structured** (affine / mixed-radix), so only its *generators*
are stored and coordinates are computed by arithmetic.

### 4.2 The mixed-radix intuition (`#blocked`)

For a `#blocked` layout, along the reduce axis `a`, an element index `i` decomposes in
mixed radix (innermost-first), with radices read off the encoding:

```
slot  =  i % c               c = sizePerThread[a]    — contiguous elems per thread (registers)
lane  = (i // c) % t          t = threadsPerWarp[a]   — lanes spanning the axis
warp  = (i // (c·t)) % w        w = warpsPerCTA[a]      — warps spanning the axis
group =  i // (c·t·w)         #groups = ceil(S/(c·t·w)) — tile replication (more registers)
```

`order` says which dim is contiguous (its head is the innermost dim), fixing the per-
component strides. Storage is a handful of integers — `O(polylog S)`.

### 4.3 The general formalism — LinearLayout over GF(2)

Triton represents this exact map as a **`LinearLayout`**: a linear map over GF(2)
(bit-vectors) from `(register, lane, warp, block)` to `(dim0, dim1, …)`:

```
L(a) = a₀·B₀ ⊕ a₁·B₁ ⊕ … ⊕ a_{M-1}·B_{M-1}        (⊕ = XOR, GF(2) addition)
L(x ⊕ y) = L(x) ⊕ L(y)                              (linearity)
```

Only the basis vectors `Bᵢ` at power-of-2 inputs are stored — `O((log S)²)` of them;
everything else is XOR-combined. Non-power-of-2 (NPOT) dimensions use ADD+UREM instead
of XOR. Key operations: `apply`, `compose`, `invertAndCompose`, `sublayout`,
`operator==`. Crucially, it is **encoding-agnostic**: `#blocked`, `#linear`, `#mma`,
`#dot_operand`, `#shared` all normalize through `toLinearLayout`, so the mixed-radix
arithmetic of §4.2 is just the `#blocked` special case. `include/triton/Tools/
LinearLayout.h`, `lib/Tools/LinearLayout.cpp`.

### 4.4 Encoding in TTGIR

TTGIR carries the layout as an attribute on each tensor type. The attributes
(`#blocked`, `#mma`, `#dot_operand` with `opIdx`/`parent`/`kWidth`, `#shared` with
swizzling, `#linear`, `#slice`) are "sugar"; `toLinearLayout(shape, attr)` is the
canonical form. `TritonGPUAttrDefs.td`, `LinearLayoutConversions.{h,cpp}`.

**Example** — a 1-D sum at `num_warps=4` (`bitequiv/tests/ttgir/sum_uno_nw4.ttgir`):

```mlir
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// ...
%0 = "tt.reduce"(%x_3) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
^bb0(%arg3: f32, %arg4: f32):
  %1 = arith.addf %arg3, %arg4 : f32
  tt.reduce.return %1 : f32
}) : (tensor<8192xf32, #blocked>) -> f32
```

Reading the encoding via §4.2 (axis 0, `S = 8192`): `c = sizePerThread = 4` (each
thread holds 4 contiguous elements), `t = threadsPerWarp = 32`, `w = warpsPerCTA = 4`
→ one CTA spans `4·32·4 = 512` elements, so `#groups = 8192/512 = 16` (16 elements
per thread total: 4 contiguous × 16 groups). The `reduction_ordering = "unordered"`
attribute fixes the fold/shuffle shapes. Together, `#blocked` + the ordering attr
fully determine the reduction tree.

### 4.5 Encoding in PTX/LLVM

The same map materializes as **per-thread index computation**: `emitIndices` builds
the input tuple `(register, laneId, warpId, blockId)`, then `applyLinearLayout`
evaluates `L` as a GF(2) matrix×vector in LLVM IR (`matrixVectorProd`, with NPOT
modulo where needed); `getLaneAndWarpId` extracts the hardware IDs. `lib/Conversion/
TritonGPUToLLVM/Utility.cpp`.

PTX *also* exposes the reduction tree directly. The within-warp butterfly is a
sequence of `shfl.sync.bfly.b32` with explicit offsets — for the unordered 4-warp sum
above (`bitequiv/tests/fixtures/ptx/sum_nw4.ptx`):

```ptx
shfl.sync.bfly.b32  %r85, %r84, 16, 31, -1;   //  within-warp tree:
shfl.sync.bfly.b32  %r87, %r86,  8, 31, -1;   //  count-down 16,8,4,2,1
shfl.sync.bfly.b32  %r89, %r88,  4, 31, -1;
shfl.sync.bfly.b32  %r91, %r90,  2, 31, -1;
shfl.sync.bfly.b32  %r93, %r92,  1, 31, -1;
shfl.sync.bfly.b32  %r98, %r36,  2, 31, -1;   //  cross-warp tree over 4 warps:
shfl.sync.bfly.b32  %r100,%r99,  1, 31, -1;   //  appended 2,1 (length grows log2(num_warps))
```

The offset sequence encodes both the within-warp tree (count-down for `unordered`,
count-up for `inner_tree`) and the cross-warp tree (its length grows
`log2(num_warps)` — `sum_nw8.ptx` shows `16 8 4 2 1` then `4 2 1`).

PTX is also where **FMA contraction** becomes visible — the same `tl.sum(x*y)`
compiles to (`dot_fuse_on.ptx` vs `dot_fuse_off.ptx`):

```ptx
// fusion ON                          // fusion OFF
fma.rn.f32 %r87, %r1, %r34, %r86;     mul.rn.f32 %r86, %r1, %r34;
fma.rn.f32 %r88, %r3, %r36, %r87;     add.rn.f32  ... ; (separate mul + add)
```

Same TTGIR, different PTX, different bits — the gap the PTX backstop catches.

### 4.6 Layout equivalence — the algorithm at each IR level

**General primitive.** Two layouts are equivalent iff their LinearLayouts are equal:
`areLayoutsEquivalent(shape, lhs, rhs)` computes `toLinearLayout` for each and
compares with `LinearLayout::operator==` (equal bases + equal out-dim sizes). This is
exactly the test that makes a `ConvertLayoutOp` a no-op. `Dialect.cpp`
(`areLayoutsEquivalent`), `Ops.cpp` (`isConvertTrivial`).

**TTGIR reduction equivalence (M1, implemented).** Each `tt.reduce` is summarized as
a canonical, hashable **signature**:

```
reduce | axis | ordering | nops | combine | layout
```

- **axis** — `op.getAxis()`.
- **ordering** — `reduction_ordering`, normalized (null/empty ⇒ `unordered`).
- **combine** — the ordered op-name sequence of the combine region (pre-order walk);
  distinguishes `addf`/`mulf`/`maxnumf`/`cmpf+select` (argmin/argmax)/Welford,
  uniformly for single- and multi-operand reduces.
- **layout** —
  - `inner_tree`: `inner_tree-invariant | sAxis=<shape[axis]>` (layout dropped because
    the order is layout-invariant; only the leaf count matters).
  - otherwise: the **axis-projected LinearLayout** of the operand,
    `toLinearLayout(srcTy).sublayout({register,lane,warp,block}, {dim<axis>})`,
    serialized via `toString()`. This is the real thread↔data map restricted to the
    reduce axis; `sublayout` keeps the out-dim size, so a different `BLOCK_SIZE`
    (different axis extent) yields a different signature automatically (shape-sound
    for free).

Two TTGIRs are equivalent iff their signature tuples are equal. The signature is
computed **MLIR-natively in C++** (using the compiler's own `toLinearLayout` + the
`tt.reduce` accessors — no regex over IR text), so it is correct across encodings and
robust to IR-printing changes. The same analysis is reusable by a future in-compiler
verification pass. `lib/Analysis/ReductionOrder.cpp`,
`include/triton/Analysis/ReductionOrder.h`, pybind `python/src/bitequiv.cc`, Python
wrapper `bitequiv/reduction_tree.py`.

**PTX backstop (implemented on `bitequiv-m1-ptx`).** TTGIR fixes the *association
order* but is provably blind to FMA contraction (§3.2). `ptx_reduction_signature`
reconstructs a signature per `.entry` from the PTX:

- the **ordered `shfl.sync.bfly` offset sequence** (within-warp + cross-warp tree);
- the **fp combine-opcode multiset with modifiers** (`fma.rn.f32` vs
  `add.rn.f32`+`mul.rn.f32` = fusion; `.rn` rounding; `.ftz` flush-to-zero);
- a derived **fused** flag (any `fma.*`).

**Refinement invariant:** PTX *refines* TTGIR — on pure-add reductions the two
partition the configs identically; on a `tl.sum(x*y)` dot reduction PTX splits the
fp-fusion pair TTGIR merges. Pair them with `reduction_equivalence_prune("both")`
(kept iff equal at both levels). `bitequiv/ptx_reduction.py`.

**Soundness contract (the guarantee, and its price).** Equal signature ⇒ identical
tree ⇒ identical bits at that IR level. The relation **never wrongly merges** two
configs whose bits differ — the kept set is always a *safe subset*. The reverse does
not hold: the check is **conservatively incomplete** — it may *over-split* genuinely-
equivalent configs (e.g. masked-zero `+0.0` padding that changes the axis extent;
`inner_tree` configs whose layouts differ). These misses cost tuning freedom, never
correctness, and are **measured, not assumed** (§5). A **soundness guard** prevents an
un-modeled reduction-like op (currently MMA `tt.dot`/`*mma*`, which has no
`tt.reduce`) from collapsing to an empty signature: `getReductionOrderSignatures`
appends a conservative `unanalyzed-mma | <fingerprint>` entry (dot op names +
attributes + operand/result types), so such modules match only on identical structure,
never on `() == ()`.

**GEMM/MMA roadmap (M3, not yet implemented).** The K-axis accumulation order has no
`tt.reduce`, so today it is covered by the conservative `unanalyzed-mma` guard (sound:
precision modes are detected, tiling is over-split). M3 will model MMA accumulation
order + precision mode as a first-class, layout-derived signature and persist an MMA
constraint across passes — lowering, e.g., `BLOCK_N=128` to two `N=64` instructions to
match a reference order, targeting bitwise-equivalence to **cuBLAS** on ~5 shapes
(H100 + B200).

---

## 5. Other essential technical details

### 5.1 Why TTGIR is the primary check level

TTGIR is the first level where the layout — and hence the reduction tree — is
explicit, yet still above the `ptxas` black box. For float `add`/`mul`, hardware
`redux.sync` is **never** emitted (only for integers, plus Blackwell f32 min/max), so
the tree is fully determined by TTGIR; PTX is needed only for FMA contraction. TTIR /
the config dict alone are layout-free → insufficient. The residual gap is `ptxas →
SASS` (SASS via `cuobjdump` would be absolute ground truth); `--fmad=false` pins the
fusion decision.

### 5.2 Why the problem is concrete, not symbolic

The **thread count** (`num_warps × 32`) and **every tensor shape** are compile-time
`constexpr` in TTGIR (Triton tensor shapes are always static). The runtime problem
size `N` only drives grid/loop counts and masks — it never enters the `tt.reduce`
tree, and is identical across the configs being compared. So the signature is a finite,
concrete object, not a symbolic expression. (The genuine exception — cross-CTA atomic
merges — is out of scope by construction, §3.4.)

### 5.3 Public interface

Import from `bitequiv` (repo root on `sys.path`; namespace package). Parsing a TTGIR
string is CPU-only — **no GPU, no kernel launch** — but it calls into the C++ analysis,
so it needs the **built triton** (`libtriton.bitequiv`); a C++ rebuild is required when
the analysis changes.

Standalone check on two TTGIR modules:

```python
from bitequiv.equivalence import reductions_equivalent, reduction_signature, classify

reductions_equivalent(ttgir_a, ttgir_b)      # -> bool : same bitwise reduction order?
reduction_signature(ttgir)                   # -> hashable descriptor (one entry per tt.reduce)
classify({"nw2": g2, "nw4": g4, "nw8": g8})  # -> OrderedDict{signature: [labels...]}
```

Getting TTGIR without a GPU (compile-only, no launch):

```python
ck = kernel.warmup(*args, grid=(1,), **constexprs)   # JITFunction.run with warmup=True
ttgir = ck.asm["ttgir"]                               # also "ttir" / "llir" / "ptx"
```

Enabling pruning in the autotuner — see §2.3. **Levels:** `level="ttgir"` (default),
`"ptx"`, `"both"`, or a list (kept iff equal at every level); levels resolve through
the injected `CHECKERS` registry. **Reference:** `reduction_equivalence_prune(level,
reference=...)` — `None` = first compiled config, or an external anchor (pre-compiled
kernel `.asm`, an `asm` dict, or raw IR text). **Extending:** a *key function*
`key_fn(config, asm, metadata) -> Hashable` maps an artifact to an equivalence key;
wrap it with `ir_based_prune_configs(key_fn, reference=...)`; add new levels to
`CHECKERS`.

### 5.4 Validation methodology

The static verdict is validated against **real bits** by `bitequiv/evaluation/`: run a
config matrix, compare each pair's static verdict against bit-identical outputs over
many random inputs, report a confusion matrix. The pass gate is **zero soundness
false-positives** ("declared equivalent but bits differ"). Latest run (H100): in-scope
kernels sound and exact; multi-operand/MMA boundary cases sound (Welford order-
divergence is *detected*, not abstained). The GPU evaluation — not a hand-written
oracle — is the ground truth.

### 5.5 Existing infrastructure reused (don't reinvent)

- `inner_tree` / `reduction_ordering` — full Python→MLIR→PTX path (count-up shuffles +
  balanced within-thread tree). Diff D100027220.
- `TRITON_STRICT_REDUCTION_ORDERING` env var (D101872700).
- `STABLE_REDUCTION` layer_norm workaround (D104785121; the 5.24% NE motivation).
- TritonParse — IR/PTX parse + multi-level diff, **text-level only** (no semantic
  tree analysis).
- Autotuner hooks: `early_config_prune`, `restore_value`, per-config IR/PTX dump,
  `CompiledKernel.metadata`.
- `triton_repro_bitwise.py` (D100024902); FBGEMM correctness-vs-perf pruning split.
- `LinearLayout` (`lib/Tools/LinearLayout`); reduce lowering (`ReduceOpToLLVM.cpp`).

### 5.6 Open questions / decisions log

- PTX vs SASS ground-truth gap (`ptxas` can still contract/reorder).
- Promote `reduction_ordering` from a request-flag to a **carried, verified
  constraint** (the §2.2 direction) — needs an in-compiler verification pass
  (`reductionOrdersEquivalent` is ready for it).
- cuBLAS reduction/MMA ordering — needs experimental determination (M3).
- TLX IR-surface differences vs standard Triton for this analysis.
- Modeling identity/`+0.0` padding to merge padded vs exact configs (reduce over-
  splitting).

---

## 6. Rules for future development (follow these)

1. **Soundness is non-negotiable.** Never declare two configs equivalent when their
   bits could differ. When unsure, over-split. Re-run the equivalence tests after any
   perf- or codegen-affecting change.
2. **Conservative-first, GPU evaluation as ground truth.** Ship the safe relation;
   measure incompleteness empirically (`bitequiv/evaluation/`, static verdict vs
   `torch.equal`). Gate = zero soundness false-positives.
3. **Keep Triton core decoupled.** Checkers are *injected* (`CHECKERS` /
   `ir_config_prune` predicates); never make `python/triton/...` import `bitequiv`.
4. **MLIR-native, not regex.** The signature is computed in C++ from parsed IR via
   `toLinearLayout`. Extend `lib/Analysis/ReductionOrder.cpp` (rebuild when changed);
   the same analysis backs the future in-compiler verification pass.
5. **Tests are the contract.** Real committed TTGIR fixtures (`tests/ttgir/`, parsed
   CPU-only via `gen_ttgir_fixtures.py`); add a fixture per new shape/case; validate
   against real bits when a GPU is available. Run `ruff` + `yapf` (Python) and
   `clang-format` (C++) — or `pre-commit` — before done.
6. **Keep this doc in sync.** Any change to the model, algorithm, interface, or scope
   updates the relevant section here; `PROGRESS.md` gets a one-line entry.

---

## 7. Appendix — code & lowering anchors

- **Analysis engine (C++, MLIR-native):** `include/triton/Analysis/ReductionOrder.h`
  + `lib/Analysis/ReductionOrder.cpp` (`getReductionOrderSignature(ReduceOp)`,
  `getReductionOrderSignatures(ModuleOp)`, `reductionOrdersEquivalent`); pybind
  `python/src/bitequiv.cc` (`libtriton.bitequiv.reduction_order_signatures`).
- **Python wrapper (TTGIR):** `bitequiv/reduction_tree.py` (`reduction_descriptor`,
  `reductions_equivalent`; cached MLIRContext + the binding).
- **PTX backstop engine:** `bitequiv/ptx_reduction.py` (`ptx_reduction_descriptor`,
  `ptx_reductions_equivalent`; parses `shfl.sync.bfly` offsets + fp-opcode multiset) —
  on the `bitequiv-m1-ptx` branch.
- **API:** `bitequiv/equivalence_ttgir.py` (`reduction_signature`,
  `same_reduction_order`, `reduction_equivalence_key`, `CHECKERS`,
  `ir_based_prune_configs`, `reduction_equivalence_prune`, `classify`); PTX-level API
  `bitequiv/equivalence_ptx.py` on the `bitequiv-m1-ptx` branch.
- **Autotuner hook:** `python/triton/runtime/autotuner.py` (`ir_config_prune`, called
  `(config, asm, metadata, reference)` after a `warmup=True` compile per config).
- **Reduction lowering:** `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`
  (`reduceWithinThreads`, `warpReduce` count-down/up, `reduceValueSequence`,
  `isInnerTree`); `ReduceScanCommon.h` (`applyCombineOp`); shuffle
  `third_party/nvidia/.../Utility.cpp` (`shuffleXor` → `shfl.sync.bfly.b32`);
  `redux.sync` gate `.../TargetInfo.cpp` (`matchReduxKind`).
- **Layout math:** `include/triton/Tools/LinearLayout.h` (`sublayout`, `toString`,
  `operator==`); `LinearLayoutConversions.{h,cpp}` (`toLinearLayout`;
  `register/lane/warp/block` → `dim{i}` conventions); per-thread index emission
  `lib/Conversion/TritonGPUToLLVM/Utility.cpp` (`emitIndices`, `applyLinearLayout`,
  `getLaneAndWarpId`, `matrixVectorProd`).
- **Layout equivalence:** `lib/Dialect/TritonGPU/IR/Dialect.cpp`
  (`areLayoutsEquivalent`); `lib/Dialect/TritonGPU/IR/Ops.cpp` (`isConvertTrivial`).
- **Inequivalence sources:** `DotOpToLLVM/FMA.cpp` (FMA), `F32DotTC.cpp` + NVIDIA
  `WGMMA.cpp` (MMA precision/TF32), `MultiCTAReduction.cpp` (out-of-scope multi-CTA).
- **Layout attributes:** `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`.
- **ReduceOp accessors:** `lib/Dialect/Triton/IR/Ops.cpp` (`getSingleCombiner`,
  `hasDefinedOrdering`).
- **Tests / fixtures:** `bitequiv/tests/test_reduction_tree.py`,
  `test_equivalence_ttgir.py`, `tests/ttgir/` (TTGIR), `gen_ttgir_fixtures.py`;
  `test_ptx_reduction.py` + `tests/fixtures/ptx/` on `bitequiv-m1-ptx`.
