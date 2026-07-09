# TTIR / TTGIR passes: which ones can break bitwise equivalence

When we compare two compilations of the same kernel, some optimization passes
are **designed** to change the floating-point result (they reassociate a
reduction, fuse into a dot accumulator, or drop precision to tf32). A bitwise
diff from those is *expected*, not a bug. Other passes only move data around,
change layouts, or delete redundant work — they should produce the **exact same
bits**, so a diff there is a strong signal of a real compiler bug.

This doc splits the NVIDIA TTIR/TTGIR optimization passes into those two groups.
Flag names are exactly as `triton-opt --help` prints them. Two ready-to-use
pipelines mirror these lists: `passes/numerics_changing.txt` and
`passes/bit_preserving.txt`.

> This is a *hypothesis* to test, not ground truth. The fuzzer is what actually
> proves it: run a "bit-preserving" pass and if it produces a diff, you have
> either a bug or an unlisted reduction/dot-order coupling (see the caveat).

---

## Group A — numerics-changing BY DESIGN (a bitwise diff is EXPECTED)

These passes alter the FP math itself (precision, fusion, or reduction order):

| Pass flag | Why it changes bits |
|---|---|
| `--tritongpu-accelerate-matmul` | Selects tensor-core MMA and tf32/lower-precision dot; different accumulation and rounding than the generic fp32 dot. |
| `--tritongpu-F32DotTC` | Emulates an fp32 dot with tf32/bf16 splitting — precision change by design. |
| `--triton-combine` | `CombineDotAddPattern` folds `dot + add` into the dot **accumulator**, and `CombineBroadcastMulReducePattern` turns `sum(x*y)` into a `dot` — both change FP accumulation/reduction order. |
| `--tritongpu-optimize-thread-locality` | Relayouts a tensor feeding a `tt.reduce` "to minimize cross-thread communication for the reduction" — i.e. it **reorders the reduction** on purpose. |

## Group B — bit-preserving (results SHOULD be identical; a diff is suspicious)

These only change layout, data movement, scheduling, or remove redundant work.
The per-element math and its evaluation order are unchanged.

Layout / data movement:
- `--convert-triton-to-tritongpu` (structural: assigns thread layouts)
- `--tritongpu-coalesce`
- `--tritongpu-remove-layout-conversions`
- `--tritongpu-reduce-data-duplication`
- `--tritongpu-optimize-dot-operands` (fuses transposes / operand layout)
- `--tritongpu-coalesce-async-copy`

Scheduling / pipelining (same ops, reordered execution, same per-element math):
- `--tritongpu-pipeline`
- `--tritongpu-schedule-loops`
- `--tritongpu-assign-latencies`
- `--tritongpu-fuse-nested-loops`
- `--tritongpu-prefetch`
- `--tritongpu-reorder-instructions`

Redundancy / cleanup / structure:
- `--triton-reorder-broadcast` (elementwise-before/after-broadcast gives same values)
- `--triton-licm` (loop-invariant code motion + masked-load hoist)
- `--triton-loop-aware-cse`, `--cse`
- `--triton-loop-unroll`
- `--triton-rewrite-tensor-descriptor-to-pointer`
- `--tritongpu-combine-tensor-select-and-if`
- `--tritongpu-optimize-accumulator-init` (`0.0 + x == x`; see signed-zero caveat)
- `--canonicalize`, `--symbol-dce`, `--sccp`

---

## Important caveat: layout passes can indirectly reorder reductions/dots

Group B is "value-preserving" for **pointwise** work. But the *layout* of a
tensor determines how a `tt.reduce` or `tt.dot` combines partial results across
threads and warps. So on a kernel that contains a reduction or a dot, a pure
layout pass (`coalesce`, `remove-layout-conversions`, `optimize-dot-operands`,
`reduce-data-duplication`) can change the summation tree and legitimately differ
in the last bits — even though it never touched the math.

Practical rule:
- On **pointwise** kernels (e.g. `vector_add`), a diff from any Group B pass is
  a real bug candidate.
- On kernels with **`tt.reduce` / `tt.dot`** (softmax, layer_norm, matmul), a
  diff from a Group B *layout* pass may still be by-design reduction/dot-order
  coupling; inspect the saved `ref.ttgir` / `cand.ttgir` before calling it a bug.

Other edge notes:
- `--tritongpu-optimize-accumulator-init` is bitwise-equal except the `-0.0`
  case (`0.0 + -0.0 == 0.0`, but skipping the add keeps `-0.0`).
- `--canonicalize` avoids unsafe FP folds unless `fastmath` flags are present;
  treat it as bit-preserving unless the kernel sets fastmath.

## Not classified here (not "optimizations" you toggle for equivalence)
Warp-specialization, TMA/MMA/TMEM lowering, memory/warp-group allocation, fence
insertion, and CTA planning are correctness-lowering or hardware-mapping passes,
not FP-optimization choices — they are not meant to be A/B'd for bitwise
equivalence and are omitted from both lists.
