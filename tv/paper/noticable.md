# Noticable Issues

A running log of issues that affect the design or scaling of the translation
validator and are worth surfacing in writeups / discussion. Each entry should
state the symptom, root cause, candidate fix, and trade-offs.

---

## 1. Store encoding explodes on realistic tile sizes

**Date noticed:** 2026-05-28
**Reproducer:** `triton-tv tv/test/TTIR/source/add_kernel.ttir tv/test/TTIR/source/add_kernel.ttir`
(self-equivalence of the standard 1024-element vector-add kernel)

### Symptom

- 4-element tile (`add_kernel_tiny.ttir`) validates in 0.14 s solver time.
- 1024-element tile (full `add_kernel.ttir`) does not complete: the process
  exceeds 3 GB resident memory within ~30 s and is killed before
  `solver.check()` returns.

### Root cause

`Memory::store` emits one `z3::store` node **per byte** of the tile. For a
1024-element f32 tile that is 1024 × 4 = **4096 nested `z3::store` calls**
per program. The full equivalence query then asks Z3 to compare two such
4096-deep store chains, modulo distinct symbolic names on every operand —
that is roughly 4096 × 4096 select-store unifications in the worst case.

```cpp
// Current Memory::store — one store node per byte
for (unsigned idx = 0; idx < n; idx++)
  for (unsigned j = 0; j < byteWidth; j++)
    array = z3::store(array, byteAddr_idx_j,
                      z3::ite(mask[idx], byte_j(val[idx]), array[byteAddr]));
```

The 4096-node AST itself is fine; the cost is in the solver. AbstractFp
axioms (universally quantified) compound the problem by triggering
quantifier instantiation on every distinct FP operand encountered along the
chain.

### Why the current encoding exists

It was introduced as a fix for a different bug: the prior encoding built
`array` as a `z3::lambda` over the address variable, which made
`checkEquivalence(s1, s2)` return `unknown` because
`solver.add(s1.array != s2.array)` requires deciding array extensionality
between two lambda terms — undecidable for the default DPLL(T) solver.
Replacing the lambda with nested `z3::store` calls put the encoding back
inside QF_AX (quantifier-free arrays + bitvectors), where Z3 is complete
but slow at scale.

### Candidate fix: quantified per-store update + pointwise equivalence

Two coupled changes:

**(a) One lambda update per store, not N×byteWidth nested stores.**

```cpp
z3::expr addr = ctx.bv_const("__addr", 64);
z3::expr i    = ctx.bv_const("__i",    32);  // tile index
z3::expr j    = ctx.bv_const("__j",     6);  // byte offset within element

z3::expr writeHere =
    z3::select(maskTile.expr, i) &&
    (addr == z3::select(ptrTile.expr, i) + z3::zext(j, 58));
z3::expr writeByte =
    toBV(ctx, z3::select(valTile.expr, i), elemTy, fpMode)
        .extract((j+1)*8 - 1, j*8);

new_array = z3::lambda(addr,
              z3::ite(z3::exists(i, j, writeHere),
                      writeByte,
                      z3::select(old_array, addr)));
```

AST size per store: **O(1)** instead of O(N × byteWidth).

**(b) Pointwise equivalence check at a symbolic witness address.**

```cpp
// Instead of:
solver.add(s1.array != s2.array);   // requires array extensionality

// Use:
z3::expr witness = ctx.bv_const("__witness_addr", 64);
solver.add(z3::select(s1.array, witness) != z3::select(s2.array, witness));
```

`select(λaddr. body, witness)` β-reduces to `body[addr := witness]`, which
is quantifier-free (modulo the inner `∃i,j` in the `ite` condition). The
solver only needs to find one witness address where the byte differs.

### Trade-offs

|                            | Nested `z3::store` (current) | Lambda + pointwise check |
|----------------------------|------------------------------|--------------------------|
| AST size per store         | O(N × byteWidth)             | O(1)                     |
| Equality check theory      | QF_AX                        | QF with witness ∃         |
| Decidable                  | yes                          | yes (after β-reduction)  |
| 1024-tile feasibility      | no                           | expected yes             |
| Implementation complexity  | low                          | medium                   |

### Open questions

- Does Z3 actually beta-reduce `select(lambda, ...)` aggressively enough,
  or does it need a manual `simplify()` pass?
- The `∃i,j` inside the `ite` condition is bounded (`i ∈ [0,N), j ∈ [0,bw)`)
  and could be statically unfolded into a finite disjunction. For very small
  `N` that may be faster than asking Z3 to handle the quantifier.
- Does this approach degrade when the kernel performs many sequential stores
  (e.g. flash attention's tiled accumulator)? Each store nests one more
  lambda layer; eventually the β-reduction depth becomes the bottleneck.

### References

- mlir-tv uses a closely related encoding (lambda-based memory + pointwise
  comparison). Cross-check their implementation before committing to this
  design.

### Resolution (2026-06-25) — FIXED

Both coupled changes are implemented:

- `Memory::store` now builds **one `z3::lambda(addr, body)`** per store instead
  of the per-byte `z3::store` chain (`tv/semantics/Memory.cpp`). The `∃i,j` is
  **statically unfolded** into a nested `ite` over the known lane/byte indices
  (tile shapes are compile-time constants) — open question #2 resolved in favour
  of unfolding, so there is no real quantifier and no instantiation cost. Lanes
  are applied ascending, so the highest lane is the outermost `ite`
  (last-writer-wins on aliasing, matching the old chain).
- Comparison switched to the **pointwise witness address**: `checkEquivalence`
  (`tv/semantics/State.cpp`) and the inlined check in `tv/triton-tv.cpp` now use
  `select(m1.array, witness) != select(m2.array, witness)` instead of
  `array != array`. No manual `simplify()` was needed (open question #1: Z3
  β-reduces `select(lambda, …)` fine).

**Measured (1024-element f32 `add_kernel`, same machine):**

|                          | Before                          | After                              |
|--------------------------|---------------------------------|------------------------------------|
| self-equivalence         | timeout >150 s, 16.7 GB, killed | EQUIVALENT, 0.30 s solver, 153 MB  |
| unoptimized vs optimized | never completed                 | EQUIVALENT, 0.31 s solver, 156 MB  |
| tiny pid×5 (address bug) | timeout >60 s                   | NOT EQUIVALENT, <1 s               |

≈300× faster, ≈100× less memory. The address-divergence timeout (separate from
the size blowup) also disappeared. All 43 unit tests pass (added
`MemoryModel.LargeTileWitnessEquivalence`).

Open question #3 (lambda layering for many sequential stores, e.g. flash
attention's accumulator) is still untested — revisit when scan/accumulator
kernels land.

---
