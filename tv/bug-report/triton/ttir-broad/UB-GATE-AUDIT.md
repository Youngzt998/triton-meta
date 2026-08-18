# UB gate audit — ttir-broad

Line `ttgir-broad` found that the determinism pre-screen has a blind spot:
a **scatter store** (a `tt.store` whose address comes from loaded index data)
is a write-write race when indices repeat, but one compiled binary has a fixed
write order, so running it three times always agrees. Only a *different*
binary reorders the writes, so the race shows up as a cross-compilation bit
difference and looks like a miscompile. A race is undefined behaviour, so per
the campaign plan (section 4) these are out of scope and are not findings.

This line added the same gate and re-checked everything already committed.

## Result: no finding needed retracting

| report | kernel | gate |
|---|---|---|
| `HIT-0005` | `ind::triton_per_fused_mean_std_var_0_fdf7c466` | clean, 3 stores, none data-dependent |
| `HIT-0006` | `ind::triton_for_fused_1_a94716eb` | clean (see the false-positive note below) |
| `HIT-0007` | `bench::_conv_transpose2d_stride2_pad1_3x3_kernel_312550c9` | clean; the store address is built only from `tt.get_program_id`, constants and `tt.make_range` |

`HIT-0007` is worth spelling out because the name invites suspicion: a
conv-transpose *can* be written as a scatter. This one is not. Its single
`tt.store` address chain is
`tt.addptr <- arith.addi <- tt.broadcast <- tt.expand_dims <- arith.addi <- tt.splat <- arith.remsi <- tt.get_program_id`,
with no `tt.load` anywhere in it. It is the gather-style formulation, where
each output element sums its own contributions, so there is no race.

## Corpus effect

Scanned all 511 kernels whose root TTIR builds:

| | count |
|---|---|
| genuine scatter store | 9 |
| already dropped earlier as non-deterministic | 3 of those 9 |
| newly dropped as `ub-race` by this gate | 6 |
| float atomic read-modify-write | 0 |
| clean | 495 |

Newly dropped: `_combine_topk_swa_indices_kernel`,
`_cyclic_jacobi_finalize_kernel`, `_unique_dim_inverse_permutation_kernel`,
`max_unpool2d_kernel`, `nonzero_kernel`, `repeat_interleave_tensor_kernel`.
Usable corpus went from 482 to 476.

That three of the nine had already been caught by the determinism pre-screen is
a useful cross-check: for those kernels the write order varies within a single
binary too, so both gates agree.

## A correctness bug in the shared scanner — line `ttgir-broad` should look

The implementation this gate was copied from,
`/home/youngzt/fuzz/ttgir-broad/fz/irscan.py`, builds **one flat
`name -> definition` dict for the whole IR text**. MLIR restarts SSA numbering
inside a nested region, so that dict silently merges names across regions and
the backward walk can leave the scope it started in.

Concretely, in `ind::triton_for_fused_1_a94716eb` (an inductor combo kernel with
three sibling `scf.if` bodies) `%40` is defined three times:

```
line  55:      %40 = tt.addptr %39, %5        <- the real store address
line 103:        %40 = arith.addf %19, %39
line 152:          %40 = tt.precise_divf %27, %39
```

The store at line 56 is `tt.store %40, %24`, whose address is
`tt.addptr(tt.splat(%arg15), arange)` — not data-dependent at all. But the flat
dict keeps only the last `%40`, so the walk followed the `tt.precise_divf` from
a different region, reached a `tt.load`, and reported a scatter store.

Measured over this corpus: the flat walk flags **16** kernels, of which **7 are
false positives** (44%). The failure is not one-sided — the same collision can
replace a genuinely data-dependent definition with a benign one and hide a real
scatter, so it can under-report too. On this corpus it happened not to
(scope-aware finds nothing the flat walk misses), but that is luck, not a
property.

False positives on this corpus: `ind::triton_for_fused_0_ca70313c`,
`ind::triton_for_fused_0_0102b695`, `ind::triton_for_fused_1_a94716eb`,
`ind::triton_poi_fused_0_12e96f86`, `bench::_kernel_H32_D128_32ca89a8`,
`bench::_kernel_H64_D128_e581c92f`, `bench::renorm_kernel_scale_51e95bd7`.

The copy used by this line, `/home/youngzt/fuzz/ttir-broad/irscan.py`, tracks
definitions **per lexical scope keyed by brace depth** and resolves a name
innermost-scope-first. It keeps the same `scan()` / `ub_reason()` interface, so
it is a drop-in replacement. It is checked against two controls: a hand-written
scatter kernel (flagged, as it must be) and the same kernel with a colliding
`%5` in a nested `scf.if` after the store (still flagged).

Line `ttgir-broad` retracted eight reports on the strength of the flat walk. At
least some of those may have been sound findings; they are worth re-checking
with a scope-aware walk before staying retracted.

## Second net, also from line `ttgir-broad`

If three distinct culprit pass sets fire on one kernel, the kernel is retired as
`suspect-unstable` rather than filed as three separate compiler bugs — that
catches order dependence a static scan cannot follow, such as aliasing between
two pointer arguments. Retired kernels are listed in
`/home/youngzt/fuzz/ttir-broad/HITS/SUSPECT.md`, which says plainly that any
report already written against one of them needs a human look.
