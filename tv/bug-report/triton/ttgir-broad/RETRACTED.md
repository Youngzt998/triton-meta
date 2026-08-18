# Retracted findings — ttgir-broad

## What happened

Eight reports were written and committed before this line had a gate for
**undefined behaviour**. They were then retracted in one block, with a single
blanket reason: that they were all on kernels doing a *scatter store* (the write
address is computed from loaded index data, so two lanes of one launch can write
the same address, the result depends on the order the hardware runs them in,
which is a write-write race, which is UB — campaign plan section 4 puts data
races out of scope).

That blanket retraction was **too broad**. It was written from the pattern of
the flood (one `_index_put` kernel producing findings from a dozen unrelated
passes) without checking each report against the scanner. Three of the eight
were not on UB kernels at all.

Since then the scanner (`fz/irscan.py`) was corrected for two bugs found by the
sister lines:

* **Scope.** MLIR restarts SSA numbering inside a nested region, so the same
  `%N` can be defined once per sibling `scf.if` body. The old flat, whole-file
  `name -> definition` dict kept only the last one, so the backward walk could
  leave the scope it started in and follow an unrelated definition. Wrong in
  both directions: it invented scatters that were not there, and a collision can
  equally hide a real one. Definitions are now tracked per lexical scope, keyed
  by brace depth, resolved innermost-first.
* **Coverage.** Only the first SSA operand of a store was walked. That is right
  for `tt.store`, but `tt.descriptor_store %desc[%x, %y], %val` puts the
  descriptor first, so a *coordinate* built from loaded data was missed. Every
  coordinate is now walked, and the lowered TMA store ops
  (`ttng.async_tma_copy_local_to_global`, `ttng.async_tma_scatter`) are covered.

Measured effect of the fix on this line's corpus: of the 33 kernels the old
scanner rejected as UB, **9 (27%) were false positives** and are back in the
sweep. (Line A measured 7 of 16, 44%, on their corpus.)

## Re-check of each retracted report, against the corrected scanner

| report | kernel | culprit pass | corrected scanner | outcome |
|---|---|---|---|---|
| HIT-0001 | `bch:matmul_kernel_10e0e451` | `--tritongpu-accelerate-matmul` | not UB | **retraction was wrong** |
| HIT-0002 | `ind:triton_per_fused_native_layer_norm_native_layer_norm_backward_0_92be64e3` | `--tritongpu-remove-layout-conversions` | not UB | **retraction was wrong** |
| HIT-0003 | `bch:_cyclic_jacobi_finalize_kernel_4665c432` | `--triton-loop-aware-cse` | scatter store (`tt.store` address depends on `tt.load`) | retraction stands |
| HIT-0004 | `bch:_index_put_jit_function_2176e2b9` | 4-pass chunk incl. `--triton-nvidia-gpu-plan-cta` | scatter store | retraction stands |
| HIT-0005 | `bch:_index_put_jit_function_2176e2b9` | `--triton-loop-aware-cse` | scatter store | retraction stands |
| HIT-0008 | `ind:triton_per_fused__softmax_backward_data_0_351b228a` | `--triton-loop-aware-cse` | not UB | **retraction was wrong** |
| HIT-0016 | `bch:_index_put_jit_function_3dacf625` | `--tritongpu-allocate-warp-groups` | scatter store | retraction stands |
| HIT-0020 | `bch:_index_put_jit_function_702d0ef4` | `--tritongpu-coalesce` | scatter store | retraction stands |

Five of eight were correctly retracted: those kernels really are racy, so two
compilations are allowed to disagree on them.

## The three that should not have been retracted

None of them is claimed here to be a compiler bug — the scope of this campaign
is bit-equivalence only, and each carries its own labels. What the retraction
got wrong is the *reason*: they are not UB.

* **HIT-0001**, `--tritongpu-accelerate-matmul` on a plain fp32 matmul. This is
  the campaign's own planted-difference check: the pass turns `tt.dot` into
  `ttng.warp_group_dot`, which is a numerics change *by design*. Correct label:
  `pass-class: numerics`, `numerics-changing-by-design`. Not UB, not a bug.
* **HIT-0002** and **HIT-0008**, layout / CSE passes on inductor reduction
  kernels (`native_layer_norm_backward`, `softmax_backward_data`). These belong
  to the `layout-reorders-reduction` label class: a layout pass decides how
  partial sums are combined, so it can legitimately change the last bits. The
  small-whole-number control is what says more.

All three kernels are back in the sweep under the corrected scanner, so if the
differences are real they will be re-found and re-reported with full labels,
measurements and both controls (heap shift, small-integer inputs).

## Note on report numbers

The hit-id counter was reset once during setup, so a few numbers were reused
before that was fixed. The table above identifies each retracted report by the
kernel and pass recorded in it at retraction time (commit `8649f363b`), not by
the current contents of the file with that name.
