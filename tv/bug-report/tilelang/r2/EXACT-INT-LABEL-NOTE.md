# Note: the `exact-int-accum:reassociation-excluded` label was wrong on kernels with a transcendental

Written 2026-08-19 17:50 by the R5 line. **No existing report was edited** — the
reviewer owns those. This file only lists which of them carry a label that the
line now knows is unsound, and what was changed so no later report repeats it.

## What was wrong

`tlfz/core.py::exact_int_ok` decided the label. It only bounded the *input*
values (integers in [-8,8], `bound^2 * n < 2^24`) and then assumed the kernel
computes a plain sum of products, so an fp32 accumulator stays exact and no
reordering of the sum can change a bit.

That assumption does not hold for every kernel. In the `fa` family `T.exp2`
runs before the row sum, and in `softmax` `T.exp` runs before `reduce_sum`.
Once a transcendental is in the chain the accumulator no longer holds an
integer, so summation order *can* change a bit — which is exactly what the
label claims to rule out.

Two findings were already rejected over this (`r2/HIT-0003`, `r2/HIT-0004`, both
on `gen_fa_b79103583e1427`; never committed here): with `-fmad=false` the
difference vanished on all three differing draws, i.e. it was ordinary fma
contraction, the very thing the label said it could not be.

## The fix (live on the line since 2026-08-19 17:45:50)

`tlfz/irscan.py::inexact_accum_reason` walks the tile-level TIR and reports the
first step that is not exact on integers:

* a call whose name is a transcendental (`exp`, `exp2`, `log`, `rsqrt`, `pow`,
  `sigmoid`, `tanh`, `sin`, `__exp10`, `ieee_fdiv`, `fast_rcp`, …), including
  the function name of a `call_extern`, and
* any floating-point division or remainder node (integer index arithmetic such
  as `FloorDiv` on `int32` from `T.ceildiv` does not count).

`exact_int_ok` now applies the label only when that scan comes back empty. If
the scan cannot run, or the kernel is not available, the label is **omitted**.
There is no weaker variant of the label — absent is fine, wrong is not.

Each new `diff` record also carries `confirm.accum_inexact`, which is the reason
string when the label was withheld and `""` when the chain is exact.

Scope: 102 of the 720 corpus kernels are flagged by the scan; the other 618 keep
the control. Of the 14 generated families, `fa`, `softmax` and `layernorm` are
flagged; `gemm`, `gemv`, `rr`, `cumsum` and the elementwise ones are not.

## Findings written before the fix that carry the label

Audited 2026-08-19 17:48 over every `HIT-*/record.json` in this directory. All
of them were written before 17:45, so all of them predate the fix. **9 carry the
label, and 6 of those 9 are unsound.**

| report | kernel | family | label is | why |
|---|---|---|---|---|
| HIT-0006 | `cap_0b881ee73fbdf83c` | captured | **unsound** | kernel contains `tirx.exp` |
| HIT-0008 | `gen_gemv_ef525af0d65e95` | gemv | sound | plain sum of products |
| HIT-0009 | `gen_gemv_ef525af0d65e95` | gemv | sound | plain sum of products |
| HIT-0016 | `gen_fa_915604d75722ed` | fa | **unsound** | `tirx.exp2` before the row sum |
| HIT-0019 | `gen_softmax_de6674ba60b530` | softmax | **unsound** | `tirx.exp` before `reduce_sum` |
| HIT-0020 | `gen_softmax_de6674ba60b530` | softmax | **unsound** | `tirx.exp` before `reduce_sum` |
| HIT-0024 | `gen_fa_d13e1488703474` | fa | **unsound** | `tirx.exp2` before the row sum |
| HIT-0028 | `cap_401ab8ec5f03638a` | captured | sound | no transcendental, no float divide |
| HIT-0029 | `gen_softmax_9915d56f9d1910` | softmax | **unsound** | `tirx.exp` before `reduce_sum` |

"Unsound" means only this: **that one label is not evidence**, so the finding
has to be judged without it. It does not say the finding is wrong. Nothing else
in those reports is affected — the difference itself, the determinism and heap
checks, the fast-math ablation and the bisection are all unchanged.

The same audit over the line's working copy `/home/youngzt/fuzz-tilelang-r2/HITS`
(which also holds the rejected HIT-0003/0004 and an uncommitted HIT-0005) finds
12 with the label, 8 of them unsound.

The table is closed: `HIT-0033` and later are written by the fixed code. The
first `softmax` finding after the fix, `HIT-0040` (17:51), carries **no**
exact-integer label and records
``confirm.accum_inexact = "`tirx.exp` is a transcendental / inexact op ..."``
instead — the same kernel shape that produced HIT-0019/0020/0029 with the label.

Re-run the audit at any time (read-only, it never writes into a report):

```bash
source /home/youngzt/fuzz-tilelang-r2/env.sh
cd /home/youngzt/fuzz-tilelang-r2 && python audit_labels.py
```

## One gap that is still open

The exactness bound `< 2^24` is the fp32 one, but the `gemm` family draws its
accumulator dtype as `float32, float32, <operand dtype>`, so one gemm instance
in three accumulates in fp16/bf16, whose exact-integer range is 2^11 / 2^8. For
such an instance the label could still be too strong. No finding written so far
is affected (no gemm-family finding carries the label today), and fixing it
needs a rule that identifies the accumulator buffer without killing the control
for the common fp32-accumulator gemm, so it was left alone rather than guessed
at on a running line.
