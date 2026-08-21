# Retracted: HIT-0015, 0016, 0018, 0019, 0020, 0021, 0022 — and the gate defect behind them

All six are warp-specialized `mm_tma` cases. They are withdrawn: in each one the
**reference build was not deterministic**, so the difference cannot be
attributed to the pass set.

| report | culprit set | why it is withdrawn |
|---|---|---|
| HIT-0015 | interleave-tmem, loop-aware-cse-2, optimize-dot-operands-1, pipeline, plan-cta | reference disagrees with itself in 19/20 reps |
| HIT-0016 | optimize-dot-operands-1 | reference disagrees with itself in 6/20 reps |
| HIT-0018 | optimize-dot-operands-1, schedule-loops | reference disagrees with itself in 4/20 reps |
| HIT-0019 | optimize-dot-operands-1, schedule-loops, triton-licm | reference disagrees with itself in 19/20 reps |
| HIT-0020 | optimize-dot-operands-1, triton-licm | difference is `+0.0` vs `-0.0` on 1 of 1048576 elements; reference not self-consistent at that seed |
| HIT-0021 | canonicalize-a, combine-tensor-select-and-if, optimize-dot-operands-1, remove-layout-conversions-1 | does not reproduce at its own seed |
| HIT-0022 | combine-tensor-select-and-if, optimize-dot-operands-1 | reference disagrees with itself in **119/120** reps, 51 distinct outputs |

The candidate build was self-consistent in all six.

## Why the determinism gate missed it

The gate re-launched the same build 3 times on the same input and compared. It
rebound the input dict each rep, so the previous rep's tensors were freed and
torch's caching allocator handed the same blocks straight back. **The buffers
never moved**, and a build that is only flaky when its addresses drift passed.
Three reps was also far too few.

Two things were wrong, and fixing only one is not enough. Measured on these six
references, 20 reps each, counting reps that disagree with the first:

| report | free between reps | keep buffers alive |
|---|---|---|
| HIT-0015 | 3/20 | **19/20** |
| HIT-0016 | **6/20** | 1/20 |
| HIT-0018 | **4/20** | 0/20 |
| HIT-0019 | **19/20** | 0/20 |
| HIT-0020 | 0/20 | 0/20 (caught by the signed-zero rule instead) |
| HIT-0021 | 2/20 | 0/20 |

Neither allocation pattern dominates: keeping buffers alive is the only thing
that exposes HIT-0015, and freeing between reps is the only thing that exposes
HIT-0018 and HIT-0019. So the gate now runs **20 reps under both patterns** and
requires both to pass.

Added at the same time: a difference made only of `+0.0` vs `-0.0` is treated as
a self-consistency failure rather than a finding, because that is what this
flakiness looks like when it does reach the comparison.

With the new gate the reference of HIT-0015, 0016, 0018, 0019 and 0020 is
rejected outright, and HIT-0021 fails to reproduce. The whole corpus has been
re-screened under it.

**HIT-0009 is unaffected** and stands: both of its builds pass the new gate
(20 reps, both patterns), and it was separately confirmed against an exact fp32
model with the non-warp-specialized build bit-identical to it.


## Second correction: 20 reps was still not enough

The 20-rep gate above is only ~54% sensitive to a build that is flaky on 4% of
reps, which is the rate measured on this family. It let HIT-0022 through. Two
more things were wrong:

* **depth** — 75 reps gives 95% detection, 115 gives 99%;
* **input** — the flakiness is input dependent. HIT-0022's reference was
  119/120 on `(tiny, its own seed)` but only 1/120 on `(randn, 1234)`, which is
  the input the pre-screen happened to use.

The gate now runs, under both allocation patterns:

| where | inputs | reps per input |
|---|---|---|
| pre-screen, warp-specialized case | randn, tiny, intvalued, wide | 60 |
| pre-screen, other case | randn, extreme | 30 |
| triage, on the input that actually produced the mismatch | that one | 120 |

HIT-0009 was re-verified under this standard at 200 reps and holds with 0
disagreements in 4000 launches; see its report.
