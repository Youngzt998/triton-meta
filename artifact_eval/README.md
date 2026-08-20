# Artifact evaluation — reviewer's manual

Everything runs from one script, `artifact.py`. Each claim in the paper is a separate step, so
you can evaluate one at a time rather than all of them.

## Quick start

`bitequiv` is not installed, so the repository root has to be on `PYTHONPATH`; and one GPU has to
be pinned, because the timing steps assume they own the device. Run `nvidia-smi` first and pick a
GPU nobody else is using — a shared GPU does not change the bit results, but it makes every
timing meaningless.

```
export PYTHONPATH=$(git rev-parse --show-toplevel)
export CUDA_VISIBLE_DEVICES=0

python artifact_eval/artifact.py --list                        # what can be run
python artifact_eval/artifact.py --run gemm.bitmatch --minutes 20
python artifact_eval/artifact.py --run gemm --minutes 20       # the whole gemm section
python artifact_eval/artifact.py --run all --minutes 20        # every section
python artifact_eval/artifact.py --export                      # cache/*.jsonl -> data/*.csv
```

Records stream into `cache/` as they are produced, so interrupting a run loses at most one record
and `--export` still works on whatever was collected. Every run also appends a line to
`data/env.csv` recording the GPU, the cuBLASLt version, and the commit.

**Not every step is written yet.** `--list` marks each one `implemented` or `PLACEHOLDER`. A
placeholder prints the measurement it is going to make — the claim, the method, the cost, what you
should see and how to judge it — then writes nothing and exits 0. Its table columns are already
declared, so `data/FORMAT.md` describes the table before it has any rows. This is deliberate: a
reviewer should see the gaps rather than wonder why a step printed nothing.

## Section 1 — the GEMM matches cuBLAS (`gemm.*`)

### `gemm.bitmatch` — the Triton GEMM returns cuBLAS's exact bytes

**Claim.** For arbitrary shapes, `cublas_equivalent_gemm` produces output byte-identical to
cuBLAS, not merely close.

**Method.** Random shapes are drawn from six regimes — square, thin, very deep `K`, decode-sized,
vector (`M` or `N` equal to 1), and small-`M` — in fp16 and fp8. Two of the regimes exist because
without them cuBLAS never reaches four of its nine kernel families: the CUDA-core chain GEMM needs
a small `M` **and** an `N` that is not a multiple of 8, and the vector kernels need `M` or `N`
exactly 1. Deep `K` is in the mix deliberately: it is the only place
cuBLAS's own split-K defect appears, and a sweep that avoids it reports a clean 100% while having
tested nothing interesting. For each shape the script runs both implementations over `--reps`
independent input draws and compares a hash of the output bytes. Even-numbered draws are ordinary
gaussian; odd-numbered draws spread the exponents across the dtype's usable range, which is what
makes a change in the *order* of the additions visible — with narrow exponents almost any
regrouping rounds to the same bits and the check passes things it should not.

**Cost.** Set by `--minutes`. Roughly 100 shapes per minute at `--reps 4`, fewer at `--reps 10`.
Twenty minutes is enough to see the shape of the result; longer is better evidence, not a
different one.

**What you should see.** A count of shapes byte-identical to cuBLAS, and a line per mismatch
naming the shape, the cuBLAS kernel family the plan resolved to, and which draws differed.

**How to judge it.** Read `data/gemm_bitmatch.csv`; `n_differ` is zero for a byte-identical shape.
**A mismatch is not automatically a failure of this work.** On some shapes cuBLAS itself sums only
the first `K - (K % block_k)` elements. Mismatches concentrated on `mode=split` at very deep `K`
are that defect; run `gemm.cublas-bug` before concluding otherwise. The script logs mismatches
rather than adjudicating them, on purpose — the judgement is yours.

### `gemm.perf.random` and `gemm.perf.static` — bit-exactness costs little — *placeholders*

**Claim.** Requiring the output to match cuBLAS bit for bit costs very little performance. This is
a different and much smaller quantity than the gap between Triton and cuBLAS, and the two are
easy to conflate.

Three arms per shape, same inputs and same timing. Arm 1 is cuBLAS through the hot closure; it is
both the baseline and the reference the bits are compared against. Arm 2 is torch with no numerics
requirement, `torch.compile(mode="max-autotune-no-cudagraphs")`, reported twice — as its autotuner
actually picks, and as its best Triton template with the extern call excluded. Arm 3 is ours,
tuned, and byte-identical to arm 1. `gemm.perf.random` runs this on random shapes and groups the
result by shape family; `gemm.perf.static` runs it on the fixed layer dimensions of real models.
Run either step for the full design.

These two replace an earlier `gemm.perf`, whose unconstrained arm was a Triton configuration sweep
we had written ourselves. A ceiling built out of our own kernel is only as good as the
configuration space we happened to type in, and a reviewer has no reason to trust it. What
`torch.compile` picks with max-autotune is the honest stand-in for what a user gets when they ask
for speed and nothing else.

### `gemm.fusion` — fusing an epilogue cuBLAS cannot express — *placeholder*

**Claim.** For an epilogue cuBLASLt cannot express, folding it into a GEMM that is still
byte-identical to cuBLAS beats the two-kernel path a user gets today — and the bit constraint is
not what decides whether it wins.

Run the step for the design. Measured data already exists and is not wired into the tables yet:
`fusion_oracle/three_way.jsonl` holds a three-way oracle run (eager unfused, Inductor, Inductor's
own Triton template, and ours tuned and untuned, each with its bit check and its device time), and
`data/summary.csv` carries the `gemm.fusion.oracle` rows summarising it. The older per-model runs
in `fusion_real/` and `fusion_moe/` filled `gemm_fusion_dense.csv` and `gemm_fusion_moe.csv`. All
of it is run by hand rather than through this driver, which is the gap.

### `gemm.cublas-bug` — some mismatches are cuBLAS's own

**Claim.** cuBLAS itself returns a wrong answer on some shapes, so a disagreement is not
automatically ours.

**Method.** This step only prints instructions; the reproducer is a standalone script that depends
on nothing in this project — ctypes, torch and `libcublasLt` only — so it can be handed to NVIDIA
unchanged. Run it yourself, once per library you want to test. `A` and `B` are all ones, so every
element of the result must be exactly `K`; nothing rounds, because the products are exact and the
output is fp32.

**What you should see.** On an sm_103 GPU with cuBLASLt 13.1.1 or 13.2.2, part one (`K = 8648`)
comes back as 8640 — eight terms of `1 * 1` missing, which is not a rounding error of any size.
The script ends in an assertion, so a successful reproduction exits non-zero.

**How to judge it.** The trigger is not "K is large". The loss happens only where cuBLASLt decides
to split `K` across threadblocks, and that decision moves with both the architecture and the
library version, which is why the script carries three different `K` values and none of them fires
everywhere. If none fires on your machine, that bounds the defect to other configurations; it does
not show the reproducer is wrong.

## Section 2 — the reduction ordering switch (`inner_tree.*`)

**This section evaluates work that is not ours.** The reduction-ordering mechanism —
`reduction_ordering` / `inner_tree` and the `TRITON_STRICT_REDUCTION_ORDERING`
environment variable — was written by **Nick Riasanovsky**, a co-author. What is ours
is the measurement: asking whether the guarantee holds bit for bit across the configurations an
autotuner would actually try, and what the layout pass built on top of it,
`tritongpu-optimize-reduction-layout` (PR #2312), costs and buys.

Both steps are implemented. `inner_tree.bitmatch` asks whether the guarantee holds;
`inner_tree.layout` asks what an optimization built on top of it costs and buys, and is the
longest single step in the artifact.

### `inner_tree.bitmatch` — does the enforced order actually hold?

**Claim.** With `reduction_ordering=inner_tree` the reduction order is fixed by the request, not
by the layout, so configurations that change the layout — `num_warps` above all — return
byte-identical output. With `unordered`, on the same kernels and the same inputs, they do not.
**Both halves are needed.** An `inner_tree` cell that never moves proves nothing on its own: if the
matching `unordered` cell does not move either, the sweep never touched the bits on that kernel and
the row is no evidence in either direction.

The mechanism under test is the one added in commit `053b50f75` (PR #1100), whose own
documentation states the guarantee this step checks: given the same input data and reduction
ordering, the result is bitwise identical regardless of `num_warps`, memory layout, or other
compilation parameters.

**Method.** A **cell** is a fixed `(kernel, dtype, enable_fp_fusion, ordering)`. Inside a cell the
layout axes are swept, every point is compiled and launched on the same five input draws, and the
outputs are grouped by their exact bytes. `n_bit_classes` is the number of groups, and 1 means the
cell is invariant.

| role | axis |
|---|---|
| **swept** — the mode must make these irrelevant | `num_warps` 1, 2, 4, 8, 16, 32 × `num_stages` 1…6 |
| **cell axes** — genuinely bit-relevant, held fixed | `reduction_ordering`, `enable_fp_fusion`, `dtype` |

The two swept axes move **jointly**, all 36 combinations, not one at a time: a mode that survives
each axis alone and breaks on the pair is exactly what a one-axis sweep cannot see.
`enable_fp_fusion` is a cell axis rather than a swept one because it decides FMA contraction below
TTGIR — `inner_tree` fixes the shape of the add tree and says nothing about whether a multiply
folds into an add — so on a mul-fed kernel such as `col_dot` the two settings genuinely differ, and
sweeping it would manufacture a failure that is not the ordering's. **`block_n` is deliberately not
swept, and should not be added:** on a chunked reduction `inner_tree` fixes the order inside one
`tl.sum`, not the accumulation across loop iterations, so the chunk width stays bit-relevant by
design.

The kernels are the 24 (kernel, dtype) pairs `inner_tree.layout` measures — eight synthetic
micro-kernels at f16 and fp8, eight verbatim TorchInductor kernels and one zoo reduction at f32 —
plus two pairs this step adds for itself: `col_max` at f16 and f32, its **control**. That
catalogue of 24 is imported unchanged and does not contain the control; this step appends it in
`control_kernels()`, so 24 plus 2 is the 26 pairs it runs. Max is order-invariant as an operation,
so that cell must come back invariant in *both* arms for reasons that have nothing to do with the
ordering switch. It carries `control = 1`, it is never counted toward the claim, and it is there so
that a harness which cannot report an unmovable kernel as unmovable is caught rather than
believed.

Every byte comparison runs on the wide draw — exponents spread across the dtype's usable range,
alternating signs so the sum cancels. On tame unit-scale data almost any regrouping rounds to the
same bits and the check passes things it should not.

```
export PYTHONPATH=$(git rev-parse --show-toplevel)
export CUDA_VISIBLE_DEVICES=<a gpu>
python artifact_eval/artifact.py --run inner_tree.bitmatch --minutes 600
```

Nothing else has to be set, and the GPU does **not** have to be idle: nothing in this step is
timed, so a neighbour on the same device changes no number. Options are environment variables,
because `artifact.py`'s CLI is shared by every step: `INNER_TREE_BITMATCH_PREFLIGHT=1`
(self-checks, the cell table and the counts, one cell pair, writes nothing),
`INNER_TREE_BITMATCH_KERNELS`, `INNER_TREE_BITMATCH_DTYPES`, `INNER_TREE_BITMATCH_SEEDS`
(default 5), `INNER_TREE_BITMATCH_WARPS`, `INNER_TREE_BITMATCH_STAGES`.

**Cost.** 104 cells × 36 configurations = 3,744 compiles and 18,720 launches; about forty minutes
on a GB300, nearly all of it compiling, because `TRITON_ALWAYS_COMPILE=1` means no build is served
from the disk cache. `--minutes` is a resumable budget, not a sample size: cells stream to
`cache/inner_tree.bitmatch.jsonl` one at a time and a second invocation continues where the first
stopped, so a short run repeated gives the same table as one long run. A cell recorded with a
different seed count or a different swept space is redone rather than reused.

**What you should see.** Three self-checks, all `YES`; a determinism sweep over all 26 of this
step's (kernel, dtype) pairs; the cell table; then one line per (kernel, dtype, `enable_fp_fusion`)
with both arms side by side, a table of which axis moved the bits where they moved, a gate, and a
headline. On an order-sensitive kernel the two arms read `inner_tree` **1** class against
`unordered` **6** classes — six because each `num_warps` value gives its own answer — with
`split_axes = num_warps`. On the fp8 pure sums both arms read 1 class, and those cells are reported
as *no evidence* rather than as a pass.

**How to judge it.** In this order.

1. **`cache_defeat_verified` must be 1.** Triton's in-memory kernel cache is keyed on the
   specialization and the launch options only. Every axis swept here is inside that key, so in
   principle a stale hit cannot happen — but "in principle" is how a table comes back a perfect,
   meaningless *invariant everywhere*. The step clears that cache before every build, sets
   `TRITON_ALWAYS_COMPILE=1` for the on-disk one, and then refuses to measure until it has watched
   two builds that must differ actually produce different TTGIR: once for a layout knob
   (`num_warps` 4 against 8) and once for the ordering itself (`inner_tree` against `unordered` at
   one configuration). The second is the important one — if the ordering argument were being
   dropped, both arms would be the same build and *both* would read invariant.
2. **`sensitive` before `n_bit_classes`.** Where `sensitive` is 0 the matching `unordered` cell
   returned one byte pattern too, so the layout never moved the bits on that kernel and an
   invariant `inner_tree` cell says nothing about the ordering. The gate counts those separately
   and never lets them read as a pass.
3. **`n_bit_classes` = 1 on every `inner_tree` cell whose `sensitive` is 1.** That is the claim.
   **A failing row looks like** `ordering = inner_tree`, `sensitive = 1`, `n_bit_classes` above 1
   and `verdict = NOT INVARIANT`. It also appears in the *which axis moves the bits* table with an
   `inner_tree` ordering, where `split_axes` names the axis responsible and `split_example` gives
   two configurations that disagreed; and the gate line `of those, NOT invariant` is non-zero,
   which makes the run a `FAIL`. The axis is the finding; the class count on its own is not.
4. **`n_ran` against `n_configs`.** A cell where most configurations failed to build is not clean,
   it is empty: with one surviving configuration there is nothing left to disagree. The step
   refuses to call a cell invariant on fewer than two, and the gate counts those as well.
5. **The control.** `col_max` must be invariant in both arms. A control that moves means the
   harness is broken — a partly written output buffer, say — not that the mode failed, and it fails
   the run on its own.

**The headline, and what it does not say.** The last lines read *N cells, of which M were
`inner_tree` cells whose matching `unordered` cell actually differed; K of those M were not
invariant*, followed by the space in full: the axis values, the draws per point, and how many
configurations compiled, failed and were attempted. The phrasing is deliberate. This is a universal
claim, and a pass over these kernels and these axes says only that nothing moved *here*. Read the
space before generalising, and read the *no evidence* cells as untested rather than as passing.

### `inner_tree.layout` — is the layout pass bit-safe, and what does it buy?

**Claim.** `tritongpu-optimize-reduction-layout` — the pass added in commit `8176cccdc`, PR
#2312 — never changes the output bits, and on the reductions it targets it takes back most of
what the ordering constraint costs. Both halves are needed and the first gates the second: a
pass that is fast but moves the bits is useless here, because the whole point of the ordering
switch is that the bits stop moving. **Read `bit_changed` before any speedup. A speedup on a row
whose bytes moved is not a result.**

This step is a re-run of the measurement reported with the pass itself. What differs from that
run is the machine (GB300 / sm_103 instead of H100), the Triton (3.8.0 instead of 3.7.0), the
input dtype on most rows, and one detail of the pass's own source. Each is printed next to the
earlier number, and each row carries a `like_for_like` flag saying whether its dtype matched the
earlier one, so you can see which axis moved.

```
export PYTHONPATH=$(git rev-parse --show-toplevel)
export CUDA_VISIBLE_DEVICES=<an idle gpu>
python artifact_eval/artifact.py --run inner_tree.layout --minutes 600
```

Nothing else has to be set. Options are environment variables, because `artifact.py`'s CLI is
shared by every step: `INNER_TREE_LAYOUT_PREFLIGHT=1` (self-checks, the kernel table, a timing
projection; minutes, writes nothing), `INNER_TREE_LAYOUT_KERNELS`, `INNER_TREE_LAYOUT_DTYPES`,
`INNER_TREE_LAYOUT_SEEDS` (default 10), `INNER_TREE_LAYOUT_BENCH_REPS` (default 3),
`INNER_TREE_LAYOUT_PATIENCE_MIN` (default 45, see below), `INNER_TREE_LAYOUT_REPORT_ONLY=1`
(rebuild the table from the records already in `cache/`; launches no kernel and changes no
number), `INNER_TREE_LAYOUT_FORCE_TIMING=1` (time even on a busy card — for debugging the code
path only, the numbers are not measurements).

**Method.** Three arms per kernel, per dtype, per configuration:

```
base = reduction_ordering=inner_tree, standard pipeline
opt  = the same, plus tritongpu-optimize-reduction-layout{ideal,8,256} appended at end-of-TTGIR
ceil = the same kernel, the SAME configuration, reduction_ordering=unordered, standard pipeline

speedup    = base_ms / opt_ms
gap_closed = (base_ms - opt_ms) / (base_ms - ceil_ms)      1.0 = opt reached ceil
```

The ceiling flips one flag and holds everything else fixed. The experiment has exactly two
variables — is `inner_tree` on, is the pass on — and a ceiling taken as the best unordered time
over the swept configurations would fold an autotuner's freedom into the denominator and make
`gap_closed` mean two things at once. That question is real and it belongs to `gemm.perf.random`
and `gemm.perf.static`, which have an arm for it.

The configuration space is 72 per (kernel, dtype): `num_warps` in 1, 2, 4, 8, 16, 32 ×
`num_stages` 1…6 × `enable_fp_fusion` on/off. Seventeen kernels from three sources, listed in
full by the step before it measures anything:

* eight synthetic reduction micro-kernels from `bitequiv/evaluation/eval_kernels.py`, measured at
  f16 and fp8 (`col_bf16` pins bf16 in its own body);
* eight **verbatim TorchInductor output** kernels from
  `bitequiv/evaluation/realistic_inductor_kernels.py` — every body in that file is a kernel
  `torch.compile` actually emitted, at the shape the model ran at — measured at f32;
* one weight-gradient reduction from the benchmark zoo, `layernorm_bwd_dwdb`, at f32.

The TorchInductor and zoo kernels stay at f32 on purpose. Their bodies are generated code that
computes in f32; narrowing them would mean editing the body, which would cost exactly the
property that makes them worth having. It also gives the comparison four **like-for-like**
kernels — `col_bf16`, `I_bias_grad_dim0`, `J_epilogue_colsum_dim0` and `layernorm_bwd_dwdb` —
where the prior has a number *and* used the same dtype, so a difference there is hardware and
Triton version alone. Every other row differs in dtype too, and the table marks which is which.

Kernel *definitions* are imported from where they live rather than copied, so you can diff them
against their source; every harness detail — input data, launch, timing, byte compare — is
written once in `steps/_inner_tree_kernels.py` and applies to all seventeen the same way.

Two reductions in that Inductor file are **not** measured and the step says so in its output:
`B_layernorm_welford_gather` and `F_cumsum_scan` have no `reduction_ordering` parameter at all —
a Welford combine and a scan are not ordered add-reductions, so `inner_tree` does not apply and
the pass has nothing to preserve. GROUP 2 and later of that file are an inspect-only reference
corpus: hard-coded shapes and references to `torch._inductor` runtime helpers, not runnable as
they stand.

**Cost.** About one hour and forty minutes on an idle GB300 for all 1,728 rows, of which roughly
a quarter is compiling and the rest is `do_bench`. `--minutes` is a resumable budget, not a
sample size: rows stream to `cache/inner_tree.layout.jsonl` one at a time and a second
invocation continues where the first stopped, so a short run repeated gives the same table as
one long run. `INNER_TREE_LAYOUT_PREFLIGHT=1` does the self-checks and one configuration per
kernel in about a minute and projects the rest.

**What you should see.** Four self-checks, all `YES`; a determinism sweep over all 24 (kernel,
dtype) pairs; the kernel table; then a row per (kernel, dtype, `num_warps`) beside the earlier
H100 number, the same medians broken out over all six `num_warps` values, and a gate. The
speedups on the strided-axis kernels are large — `sum_3d_outer` above 4x, `J_epilogue_colsum_dim0`
around 3x on the earlier run — and `gap_closed` sits near 1, meaning the pass reaches roughly
where dropping the ordering constraint entirely would land.

**How to judge it.** In this order.

1. **`pass_available` must be 1.** A build with no binding for the pass produces a completely
   clean table — every arm identical, 1.00x everywhere, zero bits changed — that measured
   nothing, and there is no way to tell it from a real result afterwards. The step aborts on it;
   the column exists so a CSV read later cannot be misread.
2. **The self-check `injection changes the IR` must be `YES`.** Triton's in-memory kernel cache
   is keyed on the specialization and the launch options only, and an injected pass is part of
   neither, so the pass-on build can silently be served the pass-off one — again a perfect,
   meaningless `1.00x, 0 bits changed`. The step clears that cache and sets
   `TRITON_ALWAYS_COMPILE=1` for the on-disk one, then refuses to measure until it has watched
   two builds of one configuration actually produce different TTGIR. A companion check compiles
   the pass at its own documented no-op setting (`strategy=off`) and requires the TTGIR back
   byte-identical to the baseline, so the difference is the transformation and not the extra
   pass-manager run.
3. **`bit_changed` = 0 and `bit_identical_perf` = 1, on every row.** These are two independent
   comparisons: ten order-sensitive draws at the small size, and the large plain-data input the
   timing arm used. **A failing row looks like** `verdict = BITS CHANGED`, and the gate line
   `bit-changed configurations` is non-zero. That is the hard failure; nothing else in the table
   matters if it fires.
4. **`order_sensitive` before `bit_changed`.** Where it is 0 the reduction is order-invariant on
   that configuration, so `bit_changed = 0` proves nothing about the pass. The gate counts those
   rows separately and never lets them read as a pass.
5. **`fired` before any 1.00x.** A kernel at 1.00x with `fired = 0` was declined by a guard, and
   that is the pass working, not a gap. The report lists every kernel it declined on with the
   guard responsible.

**Failures that are expected on this tree, and are not defects.**

*Twelve compile regressions.* `sum_2d_col_big`, f16, `num_warps=4`, all six `num_stages` × both
`enable_fp_fusion` settings: the baseline builds and the optimized build dies in `ptxas`. They
sit exactly on the register-pressure guard's boundary — a 1024×32 tile over 4 warps is
`1024*32/(32*4) = 256` elements per thread, and the guard declines only *above*
`max-elems-per-thread=256`, so a configuration landing precisely on the limit is still
rewritten. The gate reports them and the run is a FAIL because of them. That is the correct
verdict for this tree; it is a real, reproducible boundary case, not a measurement artefact.

*`col_dot` at 1.00x with `fired = 0`.* The prior reported 1.19x for it. This is a difference in
the code, not in the hardware. The pass in this tree skips **every** `arith.mulf`-fed reduce
unconditionally, and no `ttg.enable_fp_fusion` module attribute exists anywhere in the tree — the
follow-up PR #2312's description mentions, which lets the pass optimize a mul-fed reduce when
fusion is off, is not on this branch. The prior's `col_dot` row is even labelled "fusion off",
which only means something with that follow-up present. The step reads the pass source, strips
its comments, and records the answer as the `guard_fp_fusion_gated` column so the table can say
which of the two states it measured.

*Whole fp8 kernels marked order-insensitive.* An fp8 e4m3 value carries a four-bit significand,
so the exact sum of a few hundred of them still fits inside an f32 mantissa and every order gives
the same answer. On those rows `bit_changed = 0` is not weak evidence, it is *no* evidence, and
the report says so rather than counting them. The step measures this rather than asserting it:
it runs the same draw the kernels use through a sequential sum and a pairwise tree on the CPU and
prints the counts — 200 of 200 fp8 draws sum exactly in f32 and 0 of 200 differ, against 0 of 200
and 85 of 200 for f16. `col_exp_sum` and `col_dot` keep their fp8 sensitivity because `exp()` and
`a*b` widen the leaves to f32 before the reduce; those are the fp8 rows worth reading.

*`col_dot` at f16 with about 20% of its output NaN.* The wide-range draw squared overflows f16.
The rows are still order-sensitive, so the check is weakened rather than vacuous, and the gate
prints the saturated fraction per kernel.

*A busy GPU, and a long gap in the log.* If another process is already on the pinned device when
the step starts, it runs the compiles and the bit comparisons, skips the timing arm entirely, and
says so — a device time taken next to somebody else's kernel is not a measurement. Re-running on
a free device fills in the timings for exactly the rows that are missing them and leaves the
finished rows alone.

If a neighbour arrives **mid-run**, the step **pauses** rather than stops. It re-checks every 24
rows; on finding company it prints `PAUSED`, measures and writes nothing until the device is
quiet again, then prints `RESUMED` and continues where it left off. It gives up only if the
neighbour outlasts `INNER_TREE_LAYOUT_PATIENCE_MIN` (45 by default), and the report says how many
times it paused and for how long. **So a long gap in the log is the safety mechanism, not a
hang.** This is why the behaviour exists rather than a simple stop: the machine these numbers
came from has several people on it, the committed run met a third user's profiling job partway
through, and a ninety-minute sweep that dies on a five-minute neighbour needs a person to notice
and restart it. Neither option — recording contaminated times, or losing the run — is acceptable,
so it waits.

## Section 3 — the equivalence checker (`checker.corpus`)

The GEMM sections ask whether one kernel matches one reference. This section asks the broader
question the project is built on: given a static checker that reads compiled IR and decides which
autotuner configurations return identical bits, **is it sound, and how much tuning freedom does it
recover?**

### `checker.corpus` — is the checker sound, and how much does it recover?

**Claim.** Over 51,152 compiled configurations spanning 93 `(kernel, dtype)` groups — reductions,
GEMM in four dtypes, and flash attention — the checker never calls two configurations equal whose
recorded output bytes differ, and on a large part of the space it recovers real tuning freedom
rather than putting every configuration in its own group.

**Method.** This is not a new experiment. It is the measurement of diff **PR #2781**, and its
table is in `prior_results.txt` next to this file, copied verbatim, so you can compare
row for row. The step reads a corpus of already-compiled kernels: for each configuration a `.ptx`
file, and an `empirical_key` recorded when the corpus was built by actually launching that
configuration on 20 random inputs (12 for the realistic-Inductor group, 50 for flash attention)
and hashing the outputs. It runs the checker over every `.ptx` and groups configurations by the
answer; it groups the same configurations by `empirical_key`; then it counts the pairs the two
groupings disagree about, in both directions. The grouping and the counting come from
`bitequiv/evaluation/equivalence_fuzzer.py`, which is standalone and knows nothing about the
checker, so the arithmetic does not depend on the thing being graded.

**The corpus is an input, not a file in this repository.** It is 11 GB — 54,120 PTX files — and it
is regenerated, not archived. The step looks for it at `$CHECKER_CORPUS`, default
`~/bitwise-equiv/local_evaluation/corpus`. Without one it prints how to get one and exits 0. The
scripts that build it ship in `corpus_builder/`; its README gives the commands, and the cost:
**11 GB of disk and the order of a day on a GPU**, because every configuration is compiled and
then launched on every seed. Flash attention is about half of both. `build_local_eval.py` alone
gives the reduction and GEMM groups for roughly 6 GB and a much shorter build, and the step grades
whatever groups it finds, so a partial corpus still produces the rows it can fill.

```
export PYTHONPATH=$(git rev-parse --show-toplevel)
export CHECKER_CORPUS=$HOME/bitwise-equiv/local_evaluation/corpus
python artifact_eval/artifact.py --run checker.corpus
```

No GPU is needed to run the step: the checker reads text. Knobs are environment variables rather
than flags, because `artifact.py`'s CLI is shared by every step — `CHECKER_CORPUS_CAP` (per-group
configuration limit, 0 = all), `CHECKER_CORPUS_KERNELS`, `CHECKER_CORPUS_WORKERS`,
`CHECKER_CORPUS_CHECKER`, `CHECKER_CORPUS_FRESH`. Rows are written as each group finishes, and a
re-run skips groups already recorded at the same cap, so a killed run resumes.

**Cost.** Tens of minutes for the full corpus at 16 workers, CPU only. Flash attention dominates:
roughly 2 seconds per configuration against 0.05 for everything else. To see the shape of the
result in a couple of minutes, cap it or pick a few kernels:

```
CHECKER_CORPUS_KERNELS=softmax,col_max,sum,dot,layernorm python artifact_eval/artifact.py --run checker.corpus
```

**What you should see.** One row per group in `data/checker_corpus.csv` plus three summed rows,
and a printed line per group. The columns to read are `checker_cls` (classes the checker proved),
`empirical_cls` (classes the recorded bytes actually fall into) and `over_merges`. Compare
`checker_cls` against the `after` column of the prior table and `empirical_cls` against its
`empirical` column.

**How to judge it.** **`over_merges` = 0 is the gate**, and it is the only hard one. Anything else
means the checker certified two configurations as identical and the bytes they returned were not,
which is a soundness bug and not a tuning trade-off. The step prints the offending groups by name.

`checker_cls` against `empirical_cls` is the other direction and is a trade-off, not a failure.
Equal means nothing is left to recover. Above means the checker is safe but conservative: those
are configurations an autotuner could have moved between and was not told it could. `recovery`
summarises it in one word, and `fail-closed` — every configuration its own class — is the checker
refusing to answer rather than answering wrongly. Flash attention is almost entirely fail-closed
and the prior table says why: its shared-memory addresses are xor-swizzled, so the checker cannot
prove what a `ldmatrix` reads and keeps the old address-blind behaviour.

Two things will make your numbers differ from the prior table, and neither is a defect.

*A cap.* `CHECKER_CORPUS_CAP` takes a strided sample per group, so its class counts are for the
sample and not for the space. The cap is recorded on every row and named in every note. The prior
table was taken with no cap.

*A corpus you rebuilt yourself.* The corpus is a snapshot of one compiler and one copy of the
kernel sources. Rebuilding it on a different Triton will not reproduce the prior table exactly.
We measured this: 32 configurations drawn across seven groups were recompiled and re-run on the
GPU and compared against their cached entries. 25 came back unchanged; 7 changed — and in all 7
the checker's answer and the output bits changed **together**, never one without the other. All 7
were `reduction_ordering=unordered`, which is the setting that leaves the order to the compiler;
no `inner_tree` configuration in that sample moved. That is 32 configurations, not a sweep, so
read it as "compiler drift on `unordered` reductions happens and the checker tracked it", not as a
guarantee about `inner_tree`. None of it affects the step itself, which never recompiles: it reads
the cached PTX and the cached key, and those two were produced by the same build.

**What this step does not do.** The prior table has a `before` column, the class counts at the
parent commit. Reproducing it needs the parent commit's checker, so it is not something the
artifact can run; it lives in `prior_results.txt` and only there. The same file's fourth
total row, a 72-group subset of the reduction side, is also not reproduced — that subset is a
frozen list which is not in this branch, and guessing at it would produce a number that looks
comparable and is not.

## Files

```
artifact.py        the entry point: shared machinery, step discovery, CLI
steps/             one file per evaluation step; the file name is the step name with . and - as _
  _common.py       the helper library a step reads and never edits; its docstring is the contract
  _inner_tree_kernels.py
                   the seventeen kernels inner_tree.layout measures and the code that compiles,
                   runs and times them; a step-private helper, not shared machinery
  gemm_bitmatch.py ... one module per step, each declaring its own table columns
corpus_builder/    the scripts that build `checker.corpus`'s input; the corpus itself never ships
prior_results.txt
                   the checker tables from the diffs, verbatim; what section 3 is compared against
README.md          this file
AGENTS.md          operational notes for an AI agent driving the artifact; CLAUDE.md points here
data/              committed. Results only, as CSV.
  FORMAT.md        every column of every table, generated from the same declaration as the CSVs
  summary.csv      the headline numbers, one row per experiment and group
  env.csv          the machine and library versions each run was taken on
cache/             not committed. Streaming records. Regenerated, not archived.
```

A step owns its table. The columns are declared in the step's own module and `artifact.py` merges
them into the one declaration that generates both the CSV header and `FORMAT.md`, so implementing a
step — or adding a table to it — is a change to that one file and to nothing shared.

Bulk data — PTX dumps, full configuration sweeps, per-draw tensors — is deliberately not
committed. The script regenerates it; only results are kept. The large per-row CSVs are generated
too, and `summary.csv` is what ships.

## Machine and versions

```
GPU        NVIDIA GB300, compute capability 10.3 (sm_103)
cuBLASLt   13.1.1
Triton     3.8.0+fb, this repository at commit 8f24688b9
PyTorch    2.12.0+cu130
Python     3.12
```

`artifact.py` prints the architecture and cuBLASLt version it actually ran against and warns when
either differs. **This matters more than it usually does.** Which kernel cuBLAS picks is a function
of the architecture *and* the library version, and the bits it returns follow from that choice, so
a row is only comparable to another row with the same two.

## Building

The prebuilt tree should already work. If you rebuild, one setting fails in a way that looks like a
compiler bug rather than a build mistake:

**Build `libtriton.so` with the same compiler that built LLVM.** Mixing them — clang against a
gcc-built LLVM — corrupts the MLIR-to-LLVM translation, and then *every* kernel, down to a trivial
elementwise add, dies with `dyn_cast on a non-existent value` inside `ModuleTranslation`.

```
export LLVM_SYSPATH=$HOME/triton-llvm/llvm-project/build   # the real path, not the ~/.triton cache
export MLIR_DIR=$LLVM_SYSPATH/lib/cmake/mlir               # CMakeLists.txt needs these explicitly
export LLVM_DIR=$LLVM_SYSPATH/lib/cmake/llvm
export JSON_SYSPATH=$HOME/.triton/json                     # required with TRITON_OFFLINE_BUILD=1
export TRITON_OFFLINE_BUILD=1
export CMAKE_CXX_COMPILER=c++                              # gcc, to match the LLVM build
export TRITON_REL_BUILD_WITH_ASSERTS=1                     # match an assertions-enabled LLVM
export MAX_JOBS=100
rm -rf build && pip install -e python
```

`CMAKE_BUILD_TYPE` and `CMAKE_CXX_COMPILER` are cached, so `rm -rf build` is required or a change
is silently ignored. `setup.py` also checks `REL_WITH_DEB_INFO` *before*
`TRITON_REL_BUILD_WITH_ASSERTS`, so leaving the former set overrides the latter without saying so.
`bitequiv/ptx_reduction.py` needs `pyptx==0.1.1`.

## How the timings are taken, and why

Two decisions in the measurement are load-bearing, and both are easy to get wrong if you
re-implement them.

**cuBLAS is timed through a hot closure.** `cublas_matmul` rebuilds the handle, all four layouts,
the preference object and re-runs the heuristic on every call; timing that measures host setup and
can be seven times slower than cuBLAS itself on small shapes. The closure in `artifact.py` sets all
of it up once. It has to keep the heuristic `results` array alive on the object: as a local it is
freed when the constructor returns, the algo pointer dangles, and `cublasLtMatmul` then returns
non-zero and does nothing — which times as roughly a petaflop and looks like a spectacular result.
If you ever see a number like that, suspect the measurement, not the kernel.

**Reported times are device time.** Timing the Python call brackets it with CUDA events on the
stream, so every microsecond the GPU spends idle waiting on the host is counted — and the two arms
are asymmetric there by an order of magnitude, one being a single ctypes call and the other a
Triton launcher. On a 30-microsecond GEMM that measures the launcher. Each call is captured in a
CUDA graph and replays are timed with the L2 cache flushed in between.
