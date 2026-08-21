# Artifact evaluation — reviewer's manual

Everything runs from one script, `artifact.py`. Each claim in the paper is a separate step, so
you can evaluate one at a time rather than all of them.

**On a machine that is not this one, read [Environment](#environment) first.** It says what you
need for each of three levels, and how far you can get without them. The short version: the shipped
records regenerate every table in the paper with no GPU, no build, and not even torch installed.
The quick start below assumes a built tree and a GB300.

## Quick start

Three things, and the first one is the one that catches people.

**Use the interpreter in `.venv/`.** Torch, this Triton and everything else are installed there and
nowhere else. On the machine these numbers came from there is no `python` on `PATH` at all, and the
system `python3` has no torch, so `python artifact_eval/artifact.py` fails before it starts. Every
command below spells the interpreter out; `source .venv/bin/activate` first if you prefer, and then
plain `python` works.

**`bitequiv` is not installed, so the repository root has to be on `PYTHONPATH`.**

**Pin one GPU.** The timing steps assume they own the device. Run `nvidia-smi` first and pick a GPU
nobody else is using — a shared GPU does not change the bit results, but it makes every timing
meaningless.

```
export PYTHONPATH=$(git rev-parse --show-toplevel)
export CUDA_VISIBLE_DEVICES=0
PY=$PYTHONPATH/.venv/bin/python

$PY artifact_eval/artifact.py --list                        # what can be run
$PY artifact_eval/artifact.py --run gemm.bitmatch --minutes 20
$PY artifact_eval/artifact.py --run gemm --minutes 20       # the whole gemm section
$PY artifact_eval/artifact.py --run all --minutes 20        # every section
$PY artifact_eval/artifact.py --export                      # cache/*.jsonl -> data/
```

Nothing else has to be exported. Every step's own options have defaults, and no step needs a
variable set to work — the tables further down are for making a step *shorter*, not for making it
run. The one exception is `checker.corpus`, whose input is an 11 GB corpus that does not ship; it
looks in a default place, and if nothing is there it prints how to get one and exits 0.

`--export` needs no GPU. It rewrites every per-table CSV, then `FORMAT.md`, then `env.csv`, then
`summary.csv` — in that order, each from the one before it, so the four cannot disagree.

Records stream into `cache/` as they are produced, so interrupting a run loses at most one record
and `--export` still works on whatever was collected. Every run also appends a line to
`data/env.csv` recording the GPU, the cuBLASLt version, and the commit. `data/records/` holds those
same streaming records, gzipped, as they stood when the artifact was frozen; its own README says
how to read one and how to unpack them back into `cache/`.

**Every invocation runs with Triton's on-disk kernel cache turned off**, whichever step you asked
for. `inner_tree.bitmatch` and `inner_tree.layout` need `TRITON_ALWAYS_COMPILE=1` — without it a
stale cache hit hands the pass-on build the pass-off binary and their tables come back a perfect,
meaningless `1.00x, 0 bits changed` — and it has to be set before Triton is imported, so both step
modules set it at import time and `artifact.py` imports every step module to build `--list`. It is
therefore on for `gemm.*` too. Nothing measured moves: the flag decides whether a kernel is rebuilt,
not what it computes. What moves is cost — no shape gets a compile free from a previous one, so a
sweep is slower than the same sweep would be with the cache warm, and a fixed configuration-search
budget such as `PERF_RANDOM_SEARCH_S` buys fewer configurations than it otherwise would. Every
number in this artifact was taken under that flag, so the numbers are consistent with each other;
they are not comparable with a run you make with it off.

**All eight steps are implemented.** `--list` marks each one `implemented`; none is a placeholder
any more.

```
gemm.bitmatch        gemm.perf.random     gemm.perf.static     gemm.fusion
gemm.cublas-bug      inner_tree.bitmatch  inner_tree.layout    checker.corpus
```

## Read this before you read a `FAIL`

**Three steps report `FAIL`, or a non-zero mismatch count, and are correct to.** Each of the three
is a finding about the thing under test, not a broken artifact. If you see one of these and stop
there, the documentation has failed you, not the run.

| step | what it reports | why that is the right answer |
|---|---|---|
| `inner_tree.layout` | `RESULT: FAIL`, 12 compile regressions | 12 of 1,728 rows are `sum_2d_col_big`, f16, `num_warps=4` — the optimized build dies in `ptxas` while the baseline builds. That configuration sits exactly on the register-pressure guard's boundary. A real, reproducible bug in the pass. The bit gate itself is clean: 0 bit-changed rows. |
| `inner_tree.bitmatch` | `RESULT: FAIL`, 2 cells not invariant | `D_masked_global_sum` at f32, at **both** `enable_fp_fusion` settings, returns 3 byte classes under `inner_tree`, split by `num_warps`. That is a counterexample to the mode's own guarantee, and finding it is the point of the step. |
| `gemm.bitmatch` | shapes with a differing draw, above 0 | 34 of 62,025 shapes on the committed 90-minute run, and 92 of 153,606 across the union of runs in `data/`. Every single one is `mode=split` at very deep `K`, and 0 of the other eight plan modes produced any. That is cuBLAS's own defect — it sums only the first `K - (K % block_k)` elements — and `gemm.cublas-bug` reproduces it standalone with no Triton in the picture. |

Everything else is expected to come back clean. In particular `checker.corpus` must report
`over_merges = 0`, and the two bit-exact arms of `gemm.perf.*` and `gemm.fusion` must be
byte-identical on every draw. Those are the hard gates.

## Naming, and what a ratio means

Arms are named, never numbered. Section 1 uses `cublas`, `torch_auto`, `torch_triton`,
`bitequiv_autotuning`, `gb300_accelerated`; `gemm.fusion` uses `cublas_unfused`, `torch_fused`,
`bitequiv_autotuner_fused`, `gb300_fused`.

**Every ratio in this artifact is taken against the baseline, and above 1 means faster.** The
baseline is `cublas` for `gemm.perf.*` and `cublas_unfused` for `gemm.fusion`. No table reports a
ratio between two non-baseline arms.

**A ratio between one of our kernels and a torch kernel is not the price of bit-exactness.** They
are different kernels — ours has TMA descriptors, a persistent grid and warp specialization, and
Inductor's `mm_template` has none of them — and that difference dominates whatever the bit
constraint costs. Read such a ratio as what a user gets by switching, and nothing more. The one
place in this project where the bit constraint *is* cleanly isolated is described under
[what bit-exactness costs when nothing else changes](#what-bit-exactness-costs-when-nothing-else-changes).

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

**Cost.** Set by `--minutes`; the default is 20. This step reads no environment variables — its
only knobs are the shared CLI flags. Measured on the two committed 90-minute runs at `--reps 10`:
one covered 62,025 shapes and the other 91,581, so budget roughly 700 to 1,000 shapes per minute
depending on how many of the drawn shapes cuBLAS declines outright (a decline is fast). The knob
that shrinks it is `--minutes`; `--reps` trades draws per shape against shapes.

```
--minutes N   wall-clock budget, default 20. The step stops cleanly at the deadline.
--reps N      independent input draws compared per shape, default 10.
--seed N      shape draw seed, default 20260816. The same seed gives the same shapes.
--max-bytes N skip a shape whose three operands exceed this, default 6 GiB.
```

**What you should see.** A count of shapes byte-identical to cuBLAS, a line per mismatch naming
the shape, the cuBLAS kernel family the plan resolved to and which draws differed, then a table of
all nine plan modes with zeros included, then the declines by reason.

**How to judge it.** Read `data/gemm_bitmatch.csv`; `n_differ` is zero for a byte-identical shape.

**A failing row looks like** `n_differ` above 0, with `draws_that_differ` naming the draws. **And
some of these are expected on this tree.** On the committed 90-minute rerun:

```
62025 shapes x 10 draws = 620250 comparisons
byte-identical to cuBLAS       61991/62025
shapes with any differing draw     34    all of them mode=split, K from 40,976 to 295,504
declined (out of scope)           109
skipped, cuBLAS had no algorithm  477
```

All 34 are `mode=split` at very deep `K`, and 0 of the other eight plan modes produced a single
differing shape. **That concentration is the signature of cuBLAS's own defect**, not of ours: on
those shapes cuBLAS sums only the first `K - (K % block_k)` elements. Run `gemm.cublas-bug`, which
reproduces it standalone with all-ones operands where the exact answer is known, before concluding
otherwise. The step logs mismatches rather than adjudicating them, on purpose — the judgement is
yours. A mismatch on any mode **other** than `split`, or on a shallow `K`, would be a real failure
and is not something this tree produces.

`data/gemm_bitmatch.csv` is the append log of every run, so it holds more rows than any single run
reported: at the export in this commit, 155,310 rows, being the 62,025-shape rerun plus an earlier
91,581-shape run plus 1,704 declines, with 92 differing shapes across the two. The two runs used
different shape-regime code and are not comparable shape for shape; the header of
`steps/gemm_bitmatch.py` says which is which.

### `gemm.perf.random` and `gemm.perf.static` — how fast a bit-exact GEMM is

Two steps, one measurement. `gemm.perf.random` runs it on random shapes and groups by shape
family; `gemm.perf.static` runs it on the fixed layer dimensions of real 2026 open-weight models.
`gemm.perf.static` **imports** every arm, every timing and every byte check from
`gemm.perf.random` rather than re-implementing them, so the two cannot drift apart. What differs
is only where the shapes come from.

**Claim.** A GEMM that must return cuBLAS's exact bytes is a usable kernel, not a toy: it is
byte-identical to cuBLAS on every draw, and its time is in the same range as what an unconstrained
`torch.compile` produces. This is a claim about **speed levels against one baseline**, not about
what the bit constraint costs — see [the isolation
section](#what-bit-exactness-costs-when-nothing-else-changes) for the one measurement in this
project that does answer that.

Five arms per shape, same operands, same timing method, same byte check:

```
cublas               cuBLAS through the hot closure, on the algorithm its own heuristic
                     returns. THE BASELINE AND THE BIT REFERENCE.
torch_auto           torch.compile(mode="max-autotune-no-cudagraphs") as its autotuner picks,
                     extern included. No numerics requirement.
torch_triton         the same with the extern excluded (max_autotune_gemm_backends="TRITON"),
                     so it is Inductor's own Triton template. No numerics requirement.
bitequiv_autotuning  bit-equivalent: the kernel BEFORE the sm_103 rewrites, its configuration
                     chosen by a search under the bit constraint.
gb300_accelerated    bit-equivalent: the GB300 rewrites — TMA, a persistent grid, warp
                     specialization — at the shipped fitted rule.
```

Every reported number is **cuBLAS time over that arm's time, so above 1 means faster than
cuBLAS**. There is deliberately no column comparing two non-baseline arms: a ratio between two of
them hides the level both sit at, and where they differ in more than one way it invites a causal
reading the measurement cannot support. Four numbers on one baseline let a reader take whatever
comparison they want — and it is what made it visible that `torch_auto` and `torch_triton` are
near-identical on every family, i.e. that excluding the extern changes almost nothing.

`gb300_accelerated` against a torch arm is **not** the price of bit-exactness. An unconstrained
autotuner's space contains the bit-exact configurations, so that ratio could not exceed 1 if the
two were the same kernel; it does because they are not. It is a product comparison — what a user
gets by switching. Even the pair of bit-equivalent arms is not a clean isolation: they differ in
the kernel **and** in how its configuration is chosen, two things at once.

These two steps replace an earlier `gemm.perf`, whose unconstrained arm was a Triton configuration
sweep we had written ourselves. A ceiling built out of our own kernel is only as good as the
configuration space we happened to type in, and a reviewer has no reason to trust it. What
`torch.compile` picks with max-autotune is the honest stand-in for what a user gets when they ask
for speed and nothing else.

**Method, and what is pinned.** `bitequiv_autotuning`'s search space follows one rule: *anything
that is part of the cuBLAS plan is pinned, because it defines the answer; launch parameters are
searched.* Pinned are `k_chunk`/`nsplit`, `k_per_dot`, the residue position, `merge_scheme`,
`gemmsn`'s `(S, B)`, `gemv`'s `(V, W, CC, count-down)` and `SPLITK_NUM`. Searched are `BM`, `BN`,
`BK`, `num_warps`, `num_stages`, and `BE` for the gemv families. The winner is then verified
byte-identical over the same draws as the other arms — the pruner is not trusted — and if the
fastest configuration fails the byte check the fastest passing one is used and `fallback` is set
on the row. Three knobs named in the design are *not* searchable because there is no constexpr for
them in the original kernels (`GROUP_M`, the index width, and `G` outside the modes where `BK` is
`G`); the step's own docstring says so at length, and that gap is part of the result.

Five timings are kept raw on every row and none of them is thrown away:
`device_flush_ms` (the headline), `device_warm_ms`, `e2e_ms`, `host_ms`, `floor_ms`. `e2e_ms` and
`host_ms` matter because device time flatters us: the TMA path spends about 100 microseconds of
host time per call building tensor descriptors, which graph replay captures once and a real caller
pays every time.

**Cost.**

*`gemm.perf.random`*: 1,400 shapes at the default — seven sampling families × two dtypes ×
`PERF_RANDOM_PER_CELL` (default 100). Each shape pays a configuration search bounded by
`PERF_RANDOM_SEARCH_S` (default 150 seconds) plus two `torch.compile` autotunes, and every shape
pays its own Triton compilation: `TRITON_ALWAYS_COMPILE=1` is on for every step in this artifact
(see the quick start), so no shape gets a build free from a previous one and the search budget buys
fewer configurations than it would with a warm cache. On this box, three workers on three GB300s
covered 751 of the 1,400 shapes in the first 22 minutes; a single GPU should be given many hours.
Three knobs decide whether this is a twenty-minute run or an overnight one:

| knob | what it does |
|---|---|
| `PERF_RANDOM_PER_CELL` | shapes per (family, dtype) cell, default 100 → 1,400 shapes. Set it to 10 and you measure 140 shapes, a tenth of the work. |
| `PERF_RANDOM_FAMILIES` | comma-separated subset of `any,aligned,unaligned,deepk,smallm,gemv,decode`. One family is a fourteenth of the sweep. |
| `PERF_RANDOM_REPORT_ONLY=1` | print the whole table from the records already in `cache/` and exit. Launches no kernel, measures nothing, costs seconds. This is how you look at a run in progress or re-read a finished one. |

*`gemm.perf.static`*: 698 (model, layer, dtype) cases over 590 distinct (M, N, K, dtype) shapes.
Same per-shape cost as `gemm.perf.random`, so the same knobs apply under `PERF_STATIC_` names. The
committed table was assembled over four invocations between 00:48 and 05:02, two of which were
deliberate re-takes (`PERF_STATIC_REDO_UNBATCHED`, `PERF_STATIC_REDO_CONTENDED`).
`PERF_STATIC_DTYPES=fp16` drops the fp8 half, which is 199 of the 590 shapes.

Both steps accept `--minutes N` as a wall-clock cap and `--minutes 0` to run to completion. Both
claim work one shape at a time through a lock file with a lease, so **extra workers can be started
on other free GPUs at any time** and a worker can be killed without losing or repeating a shape.

**Options.** Neither step has flags of its own — `artifact.py`'s CLI is shared by every step — so
everything is an environment variable. `gemm.perf.random` reads twelve:

| variable | default | what it does |
|---|---|---|
| `PERF_RANDOM_PER_CELL` | 100 | shapes drawn per (family, dtype) cell; 14 cells, so 1,400 shapes. **The main cost knob.** |
| `PERF_RANDOM_FAMILIES` | all seven | comma-separated subset of the sampling families, for a validation slice. **A cost knob.** |
| `PERF_RANDOM_REPORT_ONLY` | unset | `1` = print the table from `cache/` and exit without measuring. **A cost knob: seconds instead of hours.** |
| `PERF_RANDOM_SEARCH_S` | 150 | seconds of configuration search per shape for `bitequiv_autotuning`. The row's `space`, `searched` and `budget_hit` say whether the budget bit. |
| `PERF_RANDOM_MAX_CONFIGS` | 0 (no cap) | hard cap on that search space. A non-zero value is recorded on every row it touched, so a bounded run cannot later be read as full coverage. |
| `PERF_RANDOM_REFINE` | 16 | how many screened configurations are re-timed with a real CUDA-graph replay. |
| `PERF_RANDOM_ROUNDS` | 3 | measurement rounds, best of. |
| `PERF_RANDOM_CUBLASLT` | `13.1.1` | which cuBLAS version's rules to match. The paper's number. Set it to your own only if you know why. |
| `PERF_RANDOM_LIMIT` | 0 (none) | stop after this many shapes **in this process**; the run itself is not finished. |
| `PERF_RANDOM_LEASE_MIN` | 45 | minutes before another worker may steal a claim whose owner has gone quiet. |
| `PERF_RANDOM_REPORT_TXT` | unset | also write the report to this path. `run_perf_random.txt` is what this produced. |
| `PERF_RANDOM_INDUCTOR_LOG` | unset | `1` = keep Inductor's per-shape candidate list. Off by default because 1,400 shapes of it is tens of megabytes that says nothing the records do not carry. Changes nothing measured. |

`gemm.perf.static` reads thirteen, eight of them the same idea under a different prefix:

| variable | default | what it does |
|---|---|---|
| `PERF_STATIC_DTYPES` | `fp16,fp8` | which dtypes to run. `fp16` alone drops 199 of the 590 shapes. **A cost knob.** |
| `PERF_STATIC_PAIRINGS` | all | comma-separated subset of `moe_up,moe_down,mlp_up,mlp_down,lora,lmhead,attn`. **A cost knob.** |
| `PERF_STATIC_REPORT_ONLY` | unset | `1` = print the table from `cache/` and exit. **A cost knob.** |
| `PERF_STATIC_SEARCH_S` | whatever `gemm.perf.random` uses | search budget per shape. Changing it makes the two steps incomparable. |
| `PERF_STATIC_REFINE` | whatever `gemm.perf.random` uses | configurations re-timed with a graph replay. Same warning. |
| `PERF_STATIC_MAX_CONFIGS` | 0 (no cap) | cap on the search space, recorded on every row it touched. |
| `PERF_STATIC_ROUNDS` | 3 | measurement rounds, best of. |
| `PERF_STATIC_CUBLASLT` | `13.1.1` | which cuBLAS version's rules to match. |
| `PERF_STATIC_LIMIT` | 0 (none) | stop after this many shapes in this process. |
| `PERF_STATIC_LEASE_MIN` | 45 | minutes before a stale claim may be stolen. |
| `PERF_STATIC_REDO_UNBATCHED` | unset | `1` = re-measure the shapes taken one call per graph, before batching existed, whose own recorded floor says a batch would move them. Idempotent, so it can be left on across a resumed run. |
| `PERF_STATIC_REDO_CONTENDED` | unset | `1` = re-measure the shapes whose latest rows record `contended`, i.e. another process was on the GPU while they were timed. Also idempotent. |
| `PERF_STATIC_REPORT_TXT` | unset | also write the report to this path. |

**What you should see.** For each step: a per-shape (or per-case) table with all five arms in
microseconds per call and four ratios against `cublas`, then aggregate blocks — by sampling family
and by resolved cuBLAS plan mode for `gemm.perf.random`, by layer group and by (layer group,
dtype) for `gemm.perf.static`. Each aggregate row carries the geometric mean, `[p25 med p75]`,
worst and best with the shape named, a geometric mean of the raw microseconds, a
`byte-identical` fraction, a `near floor` count and an `uncaptured` count.

`gemm.perf.static` finished: 698 cases, 590 shapes, 5,345 arm records. Per layer group, against
`cublas`, above 1 faster:

| layer group | cases | `torch_auto` | `torch_triton` | `bitequiv_autotuning` | `gb300_accelerated` |
|---|---|---|---|---|---|
| `moe_up` | 132 | 0.558 | 0.557 | 0.276 | 0.530 |
| `moe_down` | 144 | 0.589 | 0.588 | 0.289 | 0.567 |
| `mlp_up` | 42 | 0.660 | 0.671 | 0.404 | 0.718 |
| `mlp_down` | 42 | 0.645 | 0.647 | 0.418 | 0.695 |
| `lora` | 168 | 0.765 | 0.766 | 0.708 | 0.651 |
| `lmhead` | 56 | 0.878 | 0.878 | 0.951 | 0.910 |
| `attn` | 114 | 0.624 | 0.629 | 0.417 | 0.686 |

Read the spread, not one number: on `lmhead`, where `K` is 7,168 or more and the GEMM is real
work, every arm is within 10% of cuBLAS and `bitequiv_autotuning` is the closest at 0.951. On the
small MoE expert GEMMs every arm is around half of cuBLAS, and `bitequiv_autotuning`'s fp8 rows
collapse to about 0.15 because the pre-sm_103 kernel makes a `_kcontig` copy of the whole weight
matrix — which is exactly the "it needed new code" gap the `gb300_accelerated` arm exists to show.
`torch_auto` and `torch_triton` agree to within 0.01 on every group, so excluding the extern
changes nothing.

`gemm.perf.random` was being re-measured when this commit was made and its table is partial.
`run_perf_random.txt` is written by the step itself and carries its own coverage line — read that
line before its numbers. Do not read a partial family as a family.

**How to judge it.** In this order.

1. **`byte-identical` must equal the total for `bitequiv_autotuning` and `gb300_accelerated`, on
   every group.** Those two arms are bit-equivalent by construction, so a short count is a bug
   report and their times mean nothing until it is fixed. On the committed `gemm.perf.static` run
   both read full on every layer group — `1320/1320`, `1440/1440`, `420/420`, `420/420`,
   `1680/1680`, `560/560`, `1140/1140`, counted per case, or 5,900 of 5,900 counted once per
   distinct shape. `torch_auto` is unconstrained and is expected to be short (5,470 of 5,900);
   how short is itself a result.
2. **`near floor` and the `F` flag.** A row whose time is under three times `floor_ms` is mostly
   launch overhead and every ratio on it is pulled toward 1. The count is per arm: on
   `gemm.perf.static`'s `lora` group it is 56 of 168 for `cublas`, 31 for the two torch arms, 38
   for `bitequiv_autotuning` and 0 for `gb300_accelerated`. `lmhead` and `attn` carry none at all.
   These rows stay in the geometric means; they are flagged, not dropped.
3. **`batch_n`.** A row with no `batch_n` was taken before calls were batched into one graph, and
   on a small shape it is *not* comparable with a batched row — it reads slower. The `-` entries
   under `calls per graph` count them.
4. **`uncaptured`, `declined`, `error`.** An arm the CUDA graph would not capture, a shape
   `cublas_equivalent_gemm` refuses, or a shape that failed, contributes no time and therefore no
   ratio. Every one is counted and named at the end of the report rather than dropped silently.

**A failing row looks like** an arm whose `bit_ok` is below its `bit_total` in the last two
positions, or a `cublas` time so small the shape reads at hundreds of teraflops. The second is a
measurement failure, not a result: it means the `cublasLtMatmul` call returned non-zero and did
nothing, and the fix is described under [how the timings are
taken](#how-the-timings-are-taken-and-why). **No failing row of either kind is expected on this
tree**, and neither step reports `FAIL` today.

### `gemm.fusion` — fusing an epilogue cuBLAS cannot express

**Claim.** For an epilogue cuBLASLt has no fused form for, an epilogue can be folded into a GEMM
that is *still byte-identical to cuBLAS*, on real model layer dimensions, including the shapes
where cuBLAS splits `K` and Inductor's own template therefore cannot be made to match. Whether
that fused kernel is also **faster** than the two-kernel path is a separate question and the table
answers it per shape rather than in one number.

Four arms, one baseline, one direction of comparison:

```
cublas_unfused            cuBLAS through the hot closure, then the epilogue as its own kernel.
                          THE BASELINE AND THE BIT REFERENCE: what a determinism user gets today.
torch_fused               torch.compile(mode="max-autotune-no-cudagraphs"), no numerics
                          requirement at all. `torch_pick` records what its autotuner chose.
bitequiv_autotuner_fused  bit-exact, fused: the kernel BEFORE the sm_103 rewrites, at a
                          configuration chosen by a search under the bit constraint.
gb300_fused               bit-exact, fused: the GB300 rewrites — TMA, a persistent grid, warp
                          specialization — at the shipped fitted tile rule.
```

Every reported number is **baseline time over that arm's time, so above 1 means faster than
`cublas_unfused`**. There is deliberately no column and no line comparing two non-baseline arms:
a ratio between two of them hides the level both sit at, and the quantity a reader wants is what a
user gains or loses against what they can reach today.

The last two arms are bit-equivalent to cuBLAS **by construction** — the cuBLAS plan defines the
addition order and the kernel implements it — which is the reason to build them. `torch_fused`
cannot be made bit-exact by any mechanism torch offers, and the earlier approach of patching the
epilogue text of Inductor's generated kernel only worked because Inductor's mm template happens to
satisfy the same mainloop contract as plan mode `plain`.

```
export PYTHONPATH=$(git rev-parse --show-toplevel)
export CUDA_VISIBLE_DEVICES=<an idle gpu>
python artifact_eval/artifact.py --run gemm.fusion
```

`--minutes` does not apply: the cost is fixed by the case list. Options are environment variables,
because `artifact.py`'s CLI is shared by every step. There are eight:

| variable | default | what it does |
|---|---|---|
| `FUSION_ONLY` | unset | a substring of `"<model> <layer>"`; runs just the cases that match. **The way to look at one case in a minute instead of the whole list in an hour.** |
| `FUSION_DTYPES` | `fp16,bf16` | which dtypes to run. `fp16` alone halves the 96 rows. **A cost knob.** |
| `FUSION_SEARCH_S` | 45 | seconds of configuration search per row for `bitequiv_autotuner_fused`. **The largest single term in the cost.** `search_space` and `search_seen` on the row say how much of the space that bought. |
| `FUSION_DRAWS` | 10 | input draws per row for the byte check. |
| `FUSION_ROUNDS` | 3 | measurement rounds, best of. |
| `FUSION_MAX_PLAIN` | 35 | cap on the `plain` half of the case selection. Does not bind today: the rule picks 31. |
| `FUSION_SKIP` | unset | comma-separated arm names to leave out of the run. |
| `FUSION_REDO` | 0 | `1` = re-measure rows already in `cache/gemm.fusion.jsonl` instead of resuming past them. |

**Method.** What makes a fused arm equal to the baseline is where it rounds. The unfused path
writes the GEMM's output to memory in the output dtype and reads it back, so the fused kernel has
to round the accumulator through that dtype before the epilogue sees it, and then round again
wherever eager has a kernel boundary:

```python
c = acc.to(out_dtype)            # the round cuBLAS did when it wrote to memory
y = epilogue(c.to(tl.float32))   # the epilogue in fp32, as torch eager does it
store(y.to(out_dtype))
```

The epilogue spellings are written in neither the step nor the kernels. They are
`fusion_oracle/inductor_kernel.EPI_SRC`, verified over their *complete* input domain — `swiglu`,
`rweight` and `resid` over all 4,294,967,296 (accumulator, operand) fp16 pairs, `swiglu_cw` over
120 billion per direction, `lora` over 2^32 pairs at six scales, zero differences everywhere;
`fusion_oracle/probe_pairs.txt` is the table. The kernels take that text and splice it in, so
there is one copy of each spelling and it is the verified one. Two traps are recorded there and
both bite: `tl.minimum`/`tl.maximum` return the *non-NaN* operand where `torch.clamp` keeps the
NaN, and a `.to(fp16)` does not survive the compiler unless the launch passes
`enable_fp_fusion=False` — LLVM contracts the chain into one `fma.rn.f16` and skips the round the
cast asked for.

The fused kernels are `bitequiv/cublas_match/fused_plain.py` and `fused_split.py`, one file per
cuBLAS plan mode. Each holds the corresponding mainloop from `bitequiv/cublas_match/kernels.py`
statement for statement, with only what happens after the loop changed. For `split` even that is
narrower: pass 1 is imported from `kernels.py` rather than copied, and the epilogue goes at the
end of pass 2, so what fusing buys there is one round trip of the `[M, N]` output and nothing else
— the fp32 workspace still has to be written and read, because that is what cuBLAS does.

**The shapes, and the selection rule.** `fusion_moe/models_2026.py` holds 405 reachable
(model, layer, epilogue) cases, every dimension read from that model's own `config.json` on
Hugging Face on 2026-08-16. On sm_103 with cuBLASLt 13.2.2 cuBLAS resolves 361 of them to plan
mode `plain`, 17 to `split`, and declines the other 27 — the `fp8q` cases on `qkv_proj`, whose
output dtype is fp8, which `cublas_equivalent_gemm` refuses because cuBLAS is not being told about
it. A 361-row `plain` table would be one answer repeated, so the step measures:

* **all 17 `split` cases.** They are the only cases in the whole model table where cuBLAS does not
  pick `plain`, and therefore the only ones where the bit constraint reaches past the epilogue and
  into the mainloop. 15 are an MoE expert `up_proj`, 2 a dense `mlp.down_proj`.
* **31 `plain` cases**, one per (model, layer group). Models are taken in the order of the ranking
  fact `models_2026.py` records for each — a trending position first, then a 30-day download count
  descending; the six models that file gives no number for are not drawn from. Within a model the
  layer groups are taken in the order they cost: routed expert FFN, dense FFN, attention output
  projection, LoRA merge. Within a group the representative is the larger token count and, for the
  MoE layers, the even routing that the hot and cold experts bracket.

Each case runs at fp16 and at bf16, so about 96 rows. `fusion_oracle/fusion_cases.py` is that rule
written out at length together with what it leaves out on purpose — **read it before pruning**,
because this list is already a pruning and the axis it pruned hardest, routing balance and token
count, is the one it says least about. The plan mode is resolved on the machine the run happens
on, not shipped in a file: which kernel cuBLAS picks depends on the architecture and the library
version.

**Timing.** As everywhere else in this artifact: device time, CUDA-graph replay, L2 flushed
between replays, `batch_n` calls per graph each on its own operand copy. Five timings are kept raw
per arm and `batch_n` and `floor_ms` are on every row. A row with `others_on_gpu` other than 0 was
taken beside another process and is not a measurement.

**Cost.** Fixed by the case list; `--minutes` does not apply. The committed run took **one hour
nineteen minutes** on an idle GB300 for all 96 rows — its own log stamps the first case at 03:04:14
and the last at 04:23:11. Most of that is compilation — Inductor's autotune for `torch_fused` and
the configuration search for `bitequiv_autotuner_fused` — not running kernels, so a second run over
the same cases is far faster. Records stream to `cache/gemm.fusion.jsonl` and the run resumes from
them. The knobs that shrink it are `FUSION_ONLY` (one case), `FUSION_DTYPES=fp16` (halves the
rows), and `FUSION_SEARCH_S` (45 seconds per row is the largest single term).

**What you should see.** One row per (case, dtype), each with all four arms in it, then geometric
means by plan mode, dtype, layer group and epilogue, all computed at print time from the rows on
disk. `bits` is one character per arm: `.` for byte-identical to the baseline on every draw, `x`
for at least one draw differing, `-` for an arm that did not run.

The run in `run_fusion.txt` — 96 of 96 selected rows, every one with all four arms, none with a
failure recorded against it, `others_on_gpu` 0 and `near_floor` 0 throughout — came out like this.

*The bits.* Both bit-exact arms were byte-identical to `cublas_unfused` on **960 of 960** draws,
on the `split` shapes as much as the `plain` ones, and no configuration the search tried was ever
byte-different (`search_bit_rejected` is 0 on every row). That is the claim, and it holds.

*What torch did.* `torch_fused` matched on **200 of 960** draws — 20 rows of 96, every one of them
`plain` mode carrying `resid` or `rweight`, which are exactly the two epilogues whose *approximate*
spelling was measured byte-exact over all 2^32 operand pairs; and 19 of those 20 are rows where
its autotuner had chosen the **unfused extern**. So it agreed where it did not fuse, on the two
epilogues where fusing changes no bit. On the 34 `split` rows it matched on none: it picked its own
mm template on 26 of them, and that template's mainloop is one fp32 accumulator over the whole `K`
while cuBLAS's is a split, so no amount of epilogue rewriting could have closed it. That is the
concrete reason the two bit-exact arms exist.

*The speed.* Geometric mean over all 96 rows against `cublas_unfused`: `torch_fused` 0.880,
`bitequiv_autotuner_fused` 0.659, `gb300_fused` 0.715. **Fusing a bit-exact epilogue does not, in
general, beat cuBLAS plus a separate kernel on a GB300** — an arm beat the baseline on 31, 8 and 12
of the 96 rows respectively. Two places it does pay: the LoRA merge, where `K` is the rank and the
GEMM is almost nothing (searched arm 1.153 geometric mean over 10 rows, 1.475 at its best), and the
split MoE `up_proj` shapes (`gb300_fused` beats the baseline on 12 of the 34 split rows, up to
1.147). The worst case is a dense `mlp.up_proj` with a SwiGLU gate at 16384 tokens — GLM-4.7-Flash
16384x10240x2048, where `gb300_fused` reads 0.178 at bf16 and 0.190 at fp16 — because there the
epilogue's extra `[M, N]` gate read costs more than skipping one round trip saves. Two caveats
belong with these numbers: the shipped GB300 tile does not fit a fused kernel on 52 of the 62
`plain` rows and was stepped down a pipeline stage or more (the row's `gb300_cfg` says so), and the
search covered a median 42% of its space inside the 45-second budget (`search_space` and
`search_seen` per row).

**How to judge it.** Read the `bits` column before any time on the row. The last two positions —
`bitequiv_autotuner_fused` and `gb300_fused` — must be `.`; **a timing from an arm marked `x` is a
bug report, not a speedup**. The `torch_fused` position is expected to be `x`, and how often it is
`x` is itself a result: that arm is what a user gets when they ask for speed and nothing else, and
it is not byte-identical to cuBLAS on essentially any draw. Then read
`*_over_cublas_unfused`: above 1 is faster than the two-kernel path a determinism user has today.
A `*` on a row marks a baseline under three times the replay floor, where every ratio is
compressed toward 1 and small differences mean nothing.

**A failing row looks like** an `x` in the second or third `bits` position. **No such row is
expected on this tree** and the committed run has none: 960 of 960 draws on both bit-exact arms.
An `x` in the *first* position is normal and expected — 760 of the 960 `torch_fused` draws differ.

### What bit-exactness costs when nothing else changes

Every number in the two sections above is a comparison between **different kernels**. That is the
honest reading of them, and it is why none of them is labelled the price of the bit constraint:
the kernel difference is much larger than the constraint, and a ratio cannot be split into the two
after the fact.

There is exactly one measurement in this project where the constraint is isolated, and it is not
one of the eight steps. It lives in `fusion_oracle/three_way.py` and it works like this:

```
the unconstrained arm   the kernel torch.compile emits for GEMM + epilogue under
                        max-autotune, taken verbatim out of its own output_code.py
the bit-exact arm       THAT SAME SOURCE with only the epilogue arithmetic replaced by the
                        rounding torch eager does. Same mainloop, same tile, same grid,
                        same launch, same number of kernels.
```

Because only the epilogue text differs, the ratio between them is the constraint and nothing else.
Over the 45 cases of the `oracle2` run — every one of which Inductor really did fuse into one
template, and on every one of which the swapped kernel came back byte-identical to the eager
reference on all draws — the unconstrained kernel over the bit-exact one has a **geometric mean of
0.936 on the 38 cases where fusing beat the two-kernel baseline in the first place**, that is, the
bit-exact spelling sits **6.4% below** the unconstrained one. Its range is 0.695 to 1.225, and it
is above 1.0 on 15 of the 38.

Reproduce it with `python fusion_oracle/report.py oracle2`, which prints the whole table from
`fusion_oracle/three_way.jsonl` and needs no GPU. Look for the block headed *THE COST OF
BYTE-EXACTNESS* and the line

```
1/3  bit-exact fused vs unconstrained fused  geomean 0.936  min 0.695  p25 0.766
     median 1.000  p75 1.081  max 1.225   15/38 above 1.0
```

**6.4% is the only number in this artifact allowed to wear that label.** Any other ratio between
one of our kernels and somebody else's is a product comparison.

### `gemm.cublas-bug` — some mismatches are cuBLAS's own

**Claim.** cuBLAS itself returns a wrong answer on some shapes, so a disagreement is not
automatically ours.

**Method.** This step only prints instructions; the reproducer is a standalone script that depends
on nothing in this project — ctypes, torch and `libcublasLt` only — so it can be handed to NVIDIA
unchanged. Run it yourself, once per library you want to test. `A` and `B` are all ones, so every
element of the result must be exactly `K`; nothing rounds, because the products are exact and the
output is fp32.

**Cost.** The step itself prints and exits in under a second. It reads no environment variables
and ignores every CLI flag. The reproducer it points at is three GEMMs and takes seconds; it is
`bitequiv/cublas_match/cublas_gemm_bug_reproduce.py` and takes one variable, `CUBLASLT=`, the path
to the library you want to test.

**What you should see.** On an sm_103 GPU with cuBLASLt 13.1.1 or 13.2.2, part one (`K = 8648`)
comes back as 8640 — eight terms of `1 * 1` missing, which is not a rounding error of any size.
The step prints the three (library, part, `K`) triples observed on this machine:

```
13.2.2   part one    K =  8648 -> 8640   short by 8
         part three  K = 11528 -> 11520  short by 8
13.1.1   part one    K =  8648 -> 8640   short by 8
         part three  K = 11528 -> 11520  short by 8
12.8.5   part two    K = 57608 -> 57600  short by 8
```

**A "failing" run here is a successful reproduction.** The script ends in an assertion, so it
exits non-zero when the defect fires. That is the intended outcome and the reason the step does
not run it for you.

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

`inner_tree.bitmatch` asks whether the guarantee holds; `inner_tree.layout` asks what an
optimization built on top of it costs and buys, and is the longest single step in the artifact.
**Both come back `RESULT: FAIL` on this tree, and both are right to** — read each step's
*failures that are expected on this tree* block before reporting either as broken.

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
because `artifact.py`'s CLI is shared by every step. There are six:

| variable | default | what it does |
|---|---|---|
| `INNER_TREE_BITMATCH_PREFLIGHT` | unset | `1` = run the self-checks, print the cell table and the counts, do one cell pair, write nothing, exit. **About a minute instead of about twenty-five.** |
| `INNER_TREE_BITMATCH_KERNELS` | all 26 pairs | comma-separated kernel names. **A cost knob: one kernel is 4 cells instead of 104.** |
| `INNER_TREE_BITMATCH_DTYPES` | all | comma-separated dtypes. **A cost knob.** |
| `INNER_TREE_BITMATCH_WARPS` | `1,2,4,8,16,32` | the swept `num_warps` values. Narrowing it narrows the claim; the report prints the space it actually covered. |
| `INNER_TREE_BITMATCH_STAGES` | `1,2,3,4,5,6` | the swept `num_stages` values. Same warning. |
| `INNER_TREE_BITMATCH_SEEDS` | 5 | input draws per configuration. A cell recorded at a different seed count is redone, not reused. |

The step also sets `TRITON_ALWAYS_COMPILE=1` for itself. It does **not** rely on
`TRITON_STRICT_REDUCTION_ORDERING`: that variable does nothing for a kernel that takes
`reduction_ordering` as a constexpr, which all of these do, so the ordering is passed explicitly on
every build instead. The report says which of the two it saw.

**Cost.** 104 cells × 36 configurations = 3,744 compiles and 18,720 launches. **About 25 minutes
on a GB300** — the committed run's header is stamped 23:16:29 and its last record landed at
23:41:56 — nearly all of it compiling, because `TRITON_ALWAYS_COMPILE=1` means no build is served
from the disk cache. `--minutes` is a resumable budget, not a sample size: cells stream to
`cache/inner_tree.bitmatch.jsonl` one at a time and a second invocation continues where the first
stopped, so a short run repeated gives the same table as one long run. A cell recorded with a
different seed count or a different swept space is redone rather than reused.
`INNER_TREE_BITMATCH_PREFLIGHT=1` is the knob for a first look.

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

**Failures that are expected on this tree, and are not defects.**

*The run ends in `RESULT: FAIL`, and the two cells that cause it are the point of the step.* The
committed gate reads:

```
cells measured                     104   (96 real, 8 control)
cache defeat verified on every row 1
inner_tree cells                   48
of those, with a sensitive partner 38
of those, NOT invariant            2     (must be 0; this is the claim)
inner_tree cells with no evidence  10
cells where fewer than 2 ran       0
controls not invariant             0
```

The two are `D_masked_global_sum` at f32, one at `enable_fp_fusion=off` and one at `on`. Each
returns **3** byte classes of sizes 24, 6, 6 over its 36 configurations, `split_axes = num_warps`,
with `num_warps=1 num_stages=1` against `num_warps=16 num_stages=1` given as the disagreeing pair.
The matching `unordered` cells split the same way, so this is not a case where the sweep failed to
move anything — the layout genuinely moves the bits there and `inner_tree` does not stop it.
`D_masked_global_sum` is a TorchInductor MSE-loss tail: 608 live elements zero-padded into a 1024
tile and summed down to one scalar. **This is a real counterexample to the guarantee the mode
documents, on a kernel that came out of `torch.compile`, and reporting it is what the step is
for.** A reviewer should check that the two failing cells are these two and no others.

*Ten cells report `no evidence`.* All ten are fp8 pure sums (`col_sum_loop`, `sum_2d_axis0`,
`sum_2d_col`, `sum_2d_col_big`, `sum_3d_outer`, at both `enable_fp_fusion` settings). An fp8 e4m3
value carries a four-bit significand, so the exact sum of a few hundred of them still fits inside
an f32 mantissa and every order gives the same answer. The step measures this rather than
asserting it: on the same draws the kernels use, 200 of 200 fp8 draws sum exactly in f32 and 0 of
200 differ between a sequential sum and a pairwise tree, against 0 of 200 and 85 of 200 for f16.
So those cells are **untested, not passing**, and the gate counts them separately.

*`col_dot` at f16 reports up to 20% of its output NaN or Inf.* The wide-range draw squared
overflows f16. Those cells are still order-sensitive, so the check is weakened rather than vacuous,
and the gate prints the saturated fraction per kernel.

**The headline, and what it does not say.** The last lines read *N cells, of which M were
`inner_tree` cells whose matching `unordered` cell actually differed; K of those M were not
invariant*, followed by the space in full: the axis values, the draws per point, and how many
configurations compiled, failed and were attempted. On the committed run that is *104 cells, of
which 38 ... ; 2 of those 38 were not invariant*, over `num_warps=1,2,4,8,16,32` ×
`num_stages=1..6`, 5 draws per point, 3,744 configurations compiled with 0 build failures. The
phrasing is deliberate. This is a universal claim, and a pass over these kernels and these axes
says only that nothing moved *here*. Read the space before generalising, and read the *no evidence*
cells as untested rather than as passing.

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
shared by every step. There are eight — seven you might want, and one you should not use:

| variable | default | what it does |
|---|---|---|
| `INNER_TREE_LAYOUT_PREFLIGHT` | unset | `1` = self-checks, the kernel table and a projection of the full run's cost, from one configuration per kernel. Writes nothing. **About a minute instead of about a hundred.** |
| `INNER_TREE_LAYOUT_REPORT_ONLY` | unset | `1` = rebuild the whole table from the records already in `cache/`. Launches no kernel, changes no number, needs no free GPU. **Seconds.** |
| `INNER_TREE_LAYOUT_KERNELS` | all 17 | comma-separated kernel names. **A cost knob: one kernel is 72 rows instead of 1,728.** |
| `INNER_TREE_LAYOUT_DTYPES` | all | comma-separated dtypes. **A cost knob.** |
| `INNER_TREE_LAYOUT_SEEDS` | 10 | order-sensitive input draws per row for the bit check. |
| `INNER_TREE_LAYOUT_BENCH_REPS` | 3 | `do_bench` medians per arm; the minimum of them is kept. Lowering it is the cheapest way to trade timing quality for wall clock. |
| `INNER_TREE_LAYOUT_PATIENCE_MIN` | 45 | minutes the step will wait for a neighbour to leave the GPU before giving up. See *a busy GPU* below. |

`INNER_TREE_LAYOUT_FORCE_TIMING=1` also exists. **Do not use it for a result**: it times even on a
busy card, which is for debugging the code path only, and the numbers it produces are not
measurements.

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
speedups on the strided-axis kernels are large — on the committed run `sum_3d_outer` f16 is 4.36x
at `num_warps=4`, `layernorm_bwd_dwdb` f32 is 3.36x and `J_epilogue_colsum_dim0` f32 is 2.62x —
and `gap_closed` sits near 1, meaning the pass reaches roughly where dropping the ordering
constraint entirely would land. The committed gate reads:

```
rows measured                 1728
bit-changed configurations       0   (must be 0)
bit-changed on the perf input    0   (must be 0; an independent second check)
compile regressions             12   (must be 0: baseline built, optimized did not)
baseline did not compile         0
rows carrying an error          12   (the compile regressions above carry theirs)
order-INSENSITIVE rows         432   (25.0% of rows -- the bit check is vacuous on these)
```

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

No GPU is needed to run the step: the checker reads text. `--minutes` does not apply. Knobs are
environment variables rather than flags, because `artifact.py`'s CLI is shared by every step.
There are eight:

| variable | default | what it does |
|---|---|---|
| `CHECKER_CORPUS` | `~/bitwise-equiv/local_evaluation/corpus` | where the corpus is. Without a corpus the step prints how to build one and exits 0. |
| `CHECKER_CORPUS_KERNELS` | all | comma-separated kernel names. **A cost knob: five small kernels finish in a couple of minutes.** |
| `CHECKER_CORPUS_CAP` | 0 (no cap) | configurations per group, taken as a strided sample. **A cost knob, and it changes the meaning of the answer:** class counts are then for the sample, not for the space. The cap is recorded on every row and named in every note. The prior table was taken with no cap. |
| `CHECKER_CORPUS_WORKERS` | 16 | checker process pool size. CPU only. |
| `CHECKER_CORPUS_CHECKER` | `bitequiv.ptx.forward.interp:forward_module_descriptor` | `module:function` of the checker being graded. Point it at another one to grade that instead. |
| `CHECKER_CORPUS_FRESH` | unset | `1` = re-grade groups already in `cache/checker.corpus.jsonl` instead of resuming past them. |
| `CHECKER_CORPUS_RECYCLE` | 5000 | files graded before the worker pool is replaced. A guard against a worker leaking memory over a long run. |
| `CHECKER_CORPUS_STALL` | 600 | seconds without a result before a group's pool is declared dead. It is then reported, not waited on, so one stuck group cannot hang the run. |

Rows are written as each group finishes, and a re-run skips groups already recorded at the same
cap, so a killed run resumes.

**Cost.** Tens of minutes for the full corpus at 16 workers, CPU only; the committed table was
assembled over several resumed invocations. Flash attention dominates: roughly 2 seconds per
configuration against 0.05 for everything else, and it is 9,580 of the 51,152 configurations. To
see the shape of the result in a couple of minutes, cap it or pick a few kernels:

```
CHECKER_CORPUS_KERNELS=softmax,col_max,sum,dot,layernorm python artifact_eval/artifact.py --run checker.corpus
```

**What you should see.** One row per group in `data/checker_corpus.csv` plus three summed rows,
and a printed line per group. The columns to read are `checker_cls` (classes the checker proved),
`empirical_cls` (classes the recorded bytes actually fall into) and `over_merges`. Compare
`checker_cls` against the `after` column of the prior table and `empirical_cls` against its
`empirical` column. The committed run's three summed rows:

```
TOTAL reduction   configs= 9384  checker= 1009  empirical= 614  over_merges=0   (77 groups)
TOTAL mma         configs=41768  checker=11447  empirical= 790  over_merges=0   (14 groups)
TOTAL everything  configs=51152  checker=12456  empirical=1404  over_merges=0   (91 groups)
```

93 groups were graded; the totals cover 91 because two of them — `col_exp_sum` at bf16 and at f16
— graded 0 configurations. All 144 configurations in each of those two failed to build when the
corpus was made, so they are counted as `infeasible` and there was nothing for the checker to
read. A group with `configs = 0` is dropped from the totals rather than counted as a clean 0.

**How to judge it.** **`over_merges` = 0 is the gate**, and it is the only hard one. Anything else
means the checker certified two configurations as identical and the bytes they returned were not,
which is a soundness bug and not a tuning trade-off. The step prints the offending groups by name.
**A failing row looks like** `over_merges` above 0 on any group. **No such row is expected on this
tree**, and the committed run has none: 0 over-merges in every one of the 93 groups.

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
fusion_oracle/     the epilogue-swap experiment and its verified epilogue spellings; `report.py`
                   prints the 6.4% isolation from saved records with no GPU
fusion_moe/        `models_2026.py`, the model layer table `gemm.perf.static` and `gemm.fusion`
                   draw their shapes from
prior_results.txt
prior_results_amd.txt   the same checker on AMD. Recorded verbatim and NOT reproduced here --
                        no AMD corpus and no AMD GPU on this machine.
                   the checker tables from the earlier work, verbatim; what section 3 is
                   compared against
run_*.txt          captured output of one run, so a reviewer can read a result before spending a
                   GPU on it: run_bitmatch, run_perf_random, run_perf_static, run_fusion,
                   run_inner_tree_bitmatch, run_inner_tree_layout, run_checker_corpus. Each is
                   what its own step printed and each carries its own coverage line -- read that
                   line first, because a partial run and a complete one look alike once averaged.
                   Regenerate one by re-running its step; do not hand-edit it
README.md          this file
requirements.txt   the exact wheel set the numbers were taken with, from `pip freeze`; see
                   Environment, level 3. Does NOT cover Triton (build it from this checkout),
                   LLVM (not a Python package) or cuBLASLt (comes from the system CUDA)
AGENTS.md          operational notes for an AI agent driving the artifact; CLAUDE.md points here
data/              committed. Results.
  FORMAT.md        every column of every table, plus the rules for recomputing a table from the
                   rows; generated from the same declaration as the CSVs
  summary.csv      the headline numbers, one row per (experiment, group, metric), each carrying
                   its own `n` against `attempted` so a partial group cannot read as a whole one
  env.csv          the machine and library versions each run was taken on
  records/         every raw record, gzipped, one file per table, 3.1 MB. What summary.csv and
                   the per-table CSVs are computed FROM: the distribution behind a geometric
                   mean, the worst single shape, which model and layer a row belongs to. Its own
                   README lists the files and says how to unpack them back into cache/
cache/             not committed. Streaming records, live. Same content as data/records/ while a
                   run is in progress; the frozen copy is what ships
```

A step owns its table. The columns are declared in the step's own module and `artifact.py` merges
them into the one declaration that generates both the CSV header and `FORMAT.md`, so implementing a
step — or adding a table to it — is a change to that one file and to nothing shared.

Bulk data — PTX dumps, full configuration sweeps, per-draw tensors, the 11 GB checker corpus — is
deliberately not committed. The script regenerates it. The large per-row CSVs are generated too;
`summary.csv` and `env.csv` are the CSVs that ship.

The raw records **are** committed, gzipped, in `data/records/`, and that is a change of mind worth
stating. They were left out while they were regenerable. They stop being regenerable when this
machine goes away, and a paper needs what a few-KB summary cannot carry — the spread behind a
geometric mean, the single worst shape, which model and which layer a row belongs to. 3.1 MB
compressed is a small price for the difference between a number and the evidence for it.

**The CSVs in `data/`, and the records they come from, are append logs, not tables.** A shape
measured twice is in there twice, and a step's own report keeps only the last record per key.
`data/FORMAT.md` states that rule and the others a reader needs to get the same numbers back out;
read it before recomputing anything.

## Environment

No container ships with this artifact, so this section has to carry it. **Find your own row in the
three-level table below before you run anything.** What you can do depends on what you already
have, and the three levels need very different things.

The dependency chain, measured on this machine:

```
the artifact's code
  needs  Triton 3.8.0+fb, built from THIS checkout, not a released wheel
  needs  LLVM 23.0.0git at commit 62b7cf962 -- build tree 197 GB -- BUILD TIME ONLY
  needs  torch 2.12.0+cu130, CUDA 13.0, cuBLASLt 13.2.2
  on     aarch64, NVIDIA GB300, compute capability 10.3 (sm_103), 152 SMs
```

**A released Triton will not do.** The artifact uses `reduction_ordering=inner_tree` and the
`tritongpu-optimize-reduction-layout` pass, which exist only in this fork, and `bitequiv/` lives in
this repository and nowhere else. Triton has to be built from this checkout.

**LLVM is needed to build `libtriton.so`, and not to run it.** That sentence decides how daunting
the rest of this section is. LLVM is the largest cost here — hours, and 197 GB of disk for the
build tree — and it is a one-off: a reviewer who already has a working build never touches LLVM
again, and a reviewer who only wants the tables never touches it at all.

| level | what you can do | what it needs | cost |
|---|---|---|---|
| **1** | recompute every committed table from the shipped records | a Python 3 and this repository | seconds |
| **2** | re-run the experiments | this checkout **already built**, and a GB300 | minutes to hours per step |
| **3** | rebuild the environment from nothing | the LLVM build | hours, and 197 GB of disk |

Level 1 is the one most reviewers will want, and it is the level the artifact fully supports today.

### Level 1 — recompute the tables from the shipped records

`data/records/` holds every raw record every step wrote, gzipped, as they stood when the artifact
was frozen. Unpack them into `cache/`, run `--export`, and every committed CSV regenerates
byte-identically.

```
cd <repo root>
mkdir -p artifact_eval/cache
for f in artifact_eval/data/records/*.gz; do
    case "$f" in *.tar.gz) tar xzf "$f" -C artifact_eval/cache ;;
                 *) gunzip -c "$f" > "artifact_eval/cache/$(basename "${f%.gz}")" ;; esac
done
python3 artifact_eval/artifact.py --export
```

**No GPU, no experiment, no rebuild — and no torch, no Triton and no virtualenv either.** The
export path is plain Python: it reads the JSON-lines records and writes CSV, and never imports
torch or Triton. Checked here two ways — with this box's stock `/usr/bin/python3`, which is 3.9.25
and has no torch installed, and again with `torch` and `triton` deliberately made non-importable.
Both produced all nine CSVs and `FORMAT.md` byte-identical to the committed ones. `PYTHONPATH` is
not needed for `--export` either. The `.venv` interpreter also works and is what
`data/records/README.md` spells out; nothing at this level requires it.

Start here. It regenerates the tables the paper quotes, so every number in the paper can be checked
against the records behind it before spending a GPU on anything.

### Level 2 — re-run the experiments

Needs the checkout **already built**, and a GB300. Everything else is in the
[quick start](#quick-start) and in each step's own section above.

**A different GPU gives different bits, by design and not by error.** Which kernel cuBLAS picks is
a function of the architecture, the SM count and the cuBLASLt version, and the bits it returns
follow from that choice. That is the subject of this artifact, not a defect in it. Every run prints
the architecture and the cuBLASLt version it actually ran against, and warns when either differs
from the fitted pair; see [Machine and versions](#machine-and-versions) below, which also explains
the cuBLASLt warning you will see on every run of this artifact.

The bit results do not need an idle GPU. The timings do — run `nvidia-smi` first and pin a free
device.

### Level 3 — rebuild the environment from nothing

Three things, in this order: the virtualenv, LLVM, then Triton.

**The virtualenv.** `requirements.txt`, next to this README, is `.venv/bin/pip freeze` from the
machine the numbers came from — the exact wheel set, 73 packages. Its header says which lines were
edited and why.

```
python3.12 -m venv .venv
.venv/bin/pip install -r artifact_eval/requirements.txt
```

Python 3.12 is what was used. (`.venv/bin/python` reports 3.12.14+meta; `pyvenv.cfg` records the
3.12.13 it was created with, because the base interpreter was updated in place afterwards. The base
interpreter here lives at a site-specific path that is no use to you; any 3.12 should do.) Two pins
will not come from pip cleanly, and the header of `requirements.txt` says so: `torch==2.12.0` may
give you a CUDA build other than the `+cu130` that was used — check that `torch.version.cuda` reads
13.0 — and `pyptx==0.1.1` was installed from a local aarch64 wheel. If pip cannot find pyptx, only
`checker.corpus` is affected: it is the only step whose imports reach it, and that step needs an
11 GB corpus that does not ship anyway.

cuBLASLt is not a wheel and is not in `requirements.txt`. `bitequiv/cublas_match/ltapi.py` globs
`/usr/local/cuda*/lib64` and `/usr/local/cuda*/targets/*/lib` and takes the newest by file name; on
this box that is 13.2.2.2, under `/usr/local/cuda-13.1/`. The `nvidia-cublas` wheel in
`requirements.txt` is 13.1.1.3 and is not what gets loaded.

**LLVM.** Build time only. `llvm-config --version` reads `23.0.0git`, at commit
`62b7cf9623fc310525f39ed69aaecc318a909731`. The build tree measures 197 GB, so check your disk
before starting. Build it with gcc, and read the first warning below before choosing otherwise.

**Triton.** Then, from the repository root:

```
export LLVM_SYSPATH=$HOME/triton-llvm/llvm-project/build   # the real path, not the ~/.triton cache
export MLIR_DIR=$LLVM_SYSPATH/lib/cmake/mlir               # CMakeLists.txt needs these explicitly
export LLVM_DIR=$LLVM_SYSPATH/lib/cmake/llvm
export JSON_SYSPATH=$HOME/.triton/json                     # required with TRITON_OFFLINE_BUILD=1
export TRITON_OFFLINE_BUILD=1
export CMAKE_CXX_COMPILER=c++                              # gcc, to match the LLVM build
export TRITON_REL_BUILD_WITH_ASSERTS=1                     # match an assertions-enabled LLVM
export MAX_JOBS=100
rm -rf build && .venv/bin/pip install -e .
```

`setup.py` and `pyproject.toml` are at the repository root, and the Python package itself is under
`python/triton/`; the editable install this box was built with records
`file:///home/youngzt/bitwise-equiv/triton`, the root. So the last line is `-e .` from the root, not
`-e python`.

Two warnings, each of which cost hours to find.

**Build `libtriton.so` with the same compiler that built LLVM.** Mixing them — clang against a
gcc-built LLVM — corrupts the MLIR-to-LLVM translation, and then *every* kernel, down to a trivial
elementwise add, dies inside `ModuleTranslation` with `dyn_cast on a non-existent value`. It looks
exactly like a compiler regression in the fork, and it is not one; there is nothing to bisect.

**`CMAKE_BUILD_TYPE` and `CMAKE_CXX_COMPILER` are cached**, so changing either is silently ignored
unless the build directory is removed first — that is what the `rm -rf build` above is for. And
`setup.py` checks `REL_WITH_DEB_INFO` *before* `TRITON_REL_BUILD_WITH_ASSERTS`, so leaving the
former set overrides the latter without saying so.

### Two things that are already decided for you

**The interpreter is `.venv/bin/python`.** Torch, this Triton and everything else are installed
there and nowhere else. There is no `python` on this box's `PATH` at all, and the system `python3`
is 3.9 with no torch, so a bare `python artifact_eval/artifact.py` fails before it starts. That is
why every command in this file spells the interpreter out. Level 1 is the one exception: `--export`
needs none of it.

**`TRITON_ALWAYS_COMPILE=1` is in force for every step**, not only the two that want it. The two
`inner_tree` step modules set it at import, and `artifact.py` imports every step module to build
`--list`, so it is on for `gemm.*` as well. Nothing measured moves — the flag decides whether a
kernel is rebuilt, not what it computes — but sweeps cost more, and a fixed search budget such as
`PERF_RANDOM_SEARCH_S` buys fewer configurations than it otherwise would. All the shipped data was
taken under it, which is why it was documented rather than changed. The
[quick start](#quick-start) explains it at length.

### Machine and versions

```
GPU        NVIDIA GB300, compute capability 10.3 (sm_103), 152 SMs
CPU/OS     aarch64, CentOS Stream 9
cuBLASLt   13.2.2 as loaded at run time, on all 83 invocations recorded in data/env.csv
           13.1.1 is what the arch profile was FITTED against, so every run prints a warning
           loaded from /usr/local/cuda-13.1, not from a wheel
CUDA       13.0, as torch reports it
Triton     3.8.0+fb, this repository, branch artifact-eval-submission
LLVM       23.0.0git at 62b7cf962 -- build time only, not needed to run
PyTorch    2.12.0+cu130
Python     3.12
```

`artifact.py` prints the architecture and cuBLASLt version it actually ran against and warns when
either differs from the fitted pair. **This matters more than it usually does.** Which kernel
cuBLAS picks is a function of the architecture *and* the library version, and the bits it returns
follow from that choice, so a row is only comparable to another row with the same two.

**So you will see a cuBLASLt warning on every run in this artifact, and it is expected.** The box
carries 13.2.2 and the recipe in `bitequiv/cublas_match/arch.py` was measured on 13.1.1. The
warning says the pair was not measured, not that the result is wrong — and in fact the bits still
came out identical: `gemm.perf.static` was byte-identical on 5,900 of 5,900 draws on each of its
two bit-exact arms, over 590 distinct shapes, and `gemm.fusion` on 960 of 960. Silencing it would
mean adding a `((10, 3), (13, 2))` entry to that file's registry, which nobody should do on the
strength of one box. `data/env.csv` records the loaded version for every invocation, so no row is
ambiguous.

Each run also records the exact repository commit it ran at; the runs behind this commit span
several, because steps were being finished while others were measuring. `data/env.csv` has the
list.

## How the timings are taken, and why

Three decisions in the measurement are load-bearing, and all of them are easy to get wrong if you
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
Triton launcher. On a 30-microsecond GEMM that measures the launcher. The call is captured in a
CUDA graph and replays are timed with the L2 cache flushed in between.

**A small kernel is timed several calls at a time.** A graph replay costs about 6.9 microseconds
on a GB300 before any work of ours runs. That is 2% of a 289-microsecond `lm_head` and 77% of a
2-microsecond MoE expert GEMM — and both arms pay the same 6.9, so a true 2x on the small one
comes out as 1.22x. Those rows are not noisy, they are flattened toward 1, which reads as "the
arms perform about the same" when nothing of the kind was measured. So `graph_ms` captures
`batch_n` calls into one graph and divides; `pick_batch` sizes it from a probe so that the fixed
part of the replay is about 5% of a per-call time, and a kernel already far above the floor keeps
`batch_n = 1` — the same measurement as before, so those rows did not move. Every row records its
`batch_n`.

The batch runs on `batch_n` **distinct operand copies**, one per call. The flush happens once per
replay rather than per call, so a batch over one operand set would leave calls 2..N reading a warm
L2 — measured at 1.7x to 2.5x faster on a MoE expert weight, which fits in this card's 129 MiB L2.
A copy per call means no call can leave anything behind for the next, and every call starts cold
exactly as it does at `batch_n = 1`: the batch changes what the floor costs, not what is timed.
Flushing inside the graph instead is not an option — evicting a 129 MiB L2 takes a memset of about
16 microseconds, a larger floor than the 6.9 being removed, and subtracting it back off is a
difference of two large numbers where the answer is small.

What is left after batching is about 1 microsecond per call: graph nodes run one after another and
that is what one empty kernel costs, so it is a floor a real caller pays too. `floor_ms` is that
empty kernel measured at the row's own `batch_n`, and `near_floor` still means "under three times
the floor". A GEMM as small as an 80x2048x8 LoRA merge stays flagged, and should — it is not
distinguishable from a launch.
