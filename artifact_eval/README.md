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

**Method.** Random shapes are drawn from four regimes — square, thin, very deep `K`, and
decode-sized — in fp16 and fp8, with half the fp16 draws rounded to multiples of 16 and fp8 always
so (cuBLAS refuses fp8 otherwise). Deep `K` is in the mix deliberately: it is the only place
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

Both steps are placeholders. Neither needs a new harness: `bitequiv/evaluation/` already contains
the rulers they call.

### `inner_tree.bitmatch` — does the enforced order actually hold? — *placeholder*

**Claim.** With `reduction_ordering=inner_tree` the reduction order is fixed by the request, not
by the layout, so configurations that change the layout — `num_warps` above all — return
byte-identical output. With `unordered` they do not.

Sweep the layout-changing axes twice, once per ordering, and count the distinct byte outputs. The
`unordered` run is the control and matters as much as the `inner_tree` run: a kernel whose bits
never depended on the layout proves nothing either way. `TRITON_STRICT_REDUCTION_ORDERING` is read
when triton is imported, so setting it from Python after `import triton` silently does nothing and
the run looks like a clean pass — every row records whether it was actually set.

### `inner_tree.layout` — is the layout pass bit-safe, and what does it buy? — *placeholder*

**Claim.** `tritongpu-optimize-reduction-layout` never changes the output bits, and on the
reductions it targets it makes them faster. Both halves are needed: a pass that is fast but moves
the bits is useless here, because the whole point of the ordering switch is that the bits stop
moving.

`bitequiv/evaluation/evaluate_opt.py` is exactly this ruler and is deliberately pass-agnostic —
you name the pass on the command line and it compiles every in-scope configuration without and
with it, answering three independent questions: does it still compile, do the bits change, is it
faster. Read `bit_changed` before the speedup; a speedup on a row whose bits changed is not a
result.

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
