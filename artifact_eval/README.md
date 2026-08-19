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
`reduction_ordering` / `inner_tree` (D100027220) and the `TRITON_STRICT_REDUCTION_ORDERING`
environment variable (D101872700) — was written by **Nick Riasanovsky**, a co-author. What is ours
is the measurement: asking whether the guarantee holds bit for bit across the configurations an
autotuner would actually try, and what the layout pass built on top of it,
`tritongpu-optimize-reduction-layout` (D110978754), costs and buys.

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

## Section 3 — the equivalence checker (`checker.*`)

The GEMM sections ask whether one kernel matches one reference. This section asks the broader
question the project is built on: given a static checker that reads compiled IR and decides which
autotuner configurations return identical bits, **is it sound, and how much tuning freedom does it
recover?** There are two checkers and they are peers, not a chain — the PTX one
(`bitequiv/ptx_reduction.py`) reconstructs the floating-point reduction tree from the PTX, and the
TTGIR one (`bitequiv/ttgir_reduction.py`) reads the association order out of the parsed MLIR with
the compiler's own layout machinery. Every step here runs both.

All four steps are placeholders, and all four mirror a stage that
`bitequiv/evaluation/evaluate.py` already runs; the work is to drive it from the artifact and
record its numbers as a table, not to design a new experiment. `bitequiv/evaluation/README.md`
describes the framework.

The ground truth throughout is the empirical fuzzer in
`bitequiv/evaluation/equivalence_fuzzer.py`: launch a configuration on many random seeds and group
configurations by the output bytes. A fuzzer can only ever *refute* equivalence, never prove it, so
more seeds is stronger evidence and never certainty.

### `checker.precision` — is the checker sound? — *placeholder*

**Claim.** The checker never calls two configurations equal that actually return different bytes,
and it recovers a useful part of the tuning freedom rather than splitting every configuration into
its own group.

**Over-merges must be 0** — that is the gate. Over-splits are the opposite direction: safe, but
recovery left on the table. The TTGIR checker is *expected* to over-merge and that is the finding,
not a defect being hidden: TTGIR fixes the association order but is blind to FMA contraction, which
is decided below TTGIR and gated by `enable_fp_fusion`, so on the multiply-fed kernels it merges
configurations whose bits differ. That is precisely the gap the PTX checker closes.

### `checker.performance` — what does staying inside one certified set cost? — *placeholder*

**Claim.** Restricting an autotuner to one checker-certified set still leaves it real choices, and
the fastest configuration it can then pick is close to the fastest available with no numerics
requirement at all.

Two ratios come out and they must not be confused. The freedom inside the set says whether the
question is interesting at all — a set of one configuration has no choice to make. The best member
against the global ceiling is the price of the constraint.

### `checker.regpressure` — does the verdict survive `ptxas`? — *placeholder*

**Claim.** The verdict is not an artefact of reading PTX. Configurations the checker certifies stay
byte-identical after `ptxas` has allocated registers and spilled.

A `.maxnreg` cap leaves the PTX body identical, so the checker puts the capped and uncapped builds
in the same group by construction and only `ptxas` behaves differently. Check that members actually
spilled before believing a clean result: if none did, the caps were too loose and the step
exercised nothing.

### `checker.korder` — which GEMM knobs move the bits? — *placeholder*

The one step here whose verdict does not come from the checker. Two configurations differing in
exactly one axis are compiled and run on the same input and compared byte for byte, which is the
ground truth a checker has to agree with. A `DIFFER` on split-K is the expected answer, not a
failure. Read what each arm actually lowered to before the verdict — if both arms took the same
path, `BIT-IDENTICAL` is trivially true and the row asked nothing.

`evaluate.py` also has a `cublas` stage, a stub there on purpose: comparing a Triton GEMM against a
cuBLAS reference is `gemm.bitmatch` above, at a far larger scale. There is deliberately no
`checker.cublas` step.

## Files

```
artifact.py        the entry point: shared machinery, step discovery, CLI
steps/             one file per evaluation step; the file name is the step name with . and - as _
  _common.py       the helper library a step reads and never edits; its docstring is the contract
  gemm_bitmatch.py ... one module per step, each declaring its own table columns
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
