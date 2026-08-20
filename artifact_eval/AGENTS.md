# Driving this artifact with an agent

You are helping a reviewer reproduce the measurements in this artifact. `README.md` describes what
is being evaluated and why; this file is the operational half — environment, build, run, interpret.
You should not need to write or modify any kernel, and you should not need to edit any file here.

## 1. Environment

```
cd <repo root>
git rev-parse --abbrev-ref HEAD          # should be artifact-eval-submission
nvidia-smi                                # pick a GPU nobody else is using
```

Three things, and the first one is the one that catches people:

```
export PYTHONPATH=$(git rev-parse --show-toplevel)   # bitequiv is not installed
export CUDA_VISIBLE_DEVICES=0                        # a free one
PY=$PYTHONPATH/.venv/bin/python                      # torch and triton live ONLY here
$PY artifact_eval/artifact.py --list
```

**Use `.venv/bin/python`.** There is no `python` on `PATH` on this box and the system `python3` has
no torch, so a bare `python artifact_eval/artifact.py` fails before it starts.

Pin one GPU. A shared GPU does not change the bit results, but it makes every timing meaningless,
and two of the steps refuse to time on a busy card rather than record a bad number.

Python 3.12, PyTorch 2.12.0+cu130, Triton 3.8.0+fb. `bitequiv/ptx_reduction.py` needs
`pyptx==0.1.1`. Nothing else has to be installed.

Nothing else has to be exported either. Every step option has a default and no step needs a
variable set in order to run; the tables in `README.md` are for making a step *shorter*.
`checker.corpus` is the one step with an input that does not ship, and with no corpus it prints how
to get one and exits 0 rather than failing.

One thing is set for you and you cannot avoid it: importing the step modules turns on
`TRITON_ALWAYS_COMPILE=1` for the whole process, so **every** step runs with Triton's on-disk
kernel cache off. The two `inner_tree` steps need it or their gates are meaningless. It changes no
measured value, only cost. `README.md`'s quick start explains it in full.

## 2. The eight steps, and what each costs

All eight are implemented; there are no placeholders left. `--list` prints them.

| step | GPU? | cost at the default | the knob that shrinks it |
|---|---|---|---|
| `gemm.bitmatch` | yes | `--minutes`, default 20. 700–1,000 shapes/minute at `--reps 10` | `--minutes` |
| `gemm.perf.random` | yes, idle | 1,400 shapes; hours on one GPU from a cold compile cache | `PERF_RANDOM_PER_CELL`, `PERF_RANDOM_FAMILIES`, or `PERF_RANDOM_REPORT_ONLY=1` for no measurement at all |
| `gemm.perf.static` | yes, idle | 698 cases over 590 shapes; same per-shape cost as above | `PERF_STATIC_PAIRINGS`, `PERF_STATIC_DTYPES=fp16`, `PERF_STATIC_REPORT_ONLY=1` |
| `gemm.fusion` | yes, idle | fixed at 96 rows; 1 h 19 min on the committed run | `FUSION_ONLY`, `FUSION_DTYPES=fp16`, `FUSION_SEARCH_S` |
| `gemm.cublas-bug` | no | prints instructions, under a second | — |
| `inner_tree.bitmatch` | yes, may be busy | 3,744 compiles; about 25 minutes | `INNER_TREE_BITMATCH_PREFLIGHT=1` |
| `inner_tree.layout` | yes, idle | 1,728 rows; about 1 h 40 min | `INNER_TREE_LAYOUT_PREFLIGHT=1`, `INNER_TREE_LAYOUT_REPORT_ONLY=1` |
| `checker.corpus` | no | tens of minutes at 16 workers, CPU only. Needs an 11 GB corpus it does not ship | `CHECKER_CORPUS_KERNELS`, `CHECKER_CORPUS_CAP` |

Every option is an environment variable, because `artifact.py`'s CLI is shared by every step.
`README.md` has a full table per step: what each variable does, its default, and what a reviewer
would plausibly set it to. **Read that table before choosing a value; do not guess a variable
name.** The step's own module docstring is the same information in the source.

## 3. Running

```
python artifact_eval/artifact.py --run gemm.bitmatch --minutes 20
python artifact_eval/artifact.py --run gemm --minutes 20     # the whole gemm section
python artifact_eval/artifact.py --run all --minutes 20      # every section
python artifact_eval/artifact.py --export                     # cache/ -> data/
```

`--minutes` bounds the sampling steps. On `gemm.bitmatch` it is a sample size: longer is better
evidence. On `inner_tree.*` and `gemm.perf.*` it is a **resumable budget**, not a sample size —
records stream to `cache/` one at a time and a second invocation continues where the first
stopped, so a short run repeated gives the same table as one long run. `--minutes 0` on the
`gemm.perf.*` steps means run to completion. `gemm.fusion` and `checker.corpus` ignore it.

Interrupting a run loses at most one record and `--export` still works on what was collected.

**Adding workers.** `gemm.perf.random` and `gemm.perf.static` claim work one shape at a time
through a lock file with a lease, so you can start another worker on any other free GPU at any
time, and kill one, without losing or repeating a shape. This is the fastest way to finish a perf
sweep. `touch cache/gemm.perf.random.PAUSE` stops the workers cleanly at the next shape boundary;
delete it and re-run the same command to resume.

## 4. Reading the output

**Every ratio is against the baseline and above 1 means faster.** The baseline is the `cublas` arm
for `gemm.perf.random` and `gemm.perf.static`, and `cublas_unfused` for `gemm.fusion`. Arms are
named, never numbered: `cublas`, `torch_auto`, `torch_triton`, `bitequiv_autotuning`,
`gb300_accelerated`, and for fusion `cublas_unfused`, `torch_fused`, `bitequiv_autotuner_fused`,
`gb300_fused`. If you see a numbered label like `4/2b` you are reading a superseded report.

**Check the byte columns before any time.** On `gemm.perf.*` that is `bit_ok` against `bit_total`
for `bitequiv_autotuning` and `gb300_accelerated`; on `gemm.fusion` it is the last two positions
of the `bits` column, which must be `.`. Those arms are bit-equivalent to cuBLAS by construction,
so a short count is a bug report and their times mean nothing until it is fixed. The `torch_*`
arms are unconstrained and are **expected** to be short.

**Do not call any of these ratios the price of bit-exactness.** Our kernels and torch's are
different kernels — ours has TMA descriptors, a persistent grid and warp specialization, and
Inductor's `mm_template` has none of them — and the kernel difference dominates whatever the bit
constraint costs. Report such a ratio as what a user gets by switching. The only isolated
measurement of the constraint is the 45-case epilogue-swap run described in `README.md`, at 6.4%;
reproduce it with `python fusion_oracle/report.py oracle2`, which needs no GPU.

**Read the coverage line before the numbers.** Every report prints how much of its space it
actually covered. A partial sweep and a complete one look identical once averaged.

## 5. Three steps report a failure, and are right to

**A `FAIL` here is not automatically a broken artifact.** Before you report one, check it against
this list. All three are findings about the thing under test.

| step | expected | why |
|---|---|---|
| `inner_tree.layout` | `RESULT: FAIL`, `compile regressions 12` | 12 of 1,728 rows are `sum_2d_col_big` f16 `num_warps=4`, where the optimized build dies in `ptxas` and the baseline builds. Exactly on the register-pressure guard's boundary. The bit gate itself is clean: 0 bit-changed. |
| `inner_tree.bitmatch` | `RESULT: FAIL`, `of those, NOT invariant 2` | `D_masked_global_sum` at f32, at both `enable_fp_fusion` settings, returns 3 byte classes split by `num_warps`. A genuine counterexample to the mode's guarantee. |
| `gemm.bitmatch` | 34 differing shapes of 62,025 on the committed run | every one `mode=split` at very deep `K`. cuBLAS's own defect; `gemm.cublas-bug` reproduces it standalone. |

Anything **outside** that list is a real problem and worth reporting: a bit-changed row in
`inner_tree.layout`, a third non-invariant cell in `inner_tree.bitmatch`, a `gemm.bitmatch`
mismatch on a mode other than `split`, a non-zero `over_merges` in `checker.corpus`, or a short
byte count on a bit-exact arm.

`gemm.cublas-bug`'s reproducer ends in an assertion, so **exiting non-zero is a successful
reproduction**, not a failure.

## 6. If something else looks wrong

**A cuBLASLt version warning.** Expected on every run in this artifact. The box carries 13.2.2 and
the recipe was fitted on 13.1.1; the warning says that pair was not measured, not that the answer
is wrong, and in fact the bits still came out identical everywhere. Do not "fix" it by editing
`bitequiv/cublas_match/arch.py`. `data/env.csv` records the loaded version for every invocation.

**An architecture warning.** The measurements are from an NVIDIA GB300 (sm_103). On other hardware
both the bit results and the speeds will differ, and the defect reproducer's three `K` values fire
on different (architecture, version) pairs — none fires everywhere.

**A timing that looks impossibly fast.** Suspect the measurement, not the kernel. If cuBLAS appears
to run at hundreds of teraflops on a small shape, the call is failing silently and being timed as a
no-op. Report it rather than recording the number.

**A long gap in `inner_tree.layout`'s log.** That is the safety mechanism, not a hang. If another
process appears on the pinned GPU mid-run the step prints `PAUSED`, measures and writes nothing
until the card is quiet, then prints `RESUMED`. It gives up only after
`INNER_TREE_LAYOUT_PATIENCE_MIN` minutes, 45 by default.

**A step fails to compile a kernel.** Almost always the build issue in section 7.

**`checker.corpus` prints how to get a corpus and exits 0.** It found no corpus at
`$CHECKER_CORPUS`. That is not a failure; building one is 11 GB and the order of a day on a GPU.
`corpus_builder/build_local_eval.py` alone gives the reduction and GEMM groups for about 6 GB, and
the step grades whatever groups it finds.

## 7. Building, if you have to

The prebuilt tree should already work. If you rebuild, one setting fails in a way that looks like a
compiler bug rather than a build mistake:

**Build `libtriton.so` with the same compiler that built LLVM.** Mixing them — clang against a
gcc-built LLVM — corrupts the MLIR-to-LLVM translation, and then *every* kernel, down to a trivial
elementwise add, dies with `dyn_cast on a non-existent value` inside `ModuleTranslation`. It is not
a Triton bug and there is nothing to bisect. `CMAKE_BUILD_TYPE` and `CMAKE_CXX_COMPILER` are
cached, so `rm -rf build` is required or a change is silently ignored, and `setup.py` checks
`REL_WITH_DEB_INFO` before `TRITON_REL_BUILD_WITH_ASSERTS`, so leaving the former set overrides the
latter without saying so. The full recipe is in `README.md`.

## 8. Please do not

Do not modify `artifact.py`, `steps/`, `bitequiv/`, or anything under `data/`. Do not hand-edit a
`run_*.txt`: each one is the captured output of its own step and the step regenerates it. If a
measurement seems wrong, report what you saw and the contents of `data/env.csv` rather than
adjusting the script to make it look right.

`cache/` is live and is not committed. The frozen copy of it that ships is `data/records/`, gzipped
— see that directory's README. Do not overwrite `data/records/` with a `cache/` you have added your
own smoke-test rows to; if you are testing, copy `cache/` aside first and put it back, or move it
aside and let the run start from an empty one.
