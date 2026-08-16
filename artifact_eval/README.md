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
python artifact_eval/artifact.py --export                      # cache/*.jsonl -> data/*.csv
```

Records stream into `cache/` as they are produced, so interrupting a run loses at most one record
and `--export` still works on whatever was collected. Every run also appends a line to
`data/env.csv` recording the GPU, the cuBLASLt version, and the commit.

## Evaluation steps

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

### `gemm.perf` — bit-exactness costs little

**Claim.** Requiring the output to match cuBLAS bit for bit costs very little performance. This is
a different and much smaller quantity than the gap between Triton and cuBLAS, and the two are
easy to conflate.

**Method.** Eight fixed shapes, three arms each, same inputs and same timing. First, cuBLAS
through a hot closure. Second, our bit-exact GEMM. Third, an ordinary Triton GEMM with no numerics
requirement, autotuned over a 108-configuration space — the ceiling, and roughly what
`torch.compile` would pick. The script also counts how many configurations in that space were
*already* byte-identical to cuBLAS, which says how much of the search space the constraint
actually removes.

**Cost.** About four minutes; there is no time budget to set.

**What you should see.** Per shape: the three times, the ratio, and the count of bit-exact
configurations.

**How to judge it.** The claim rests on `unconstrained_ms / ours_ms` — the best unconstrained
configuration against ours. `ours/cuBLAS` is printed for reference only; it also contains
Triton-versus-cuBLAS, which is a separate and much larger term and is not what this step is
about. The feasibility count matters too: if nearly every configuration is already bit-exact, the
constraint is removing little from the search.

### `gemm.fusion` — placeholder

Prints the measurement design for the fused case; no code yet. Left in the listing so the gap is
visible rather than silently absent.

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

## Files

```
artifact.py        the only entry point
README.md          this file
AGENTS.md          operational notes for an AI agent driving the artifact; CLAUDE.md points here
data/              committed. Results only, as CSV.
  FORMAT.md        every column of every table, generated from the same declaration as the CSVs
  gemm_bitmatch.csv
  gemm_perf.csv
  env.csv          the machine and library versions each run was taken on
cache/             not committed. Streaming records. Regenerated, not archived.
```

Bulk data — PTX dumps, full configuration sweeps, per-draw tensors — is deliberately not
committed. The script regenerates it; only results are kept.

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
