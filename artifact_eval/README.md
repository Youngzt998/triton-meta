# Artifact evaluation

Everything is driven by one script, `artifact.py`. Pick what to run with `--run`; each step
appends its records to `cache/`, and `--export` turns those into the CSVs under `data/` that the
paper's tables are built from.

```
python artifact.py --list                          # what can be run
python artifact.py --run gemm.bitmatch --minutes 20
python artifact.py --run gemm --minutes 20         # every gemm step
python artifact.py --export                        # cache/*.jsonl -> data/*.csv + FORMAT.md
```

Two things must be set on every invocation. `bitequiv` is not installed, so the repository root
has to be on `PYTHONPATH`; and one GPU has to be pinned, because the timing steps assume they
have the device to themselves.

```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(git rev-parse --show-toplevel) \
    python artifact_eval/artifact.py --run gemm.bitmatch --minutes 20
```

Run `nvidia-smi` first and pick a GPU nobody else is using. A shared GPU does not change the bit
results, but it makes every timing meaningless.

## What each step does

| step | what it answers | needs |
|---|---|---|
| `gemm.bitmatch` | over random shapes, is the Triton GEMM byte-identical to cuBLAS | `--minutes`, `--reps` |
| `gemm.perf` | what the bit constraint costs on a bare GEMM | about 4 min |
| `gemm.fusion` | placeholder; prints the measurement design, no code yet | — |
| `gemm.cublas-bug` | prints how to run the cuBLAS defect reproducer | — |

`gemm.bitmatch` is the headline. It draws random shapes — deliberately including very deep `K`,
because that is the only place cuBLAS's own split-K defect appears — and for each shape compares
our output to cuBLAS's over `--reps` independent input draws. Half the draws use ordinary
gaussian inputs; the other half spread the exponents across the dtype's usable range, which is
what makes a change in the *order* of the additions visible. With narrow exponents almost any
regrouping rounds to the same bits and the check passes things it should not.

Mismatches are logged, not adjudicated. A mismatch is not automatically ours: on some shapes
cuBLAS itself sums only the first `K - (K % block_k)` elements. `gemm.cublas-bug` points at the
standalone reproducer for that; it is worth running before concluding anything from a mismatch.

`gemm.perf` reports three numbers. The one the paper uses is
`unconstrained_ms / ours_ms` — the best configuration of an unconstrained search against ours,
which is the price of the constraint. It also reports how many configurations in the space were
already byte-identical, and `ours/cuBLAS` for reference only: that ratio also contains
Triton-versus-cuBLAS, which is a different and much larger term.

## Files

```
artifact.py        the only entry point
README.md          this file
AGENTS.md          instructions for an AI agent driving the artifact; CLAUDE.md points here
data/              committed. Results only, as CSV.
  FORMAT.md        every column of every table, generated from the same declaration as the CSVs
  gemm_bitmatch.csv
  gemm_perf.csv
  env.csv          the machine and library versions each run was taken on
cache/             not committed. Streaming jsonl, dumps, sweeps. Regenerate, do not archive.
```

Bulk data — PTX dumps, full configuration sweeps, per-draw tensors — is deliberately not
committed. The script regenerates it; only results are kept.

## Machine and versions

The numbers in the paper were taken on:

```
GPU        NVIDIA GB300, compute capability 10.3 (sm_103)
cuBLASLt   13.1.1
Triton     3.8.0+fb, this repository at commit 8f24688b9
PyTorch    2.12.0+cu130
Python     3.12
```

`artifact.py` prints the architecture and cuBLASLt version it actually ran against and warns
when either differs, and `data/env.csv` records both for every run. **This matters more than it
usually does.** Which kernel cuBLAS picks is a function of the architecture *and* the library
version, and the bits it returns follow from that choice, so a row is only comparable to another
row with the same two. The same is true of the defect reproducer: the three `K` values it
carries fire on different (architecture, version) pairs, and none of them fires everywhere.

## Building

The repository builds against a prebuilt LLVM. The one setting that is easy to get wrong, and
which fails in a way that looks like a compiler bug rather than a build mistake:

**Build `libtriton.so` with the same compiler that built LLVM.** Mixing them — clang against a
gcc-built LLVM — corrupts the MLIR-to-LLVM translation, and then *every* kernel, down to a
trivial elementwise add, dies with `dyn_cast on a non-existent value` deep inside
`ModuleTranslation`. It is not a Triton bug and there is nothing to bisect.

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

## How the measurements are taken

Both facts below cost real time to learn and are easy to get wrong when re-implementing.

**Time cuBLAS through a hot closure.** `cublas_matmul` rebuilds the handle, all four layouts, the
preference object and re-runs the heuristic on every call; timing it measures host setup and can
be seven times slower than cuBLAS itself on small shapes. The closure in `artifact.py` sets all of
that up once. It must keep the heuristic `results` array alive on the object: as a local it is
freed when the constructor returns, the algo pointer dangles, and `cublasLtMatmul` then returns
non-zero and does nothing — which times as roughly a petaflop and looks like a spectacular result.

**Report device time, not end-to-end.** Timing the Python call brackets it with CUDA events on the
stream, so every microsecond the GPU spends idle waiting on the host is counted — and the two
arms are wildly asymmetric there, one being a single ctypes call and the other a Triton launcher.
On a 30 microsecond GEMM that measures the launcher. `artifact.py` captures each call in a CUDA
graph and times replays with the L2 cache flushed in between.
