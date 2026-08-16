# Reproducing this artifact with an agent

You are helping a reviewer reproduce the measurements in this artifact. `README.md` describes what
is being evaluated; this file is the operational part — how to get the environment right, what to
run, and what the output means. You should not need to write or modify any kernel.

## Before anything else

```
cd <repo root>
git rev-parse --abbrev-ref HEAD          # should be the artifact branch
nvidia-smi                                # pick a GPU nobody else is using
```

Two things must be set on every invocation. `bitequiv` is not installed, so the repository root
has to be on `PYTHONPATH`; and one GPU has to be pinned, because the timing steps assume they own
the device. A shared GPU does not change the bit results but makes every timing meaningless.

```
export PYTHONPATH=$(git rev-parse --show-toplevel)
export CUDA_VISIBLE_DEVICES=0            # a free one
python artifact_eval/artifact.py --list
```

## Running

```
python artifact_eval/artifact.py --run gemm.bitmatch --minutes 20
python artifact_eval/artifact.py --run gemm --minutes 20     # the whole gemm section
python artifact_eval/artifact.py --export                     # cache/*.jsonl -> data/*.csv
```

`--minutes` bounds the sampling steps; larger is better evidence, not a different result. Records
stream into `cache/` as they are produced, so interrupting a run loses at most one record and
`--export` still works on what was collected.

## Building, if you have to

The prebuilt tree should already work. If you rebuild, there is one setting that fails in a way
that looks like a compiler bug rather than a build mistake:

**Build `libtriton.so` with the same compiler that built LLVM.** Mixing them — clang against a
gcc-built LLVM — corrupts the MLIR-to-LLVM translation, and then *every* kernel, down to a trivial
elementwise add, dies with `dyn_cast on a non-existent value` inside `ModuleTranslation`. It is not
a Triton bug and there is nothing to bisect. `CMAKE_BUILD_TYPE` and `CMAKE_CXX_COMPILER` are cached,
so `rm -rf build` is required or a change is silently ignored, and `setup.py` checks
`REL_WITH_DEB_INFO` before `TRITON_REL_BUILD_WITH_ASSERTS`, so leaving the former set overrides the
latter without saying so. The full recipe is in `README.md`.

`bitequiv/ptx_reduction.py` needs `pyptx==0.1.1`.

## Reading the output

`gemm.bitmatch` prints how many random shapes were byte-identical to cuBLAS and logs the rest.
**A mismatch is not automatically a failure of the artifact.** On some shapes cuBLAS itself sums
only the first `K - (K % block_k)` elements; `gemm.cublas-bug` points at a standalone reproducer
for that defect. Check it before drawing a conclusion from a mismatch.

`gemm.perf` prints three numbers. The claim rests on `unconstrained_ms / ours_ms`, the price of the
bit constraint. `ours/cuBLAS` is printed for reference only — it also contains
Triton-versus-cuBLAS, which is a different and much larger term.

## If something looks wrong

**A warning about the cuBLASLt version.** Expected if your library differs from the paper's. It is
not cosmetic: which kernel cuBLAS picks depends on the architecture *and* the library version, and
the bits follow from that choice, so results are only comparable within one such pair.
`data/env.csv` records both for every run.

**A warning about the architecture.** The measurements are from an NVIDIA GB300 (sm_103). On other
hardware both the bit results and the speeds will differ, and the defect reproducer's three `K`
values fire on different (architecture, version) pairs — none of them fires everywhere.

**A timing that looks impossibly fast.** Suspect the measurement, not the kernel. If cuBLAS appears
to run at hundreds of teraflops on a small shape, the call is failing silently and being timed as a
no-op. Report it rather than recording the number.

**A step fails to compile a kernel.** Almost always the build issue above.

## Please do not

Do not modify `artifact.py`, `bitequiv/`, or anything under `data/`. Do not commit anything from
`cache/` — it is regenerated, not archived. If a measurement seems wrong, report what you saw and
the contents of `data/env.csv` rather than adjusting the script to make it look right.
