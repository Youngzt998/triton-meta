# Working on this artifact as an agent

Read `README.md` first for what the artifact is and how to run it. This file is the part a
coding agent needs that a human reader does not: the invariants, and the mistakes that are easy
to make and hard to notice.

## The one rule

**A performance number from a kernel that is not byte-identical to cuBLAS is not a result.**
Check the bits first, then time. If a change makes something faster and the bits move, the
change is rejected — do not weaken the check to accommodate it, and do not report the speed.

This is worth stating because the failure mode is real: several times during this work an
optimisation looked like a large win and turned out to be computing something else.

## Invariants of the kernels

A Triton GEMM reproduces cuBLAS's `plain` arithmetic if and only if it keeps **one fp32
accumulator over the whole K, consumed in ascending k, rounded to the output dtype exactly once
on the store.** These are measured bit-free and may be changed freely: tile size, `num_warps`,
`num_stages`, `GROUP_M` tile swizzle, TMA, warp specialization, `num_ctas`, a persistent grid,
int32 versus int64 indexing, and whether an operand is repacked before the loop.

These are **not** free and must not be touched: which k-elements group into one `tl.dot`, the
split-K partition, the merge scheme, the accumulator dtype, and where the rounding to the output
dtype happens.

Two exceptions that look free and are not. For fp8, `BM < 64` moves the MMA off
`tcgen05.mma.kind::f8f6f4` onto `mma.sync.m16n8k16`, halving the k per accumulator update — it is
byte-identical under ordinary inputs and wrong under a wide exponent spread. And in the gemv
kernels, the elements-per-program and `num_warps` knobs are not bit-neutral once the per-lane
vector length exceeds one.

## Testing

**Two input distributions, many draws.** Narrow exponents hide a regrouping: a deliberately wrong
recipe once produced byte-identical output on 91.7% of randomly drawn inputs. Every bit claim
needs several independent draws per shape, half of them with the exponents spread across the
dtype's usable range.

**Sweep the recipe space, not just shapes.** cuBLAS's heuristic hands any given shape only a small
corner of a plan's parameter space, so no amount of random shape sampling reaches, say,
`merge_scheme=3` or a split count of 16. Two of the three real defects found in this work were
reachable only by walking the plan parameters directly and calling the reconstruction with a
hand-built plan.

**Cover deep K.** cuBLAS's own split-K defect only appears there. A sweep capped at moderate K
will report a clean 100% and will have proved nothing about the shapes where the interesting
disagreement lives.

**A check that cannot fail is not evidence.** Before trusting a new sweep, run it against a
deliberately broken kernel and confirm it reports the difference.

## Traps in the measurement

- The hot cuBLAS closure must keep its heuristic `results` array alive on the object. As a local
  it is freed when the constructor returns and `cublasLtMatmul` silently does nothing, which
  times as roughly a petaflop.
- Report device time from CUDA-graph replay with the L2 cache flushed between replays. End-to-end
  timing counts host idle time, and the two arms are asymmetric there by an order of magnitude on
  small shapes. Report end-to-end as a second number if useful, never as the one being optimised.
- This machine runs at a high load average from other users. Device time is robust to that;
  end-to-end is not.
- Results are only comparable within one (architecture, cuBLASLt version) pair. Record both.

## Building

Build `libtriton.so` with the **same compiler that built LLVM**. Mixing clang against a gcc-built
LLVM corrupts the MLIR-to-LLVM translation and every kernel dies with `dyn_cast on a non-existent
value`; it looks exactly like a compiler regression and there is nothing to bisect. `rm -rf build`
after changing `CMAKE_CXX_COMPILER` or `CMAKE_BUILD_TYPE` — both are cached. Full recipe in
`README.md`.

## Where things go

`data/` is committed and holds results only, as CSV, with `FORMAT.md` generated from the same
declaration as the CSV headers so the two cannot drift. `cache/` is not committed and holds the
streaming jsonl and anything bulky. Do not commit PTX dumps, full configuration sweeps or
per-draw tensors — the script regenerates them.

To add a table, add it to `TABLES` in `artifact.py`; the CSV header and the format document both
come from that one declaration.

## Sharing the machine

Pin one GPU with `CUDA_VISIBLE_DEVICES`. Run `nvidia-smi` first and stay off GPUs other people
are using. Never kill a process that is not yours. Long runs should append their records as they
go so that being killed loses at most one record.
