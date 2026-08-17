# ttgir-deep — Line C of the Triton compiler fuzzing campaign

Deep dive on the highest-suspicion TTGIR passes, run against **public** Triton
`3.8.0+git3277063a` (commit `3277063a6`) on an H100 (sm_90).

## What the fuzzer compares

For one kernel it builds the real `CUDABackend.make_ttgir` step list twice:

* **candidate** = the full production pipeline;
* **reference** = the same list with one or more steps removed.

Both results are written to a `.ttgir` file and handed to `triton.compile`,
which skips every stage up to and including `ttgir`.  So `make_llir`,
`make_ptx` and `make_cubin` are the *same* code on both sides, and the only
difference between the two binaries is the TTGIR pass set.  Both are then
launched on the same inputs and the output buffers are compared bit for bit.

## Passes under the microscope

On sm_90 the runnable warp-specialization path is `nvgpu-warp-specialization`
(the Hopper one in `third_party/nvidia/hopper/`); the
`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/` automatic path only runs
for compute capability >= 100 in the default pipeline.  So this line hammers,
in order: Hopper warp specialization, software pipelining
(`assign-latencies` / `schedule-loops` / `pipeline` / `fuse-nested-loops`),
TMA / descriptor lowering, MMAv3 selection, and the layout family.

## What is in this directory

`HIT-NNNN.md` — one report per finding, plus a `HIT-NNNN/` directory with
`ref.ttgir`, `cand.ttgir`, `ir.diff` and `meta.json`.  `HITS.md` is the rolling
summary table.

## Scope of a finding (PLAN section 4, as rewritten 2026-08-17)

A finding here means exactly one thing: **two compilations of the same kernel
that differ only in the TTGIR pass set produce different bits on the same
input.** No claim that it is a compiler bug is made at record time. Nothing is
dropped for looking like floating-point noise; judgment calls are attached as
labels so the whole set can be filtered later, with all results in hand.

Two things ARE dropped, because they would be fake findings:

* a kernel that is not bit-identical to itself (a data race is undefined
  behaviour, so the compiler may do anything -- these are recorded in
  `state["nondet_cases"]` and not investigated);
* a difference that does not survive perturbing the GPU allocator (heap shift),
  which points at an out-of-bounds or uninitialised read rather than a value
  difference.

The most useful label is the **integer-input control**. With inputs that are
integers in `[-8, 8]`, every fp32 accumulator in these kernels stays below
`2^24`, so every accumulator add is exact and summation order cannot matter. A
difference that survives integer inputs therefore cannot be floating-point
reassociation. It separates the classes cleanly: `accelerate-matmul` on an fp32
dot differs on random inputs but is *equal* on integer inputs (reassociation),
while `hopper-warpspec` on `mm_epilogue` still differs (not reassociation).
`backfilled_labels.json` holds this label set for the findings recorded before
the rule change.

The **SMT reachability analysis** section of each report is left as
`TODO: written by a reviewing agent` on purpose: the fuzzer is a script with no
agent in the loop, and that section needs real reasoning about the semantics at
the root of the bug.  The script records the raw material instead (culprit pass
set, pass groups, determinism evidence, which concurrency / async constructs
appear in the IR diff, whether cross-instance behaviour is involved, and the
op-count delta).

Inputs are never checked in: every report gives the RNG seed and the
distribution name, and `harness/kernels.py` in the working copy
(`/home/youngzt/fuzz/ttgir-deep/`) regenerates them exactly.
