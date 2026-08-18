# TileLang differential fuzzing — line D

Findings from a differential fuzzing campaign on **TileLang 0.1.12** (pypi
wheel), run on an H100 (sm_90a) with nvcc 12.8.93.

## What a finding here means

Scope is **bit-equivalence only**. One finding = *two compilations of the same
TileLang `PrimFunc`, differing only in the pass set, produced different output
bits on the same input.* No claim that it is a "real bug" is made at record
time; each report carries labels (pass class, how many elements differ, ULP
distance, whether an exact-integer control still shows it) so a reviewer can
judge later.

## How the two compilations differ

TileLang's pass pipeline is plain Python
(`tilelang/cuda/pipeline.py::CUDAPassPipelineBody`, ~70 `mod = pass(mod)`
lines). Two knobs are used:

1. **Pipeline re-registration.** `register_pipeline(PassPipeline("cuda", body))`
   overwrites the pipeline by name, so a copy of the step list minus one pass
   can be installed from outside. Note that TVM's own
   `PassContext(disabled_pass=[...])` and `opt_level` are **silent no-ops**
   here: the pipeline calls each pass directly instead of through
   `Sequential`, and `tilelang/jit/kernel.py` builds its own `PassContext`
   that shadows any outer one.
2. **`pass_configs`** — the ~69 on/off keys registered in the built binary.

Before every run the harness checks that its copy of the pipeline, with an
empty skip set, reproduces the stock CUDA source **byte for byte**.

## Gates against fake findings

* Non-deterministic kernels are dropped from the corpus permanently (a data
  race is undefined behaviour, not a finding).
* Every tensor parameter — inputs *and* outputs — is filled with identical
  deterministic content on both sides and compared bit for bit afterwards, so
  the runtime never hands the comparison an uninitialised `torch.empty`.
* Every reference run is repeated under a 17 MB heap shift; a kernel whose
  output depends on what else is in the heap is reading out of bounds and is
  dropped.

## Layout

    HITS.md          rolling summary table, one row per finding
    HIT-NNNN/
      HIT-NNNN.md    the report (PLAN.md section 7 format)
      record.json    the raw result record
      ir/            reference.cu, candidate.cu, and the IR immediately
                     before and after the culprit pass

`## SMT reachability analysis` in each report is left as `TODO` on purpose —
it needs reasoning and is written by a reviewing agent, never by the sweep.

## Extra gate added after launch: undefined behaviour

A determinism pre-screen **cannot** see a data race. A kernel that stores to an
address computed from loaded index data (a scatter) has a write-write race when
two lanes produce the same index — but one compiled binary has a fixed write
order, so running it three times always agrees. Only a *different* compilation
reorders the writes, and the race then looks exactly like a miscompile.

So the corpus is also scanned **statically** (`tlfz/irscan.py`): any kernel with
a store whose address depends on a load, or with an atomic read-modify-write, is
dropped as `ub-race` before any comparison. On this corpus that removed 53 of
717 kernels (14 scatter stores, 31 atomics, 8 float atomics).

A second, generic net catches what a static scan cannot follow (aliasing): if
**three distinct culprit pass sets** each produce a difference on the same
kernel, that kernel is retired as `suspect-unstable` and every report that used
it gets a `## RETRACTED` section appended.
