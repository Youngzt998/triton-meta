# cublas-match

A Triton GEMM whose output is **bit-identical to cuBLAS**, driven by cuBLAS's own heuristic.

Ask `cublasLtMatmulAlgoGetHeuristic` what it would run for a shape, read the recipe off the
algo config it returns, and launch the Triton kernel that reproduces that arithmetic. Nothing is
executed to decide a shape and no bytes are compared — the cost over cuBLAS's own answer is one
heuristic query, about 68 µs, and only the first time a shape is seen.

A config the measured tables do not carry **declines** and the caller falls back to cuBLAS, so
an unfamiliar kernel costs coverage rather than returning wrong bits.

## Use

```python
import torch
from bitequiv.cublas_match import cublas_equivalent_gemm, cublas_matmul

M, N, K = 4096, 4096, 4096

# fp16 (bf16 is the same call).  a is [M,K] row-major, b is [K,N] row-major.
a16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
b16 = torch.randn(K, N, device="cuda", dtype=torch.float16)
c16 = cublas_equivalent_gemm(a16, b16, cublaslt="13.1")

# fp8 e4m3.  a is [M,K] row-major, b is [K,N] COLUMN-major -- w.t() of an [N,K] weight.
# That layout is cuBLAS's own requirement for e4m3; the other one raises.
a8 = (torch.randn(M, K, device="cuda") / 4).to(torch.float8_e4m3fn)
w8 = (torch.randn(N, K, device="cuda") / 4).to(torch.float8_e4m3fn)
c8 = cublas_equivalent_gemm(a8, w8.t(), scale_a=1.3, scale_b=0.017, cublaslt="13.1")

assert torch.equal(c16.view(torch.uint8),
                   cublas_matmul(a16, b16, cublaslt="13.1").view(torch.uint8))
```

Two things are selectable because **both are inputs to which kernel cuBLAS picks**, and so to
the bits it returns:

| | per call | per process |
|---|---|---|
| which `libcublasLt` | `cublaslt="13.1"` / a full path | `set_cublaslt(...)` |
| how much workspace cuBLAS may use | `workspace_bytes=` | `set_workspace_bytes(...)` |

Omit the library and the newest installed one is used. Libraries are cached, so alternating
between two costs nothing after the first use of each.

## Modules

| file | what is in it | depends on an architecture? |
|---|---|---|
| `ltapi.py` | cuBLASLt through ctypes: library/version/workspace selection, descriptors, the heuristic query, `cublas_matmul` (the reference) | no |
| `plan.py` | `CublasGemmPlan` and `static_plan`: heuristic config → which kernel and its parameters | no — it reads a profile |
| `kernels.py` | the Triton twins and their launchers | no |
| `gemm.py` | the public API, the per-shape plan cache, `verify()` | no |
| `arch/` | **the measured data**: one file per (architecture, cuBLASLt major.minor) | this is where it lives |
| `example.py` | a runnable tour of the API — `python -m bitequiv.cublas_match.example` | |
| `evaluate.py` | the fuzz harness behind the coverage numbers — `python -m bitequiv.cublas_match.evaluate --timeout 300` | |
| `cublas_gemm_bug_reproduce.py` | a standalone reproducer for the cuBLASLt split-K tail loss, depending on nothing here | |

`static_plan` is a pure function — the heuristic config is an *input*, not something it queries
— so the whole planner can be exercised for every architecture from any machine, with no GPU.
That is what `tests/test_cublas_match.py` does.

## What the plan says

```python
CublasGemmPlan(mode="splitk_groups", algo_id=24, k_chunk=1024,
               block_k=64, k_per_dot=16, merge_scheme=2, raw_config=(...))
```

`mode` names one of nine kernels; the rest are its parameters. Three facts are **not** fields
because they are the same for every plan, and duplicating them here could only ever drift out of
agreement with `kernels.py`:

* the accumulator is **fp32**, never tf32;
* split-K partials are merged in **fp32**;
* the reduction order is **linear, forward**, and the result is rounded to the output dtype once,
  on store.

## Adding a GPU, or a cuBLASLt version

One new file in `arch/` plus one line in `arch._REGISTRY`. No dispatch code changes.

Each `ArchProfile` field carries a comment saying **how it was measured**; nothing here may be
guessed from another architecture. Profiles do not import each other even where their tables
agree — sm_103 and sm_90 happen to share most of sm_100's, and 13.1 and 12.8 agree completely,
but those are findings, not rules, and a future divergence should be a change to one file.

An architecture with no profile is refused (`CublasUnsupportedPlatform`). An unmeasured
*version* of a known architecture warns once and runs, because a version bump usually moves
nothing and one that renumbered `ALGO_ID` or `STAGES_ID` would fall out of the tables and
decline rather than answer wrong.

## The kernel families

Three run on the tensor core and are reconstructed with `tl.dot`:

* **`ALGO_ID` 66** — cuBLAS's own `nvjet` kernels, the default on Blackwell. One fp32
  accumulator, `block_k`-grained split-K.
* **`ALGO_ID` 12, 21, 23, 24** — CUTLASS, the fallback when a contiguous dimension is not
  8-element aligned. They differ only in the alignment they require (16 / 4 / 2 bytes). The
  accumulator is updated once per MMA rather than once per `block_k` step.
* **`ALGO_ID` 74** — `cutlass3x_sm100_tensorop_*`, CUTLASS 3.x, and a family of its own rather
  than a fifth member of the one above: its accumulator is never closed. fp8 only. Its one
  subtlety is where the MMA groups start: `align4` is below TMA's 16-byte minimum, so CUTLASS
  builds this kernel on the cp.async collective, whose mainloop shifts the k axis to put the
  residue at the origin — groups run `[0, K % 32)` then 32 apart. `SPLITK_NUM` is 1 or -2; -2 is
  not a split count but cuBLAS's mark for the stream-K tile scheduler, and it declines (see
  below).

  cuBLAS reaches this family down two routes, and both were a slice earlier fp8 sweeps here
  rounded away, which is why it went unnoticed rather than being new. Whichever operand ends up
  4-byte aligned rather than 16-byte aligned is what puts a shape on `align4`:

  | | what cuBLAS returns |
  |---|---|
  | N even, not a multiple of 16, M free | ALGO 74 on most of them, nvjet on the rest |
  | N == 1, M even and not a multiple of 8 | ALGO 74, always |
  | N == 1, M a multiple of 8 | nvjet |
  | N == 1, M odd | no algorithm at all |

  The second row is a corner the first sweep could not reach, so it was measured separately and
  the recipe carried over unchanged. The M rule was read over 1,215 heuristic queries with no
  exception.

The other four run on the CUDA cores (SIMT), one fp32 accumulator per output element and no
tensor core at all. `tl.dot` cannot reproduce them — it loses by 1 ulp even at K = 2, and no
regrouping fixes that — so they are rebuilt from explicit `a[:, None] * b[None, :]` products:

* **`ALGO_ID` 11 `gemmSN_NN_kernel` and 16 `magma_sgemmEx_kernel`** — one kernel with three
  levels of accumulation. `CUSTOM_OPTION` gives (S, B): S threads share an output column, so K is
  cut into S contiguous chunks, and B is the k per inner accumulator. 16 is the (S, B) =
  (1, whole K) corner of the same kernel, i.e. one plain ascending chain. 11 lives at M 2..16
  with K up to about 1200, 16 at M 6..16 with K 2..45.
* **`ALGO_ID` 13 `gemv2T`/`gemv2N_kernel`** — a gemv (M == 1 or N == 1). `CUSTOM_OPTION` selects
  (V, W, CC, lane tree): V k-elements per lane load, W lanes per output element, CC tiles per
  chunk, and the W lane totals combined either as a count-down butterfly or left to right.
  `CUSTOM_OPTION` 5 is the same algo with a different lane layout — a contiguous k slice per
  lane, chunked by 16 inside it — so it has its own kernel.
* **`ALGO_ID` 14 `dot_kernel` + `reduce_1Block_kernel`** — a gemv with an fp32 workspace. Tiles
  are handed to blocks strided by `SPLITK_NUM`, each thread's vector lanes are separate
  accumulator chains, and the second kernel merges the block partials with a 4-warp butterfly.

## How the recipe is read from the config

| attr | field | what it decides |
|---|---|---|
| 0 | `ALGO_ID` | which kernel family, and so which planner |
| 6 | `STAGES_ID` | the threadblock `block_k` and the k per dot (tensor-core families) |
| 5 | `CUSTOM_OPTION` | the k split and the lane tree (CUDA-core families) |
| 3 | `REDUCTION_SCHEME` | which split-K merge scheme |
| 2 | `SPLITK_NUM` | the partition, at the `block_k` grain |

`STAGES_ID` is the public enum `cublasLtMatmulStages_t`, whose names spell the tile out
(`32x1`, `64x5`, `64xAUTO`), so `block_k = 16 << ((id - 1) // 6)` for ids 1..24. Attrs 7 and 8
are collected for completeness and read by nothing: `INNER_SHAPE_ID` is `UNDEFINED` on every one
of 2,448,266 sm_100 shapes measured, and the cluster shape does not change the k-loop grouping.

## Coverage, and what declines

Known unsupported (raise `CublasUnsupportedShape`), all inside `ALGO_ID` 13:

* a `CUSTOM_OPTION` outside `gemv_recipe` and `gemv_cslice_recipe`. Seven were seen in the
  sweeps (8, 45, 46, 48, 50, 67, 95), together about 2% of the ALGO 13 shapes drawn. They are
  not one corner: 8, 48, 50 and 95 turn up at K under 80, 45 in the hundreds, 46 and 67 above
  K 50,000.
* `CUSTOM_OPTION` 5 outside `SPLITK_NUM` 1 with M == 1, the only place it was ever seen.
* `CUSTOM_OPTION` 10 on a gemv longer than `gemv_max_elems` output elements, where cuBLAS keeps
  the config but changes the order — it picks the lane width from occupancy, which the config
  does not carry, so two shapes with identical nine-field configs run different orders. This one
  is a decline by choice rather than by ignorance: the population above the cap is one island
  (130 of 2,464 `(13, 10)` hits in a dense scan, all `M == 1`, output length 10,500..14,550, K in
  the 150..210 band), and the lane counts there are 26, 24 and 20. None is a power of two, and
  `_triton_gemv13` shapes its lane axis with `tl.arange(0, W)` and combines with a count-down
  butterfly, both of which need one. Closing it would take a new lane-tree kernel plus a table
  for the lane count as a function of output length and K — a launch cost model, not a config
  field. Two new pieces of machinery for an island this size is the wrong trade.
* either gemv family with M > 1 **and** N > 1 (never observed; it would not be a gemv).

Inside `ALGO_ID` 74 (sm_103 only, the only profile it is measured on), one more:

* **`SPLITK_NUM` -2**, the stream-K tile scheduler — about a quarter of the ALGO 74 population.
  Stream-K walks one flat concatenation of every stream-K tile's k range and cuts it into equal
  units, so the cut lands at a different k in each output tile, some tiles are not split at all,
  and per-unit snapping shifts the boundaries again. Every plan mode here applies one chunk to
  every output element, so this is a structural decline rather than an unmeasured one.

  It is measured to be needed, not assumed. Against the unsplit reconstruction, 16 of 1,218
  stream-K shapes match. Against **every** uniform partition — each of the `Kt - 1` chunk lengths
  at the 128-element k-tile grain, plus the unsplit form — 16 of 18 shapes with a long enough
  output have no match at all, and the two that do have one and three survivors their own
  neighbours contradict. The same sweep finds the answer when one exists: on 12 `SPLITK_NUM` 1
  shapes of the same size it reports the unsplit form every time, so a zero is a result and not a
  broken search. **The sweep only means anything on a long output.** At `M == 2, N == 1` all 374
  chunk lengths tried reproduced cuBLAS's two fp16 numbers — re-associating an fp32 chain of
  similar-sized partials does not move a value that is then rounded to fp16 — so on a short
  vector this byte test cannot see the partition at all, and a wrong one would look right.

  What it would take, read off `sm90_tile_scheduler_stream_k.hpp` and `tile_scheduler_params.h`
  (the SM100 scheduler defers all the split and reduction maths to the SM90 one): the combine
  order is the easy half — deterministic, ascending k, left-associated, fp32, no atomically
  reordered sums. The partition is the hard half. It is a **per-output-tile list of chunks**,
  which no plan field can carry, and computing it needs six host inputs cuBLAS chooses and does
  not expose in the config (`sm_count`, `max_active_clusters`, `splits`, `max_swizzle_size`,
  `raster_order`, `decomposition_mode`). A twin would also have to reproduce the split between
  the last peer's chunk, which stays in the MMA accumulator, and the others, which come back
  through a global fp32 workspace. That is a new plan mode and a new kernel, so it is a task of
  its own rather than a table row.

fp8 reaches none of the four CUDA-core families and declines there. fp8 also needs `BM >= 64`,
or Triton stops using the native fp8 tensor-core path and rounds differently from cuBLAS.

**Known residual**, a follow-up rather than a property of the approach: over 100,692 shapes on
sm_100 the derivation covers 99.91% and 99.891% of those are byte-identical to cuBLAS. The 110
that are not are all nvjet split-K at very deep K, and no partition at all reproduces cuBLAS
there — a brute-force sweep of every chunk found nothing, and neither the config nor the
launched kernel name separates them from the 25,808 nvjet split-K shapes that do match. Until
that is settled these shapes return a mismatching result instead of declining.

The evidence now points at cuBLAS itself: on the shapes it covers,
`cublas_gemm_bug_reproduce.py` shows `cublasLtMatmul` returning the product over only the first
`K - (K % block_k)` elements. That is an observation, not a verdict.

bf16 used to be a second residual of the same kind and is not any more. The split-K merge
rounded to fp16 for two of the merge schemes whatever the output dtype, so every bf16 shape on
that path was wrong; it stayed hidden because only fp16 was ever fuzzed, and for fp16 the two
agree. The merge now rounds to the output dtype, and a 30-minute bf16 cell over 8,428 shapes —
including 5,666 CUTLASS split-K shapes at exactly the affected `REDUCTION_SCHEME` — is
33,712/33,712 byte-identical.

For the full measurement report, see `CUBLAS_GEMM_BEHAVIOR.md`.

## Tests

```bash
pytest bitequiv/tests/test_cublas_match.py
```

* **plan replay** — every recipe-table row and every decline branch of all four profiles, from
  `tests/fixtures/cublas_plan_*.json`. No GPU, no cuBLAS, no GEMM. Regenerate with
  `tests/gen_cublas_plan_fixtures.py`; the fixtures should only change when the planner is
  deliberately changed, and that should be a diff of its own.
* **end to end** — runs the reconstruction against cuBLAS on the local box and compares bytes.
  Only reaches the plan modes this GPU and cuBLASLt happen to route to, which is a subset; it
  is there to catch a launcher wired to the wrong plan field, not to prove coverage.

`verify()` prints a quick per-shape summary on the current device.

## Dependencies

`torch`, `triton`, `ctypes` (stdlib), and the CUDA `libcublasLt` shared library (always present
with a CUDA runtime). `ltapi.py` and `plan.py` need neither triton nor a GPU.
