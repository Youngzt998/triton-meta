# `inductor_kernels.py` — TorchInductor kernel corpus

Second kernel corpus for the `tv` translation validator, next to the
hand-written `benchmark_kernels.py` (636 kernels from FlagGems / FLA /
TritonBench / torchao).

These kernels are what PyTorch's `torch.compile` writes for you. They are a
very different distribution from hand-written kernels: mostly fused pointwise
and reduction code plus template code (matmul, attention), all machine
written from a fixed set of code patterns. They are also the most-run Triton
code in the world, so a miscompile here hurts the most.

- **184 kernels**, one per structural signature.
- Built from **144 small torch programs**; **759 distinct kernel texts** were
  captured and folded down to 184 (**4.1x** dedup).
- No pretrained model or dataset is used. Everything is built from
  `torch.nn` / raw tensor ops, so it runs offline.

## Versions used

| what | value |
| --- | --- |
| torch | `2.12.1+cu130` |
| triton | `3.7.0+fb.beta` (the beta fbsource checkout) |
| python | 3.12 |
| GPU | NVIDIA H100, sm90, CUDA 13.0 |

The kernel text depends on the torch version (inductor changes its codegen
often) and a little on the GPU (`DeviceProperties`, `cc=90`).

## How to regenerate

### 1. Pick an environment with torch

The `tv` venv (`/home/youngzt/tv/triton/.venv`) has triton but **no torch**, so
the capture used the beta checkout's venv:

```bash
PY=/data/users/youngzt/fbsource/third-party/triton/beta/triton/.venv/bin/python
```

That venv needs three env vars, otherwise triton cannot find `ptxas` and every
compile fails with `RuntimeError: Cannot find ptxas`:

```bash
export TRITON_PTXAS_PATH=/home/youngzt/.triton/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.9.86-archive/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/home/youngzt/.triton/nvidia/cuobjdump/cuda_cuobjdump-linux-x86_64-13.1.80-archive/bin/cuobjdump
export TRITON_NVDISASM_PATH=/home/youngzt/.triton/nvidia/nvdisasm/cuda_nvdisasm-linux-x86_64-13.1.80-archive/bin/nvdisasm
```

Other machine rules that mattered:

```bash
nvidia-smi                      # pick an idle GPU first
export CUDA_VISIBLE_DEVICES=3   # pin one GPU; other sessions share this box
export TORCHINDUCTOR_COMPILE_THREADS=1
```

`TORCHINDUCTOR_COMPILE_THREADS=1` is needed. With the default subprocess
compile pool, the worker fails with
`RuntimeError: 0 active drivers ([]). There should only be one.`, which kills
max-autotune runs.

### 2. Run one workload per process, each with a fresh cache dir

```bash
rm -rf /tmp/indgen/cache/$NAME
TORCHINDUCTOR_CACHE_DIR=/tmp/indgen/cache/$NAME $PY run_one.py $NAME
```

A fresh `TORCHINDUCTOR_CACHE_DIR` per workload does two jobs: it stops the FX
graph cache from skipping codegen, and it tells you which workload each kernel
came from. Six workloads at a time (`xargs -P 6`) finished all 144 in about 10
minutes.

Each workload is a small function or `nn.Module` plus a `torch.compile` call.
Some also call `.backward()`, because backward graphs give different kernels.
Some set inductor config through `torch._inductor.config.patch(...)`, for
example `max_autotune_gemm=True`, `triton.use_block_ptr=True`,
`combo_kernels=True`. The per-kernel comment block in the corpus records the
torch code and the config for that kernel.

### 3. Read the kernels out of the cache dir

Two kinds of generated file hold Triton source, and the corpus uses both:

1. **The wrapper file** — has blocks like

   ```python
   triton_poi_fused_add_mul_0 = async_compile.triton('triton_poi_fused_add_mul_0', '''
   ... kernel source ...
   ''', device_str='cuda')
   ```

   Above each block is the provenance comment
   (`Original ATen: [aten.mul, aten.add]`), which the corpus keeps.

2. **The standalone kernel files** that `PyCodeCache` writes for *every*
   compiled kernel. These also hold the max-autotune template candidates that
   lost the benchmark — that is where the TMA descriptor matmul and the extra
   block-size variants come from. They carry no provenance comment, so the
   corpus falls back to the workload's aten ops and says so.

`TORCH_LOGS="output_code"` and `torch._inductor.config.trace.enabled` also
work, but scanning the cache dir was simpler and caught more.

### 4. Clean up each kernel

- Cut everything before `@triton.jit` and drop the `@triton_heuristics.*(...)`
  decorator (its `triton_meta` / `inductor_meta` argument is huge). The useful
  bits from it — the argument types and `size_hints` — are kept in the comment
  block.
- Keep only the span that holds the `@triton.jit` functions. Template files
  also define subgraph helpers (`forward_inner`, `_triton_helper_fn_add0`,
  ...); those are kept, the benchmark harness after them is dropped.
- The main kernel is the `@triton.jit` function nobody else in the file calls
  (in template files the main kernel comes first and its helpers after).
- Rename every top-level def in an entry to `<name>_<sha1(src)[:8]>`, so names
  are unique across the corpus. This is the only edit to the kernel body.

### 5. Dedup by structural signature

Signature of a kernel (per the task):

```
(multiset of tl.* calls with counts,   # includes libdevice.*, tl_math.*,
                                       # triton_helpers.*, and `.to()` casts
 has for-loop, has while-loop, has data-dependent if,
 number of loads, number of stores,
 is a reduction)
```

Names are ignored. Among kernels with the same signature the shortest source
is kept, and the block says how many captured kernels it stands for.

### 6. Check the file

```bash
source /home/youngzt/tv/triton/.venv/bin/activate
python3 -m py_compile tv/benchmark/inductor_kernels.py
python3 -c "import importlib.util as u; s=u.spec_from_file_location('ik','tv/benchmark/inductor_kernels.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(len(m.INDUCTOR_KERNELS))"
```

Both pass in the `tv` venv, which has **no torch**: the `@triton.jit` helpers
from `torch._inductor.runtime.triton_helpers` that inductor kernels call
(`maximum`, `max2`, `welford_reduce`, `online_softmax_reduce`,
`bucketize_binary_search`, `exclusive_scan_decoupled_lookback`, ...) are
inlined verbatim at the top of the file and re-exported as a
`triton_helpers` namespace.

`INDUCTOR_KERNELS` at the end of the file lists the 184 entries in file order.
Everything else with `@triton.jit` is a helper they call.

## What is in the corpus

### By category

| category | kernels |
| --- | ---: |
| normalization (softmax, log_softmax, layernorm, rmsnorm, batchnorm, groupnorm, cross entropy; fwd and bwd) | 28 |
| reduction (sum/mean/max/argmax/var/norm/any/prod, one axis and many axes, small and large reduction dims) | 21 |
| indexing (embedding, index_select, gather, scatter_add, index_put, masked_fill, where/clamp, tril/triu) | 20 |
| attention (sdpa, hand-written attention, flex_attention fwd + bwd) | 17 |
| end2end (MLP block, transformer block, RoPE, Adam step, MoE router) | 17 |
| pointwise (chains, activations, special functions, bool logic, foreach) | 15 |
| codegen-knob (block pointers, TMA descriptors, combo kernels, no-upcast fp16, emulated precision casts, n-d tiling, native matmul) | 13 |
| matmul (mm, bmm, addmm, baddbmm, int8 mm, fp8 scaled_mm, epilogue and prologue fusion) | 11 |
| shape (transposed, non-power-of-2, broadcast, cat, roll/flip, pad, pixel_shuffle, unfold) | 10 |
| pooling / vision (max_pool2d fwd+bwd, avg_pool2d, interpolate, conv) | 10 |
| dynamic (`torch.compile(dynamic=True)`: symbolic `ks0` sizes) | 8 |
| pointwise/dtype (fp16, bf16, fp64, fp8, int8 quant, mixed precision, autocast) | 7 |
| scan (cumsum, cumprod, cummax, logcumsumexp, split scan) | 5 |
| random (dropout, randn_like: philox) | 2 |

### By inductor kernel kind

| prefix | meaning | kernels |
| --- | --- | ---: |
| `triton_poi_` | pointwise | 77 |
| `triton_per_` | persistent reduction (whole reduction axis in one block) | 51 |
| `triton_red_` | looped reduction (`for r0_offset in tl.range(...)`) | 20 |
| `triton_mm_` / `triton_bmm_` / `triton_tem_` / `triton_convolution_` / `triton_` | matmul / conv templates and their autotune candidates | 27 |
| `triton_flex_attention*` | flex attention fwd + bwd templates | 3 |
| `triton_for_` | combo / foreach kernels (one launch, several sub-kernels behind `if pid < ...`) | 4 |
| `triton_spl_` / `triton_unk_` | split scan, other | 2 |

### Notable features (entries that use them)

| feature | entries | note |
| --- | ---: | --- |
| `libdevice.*` | 54 | exp, rsqrt, erf, tanh, pow, floor, isnan, signbit, ... |
| `tl.dot` | 25 | mm / bmm / conv / attention templates |
| `tl.range` | 22 | the reduction and k loops |
| `tl.debug_barrier` | 15 | in-place buffer reuse ordering |
| `tl.device_assert` | 15 | bounds check on every indirect (gather-style) index |
| `tl.associative_scan` | 5 | cumsum / cumprod / cummax |
| `tl.atomic_add` | 5 | scatter_add, index_put, embedding backward |
| `triton_helpers.welford_*` | 4 | var / std / layernorm |
| `tl.make_block_ptr` | 3 | `triton.use_block_ptr=True` |
| `tl.make_tensor_descriptor` + `tl.load_tensor_descriptor` | 1 | TMA persistent matmul candidate |
| `triton_helpers.x_grid_barrier` | 1 | cooperative reduction: a grid-wide barrier built from atomics |
| `exclusive_scan_decoupled_lookback` | 1 | split scan: atomics + spin wait across blocks |
| `tl.rand` / `tl.randn` | 2 | dropout, `randn_like` (philox) |
| symbolic sizes (`ks0`) | 3 | `dynamic=True` |
| `tl.float64` / `tl.float8e4nv` / `tl.int8` | 1 / 1 / 4 | dtype breadth |

### Dedup

| | |
| --- | ---: |
| distinct kernel texts captured | 759 |
| kept (one per signature) | 184 |
| dedup ratio | 4.1x |
| entries that stand for more than one capture | 47 |
| biggest group | 152 (matmul template, same shape with different block sizes) |

Most of the collapse is matmul: max-autotune writes one kernel per candidate
config, and they share a signature.

## Coverage against what tv models today

Method: `/tmp/coverage.py`, reused as `coverage2.py`. A kernel is "modelable
today" when it uses nothing outside `MODELED`
(`tl.program_id`, `tl.arange`, `tl.load`, `tl.store`, `tl.exp`, `tl.max`,
`tl.sum`, `tl.zeros`, `tl.full`, `tl.cdiv`, `tl.constexpr`, `tl.multiple_of`,
`tl.max_contiguous`, `tl.assume`, `tl.static_assert`, `tl.static_print`, plain
arithmetic and compares). Every other `tl.*` attribute, plus for-loop,
while-loop and data-dependent if, counts as a missing feature.

The inductor corpus keeps its helpers as separate `@triton.jit` functions, and
the hand-written corpus has them inlined. To compare fairly, the script counts
only the 184 corpus entries as kernels and folds the features of the helpers
they call into the caller. On the hand-written corpus this rule changes
15.7% -> 14.5%, so the two ways of counting agree closely.

### Headline

| | hand-written (636) | inductor (184) |
| --- | ---: | ---: |
| modelable today | **14.5%** (92) | **3.3%** (6) |
| same, original script (all `@triton.jit` defs, no helper folding) | 15.7% (100 / 636) | 10.2% (24 / 236) |
| modelable today, `--ext` (also count `libdevice.*`, `tl_math.*`, `triton_helpers.*`, `.to()` casts) | 14.3% (91) | 2.2% (4) |

Inductor kernels are **4x harder** for tv today than hand-written ones. The
reason is not exotic hardware features — it is that inductor writes the same
few idioms into *every* kernel, and tv does not model them yet.

### Top blockers, side by side

| feature | blocks (inductor, of 184) | blocks (hand-written, of 636) |
| --- | ---: | ---: |
| `tl.float32` (dtype literal in casts / `tl.full`) | 153 (83%) | 274 (43%) |
| `tl.where` | 109 (59%) | 206 (32%) |
| `tl.broadcast_to` | 108 (59%) | 4 (0.6%) |
| `tl.int1` (bool masks) | 101 (55%) | 14 (2%) |
| `.to()` dtype cast (`tl.to`) | 83 (45%) | 203 (32%) |
| data-dependent if | 71 (39%) | 145 (23%) |
| `tl.int32` | 63 (34%) | 122 (19%) |
| for loop | 45 (24%) | 203 (32%) |
| `tl.reduce` (higher-order reduce with a combine function) | 31 (17%) | 13 (2%) |
| `tl.dot` | 25 (14%) | 46 (7%) |
| `tl.debug_barrier` | 16 (9%) | 4 (0.6%) |
| `tl.device_assert` | 15 (8%) | 3 (0.5%) |
| `tl.atomic_add` | 7 (4%) | 24 (4%) |
| while loop | 3 (2%) | 16 (3%) |
| `tl.maximum` | 3 (2%) | 112 (18%) |

(The hand-written column is from the same script on `benchmark_kernels.py`
with helper folding, so it differs a little from the earlier 15.7% run:
`tl.float32` 266 -> 274, for-loop 197 -> 203, `tl.where` 185 -> 206,
data-dependent if 121 -> 145, `tl.maximum` 100 -> 112, `tl.dot` 37 -> 46.)

### Greedy unlock order (inductor corpus)

```
+tl.float32        + 13 kernels -> 19/184 (10.3%)
+tl.int1           + 10        -> 29/184 (15.8%)
+tl.to             +  5        -> 34/184 (18.5%)
+<data-dependent if> + 2       -> 36/184 (19.6%)
+tl.where          +  3        -> 39/184 (21.2%)
+tl.broadcast_to   + 13        -> 52/184 (28.3%)
+tl.reduce         + 13        -> 65/184 (35.3%)
+tl.int32          +  9        -> 74/184 (40.2%)
+tl.int64          + 13        -> 87/184 (47.3%)
+tl.debug_barrier  +  9        -> 96/184 (52.2%)
+tl.device_assert  + 10        -> 106/184 (57.6%)
+tl.atomic_add     +  5        -> 111/184 (60.3%)
+tl.sigmoid        +  4        -> 115/184 (62.5%)
+tl.int8           +  4        -> 119/184 (64.7%)
```

For the hand-written corpus the same greedy walk starts with
`<data-dependent if>` (+47), `tl.where` (+22), `tl.atomic_add` (+17),
`tl.float32` (+16), `tl.to` (+19), `<for loop>` (+22), and reaches 54.9%
after 14 features.

Reading: the first six features to model are almost the same set in both
corpora (dtype literals and casts, bool masks, `tl.where`, broadcast,
data-dependent if). Getting those six done moves inductor coverage from 3.3%
to about 28% and hand-written coverage to about 33%. `tl.dot` is *not* on the
critical path for either corpus.

## Things worth knowing (found while collecting)

- **`tl.broadcast_to` + `tl.int1` are everywhere in inductor and almost absent
  in hand-written code.** Every masked reduction is
  `tmp = tl.where(mask, tl.broadcast_to(x, [XBLOCK, R0_BLOCK]), 0)` and every
  "no mask needed" case is `tl.full([R0_BLOCK], True, tl.int1)`. These two
  alone block ~59% and ~55% of the corpus. They are cheap to model and were
  not on the hand-written top-5 list.
- **`tl.reduce(x, dim, combine_fn)` with a `@triton.jit` combine function.**
  Inductor uses this for min/max (`triton_helpers.max2`), product, `any`,
  argmax/argmin (tuple reduce with index), welford (3-value tuple), and online
  softmax. tv will need reductions whose combine step is a user function, and
  tuple-valued reductions. 31 of 184 kernels need it.
- **`tl.device_assert` on every indirect index.** Inductor emits
  `tl.device_assert((0 <= tmp) & (tmp < 512), "index out of bounds")` for each
  gather-style load. Hand-written kernels almost never do this (3 uses vs 36).
  A validator has to either model it or treat it as a no-op with a side
  condition.
- **`tl.debug_barrier()` in normal pointwise kernels.** It appears when
  inductor reuses a buffer in place, so it is an ordering fact, not a
  hardware feature.
- **Max-autotune candidates are a free source of shape variety.** The lost
  candidates stay in the code cache. That is where the only TMA
  (`tl.make_tensor_descriptor` / `tl.load_tensor_descriptor`) and most block
  pointer kernels came from — the winning config was the plain one.
- **Two grid-wide synchronization patterns show up**, both rare but nasty for
  a validator: `triton_helpers.x_grid_barrier` (cooperative reduction, spins
  on an atomic counter) and `exclusive_scan_decoupled_lookback` (split scan,
  publishes partial sums with `tl.atomic_xchg` and spins with
  `tl.atomic_load`). Both have real inter-block communication, which a
  per-program semantics cannot express.
- **Combo / foreach kernels branch on the program id** (`if pid <
  num_xblocks_0: ... elif pid < num_xblocks_1: ...`). That is a
  data-dependent if in the AST sense, but the condition only depends on
  `tl.program_id`, so it is cheap to support and unlocks a whole class.
- **Hand-written attention gets pattern-matched away.** `q @ k.T -> softmax ->
  @ v` in plain torch becomes one `aten._scaled_dot_product_cudnn_attention`
  call and produces *no* Triton at all. You need
  `torch._inductor.config.pattern_matcher=False` (or flex attention) to see
  real attention kernels.
- **`F.scaled_dot_product_attention` alone gives no Triton kernel either** for
  the same reason; `flex_attention` is the way to get an inductor-generated
  attention kernel, and its template is the biggest kernel in the corpus
  (~300 lines with 4 helper functions).
- **Inductor bakes shapes in as constants** (`xnumel = 64`, `r0_numel = 128`,
  strides as literals). That is good news for tv: most kernels have no
  unknown sizes at all. Only the `dynamic=True` kernels carry symbolic `ks0`
  arguments.
