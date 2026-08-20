---
name: cublas-bit-match
description: >
  Work out the exact order in which cuBLAS adds up a GEMM, on a GPU or a
  cuBLASLt version nobody has measured yet, and write that order down so a
  Triton kernel can return byte-identical output. Use when porting a
  bit-exact cuBLAS twin to a new architecture or library version, when some
  shapes come back with wrong bits, when deciding whether a tuning knob can
  change the result, or when you need to tell "our kernel is wrong" from
  "cuBLAS dropped a term". Covers the heuristic query, cuBLASLt logging,
  reading the launched kernel name, reading CUTLASS source, the
  floating-point probe, controls, and the verification a bit claim needs.
---

# Matching cuBLAS bit for bit

> To let Claude Code pick this up automatically, copy or symlink this directory into
> `.claude/skills/cublas-bit-match/`. It is plain markdown, so any other agent — or a person —
> can read it in place instead.

## The problem

Floating-point addition is not associative. `(a+b)+c` and `a+(b+c)` can differ in the last bit.
So a GEMM that computes the right answer is not the same thing as a GEMM that computes
**cuBLAS's** answer. To get the same bits you have to add the same numbers in the same order.

cuBLAS will not tell you the order. It will tell you which kernel it picked, as nine opaque
integers. This skill is how to turn those integers into a reduction order you can rebuild, and
how to know when you have really got it rather than only appeared to.

**Scope.** This is about bit-matching a GEMM to cuBLAS. Nothing else.

## What you need

A GPU, a `libcublasLt` shared library, a way to write a kernel with explicit control over the
k loop (Triton is what this was done in), a profiler that reports launched kernel names, and an
agent to drive the scans. **No particular repository.** Everything below is against the public
cuBLASLt C API, public NVIDIA documentation, and the open-source CUTLASS tree.

## The shape of the work

```
offline, once per (architecture, library version)
    nine config integers  ─────────────────────────────>  "how this will be summed"
                          reverse engineering, producing a static table

runtime, once per shape
    shape ──[heuristic query, no GEMM runs]──> nine integers ──[lookup]──> kernel + parameters
```

The runtime path executes nothing to decide and compares no bytes. One heuristic query cost about
68 microseconds on the machine this was measured on, and only happens the first time a shape is
seen. A config the table does not carry **declines**, and the caller falls back to cuBLAS.

### What has to be captured

You choose the representation — a Python dict, a JSON file, a generated header, whatever fits
your codebase. What the representation has to be able to say, for each kernel family:

1. **Which products land in the same accumulator.** How the k axis is cut up and dealt out.
2. **How often an accumulator is closed** — closed meaning written out and a fresh one started.
   Those are the rounding boundaries.
3. **How partial sums meet each other.** The shape and direction of the merge tree.
4. **Where the first group starts.** Some kernels put the leftover `K % block_k` elements at the
   front, not the back. This is a separate fact from (1) and it is easy to miss.

The accumulator dtype is usually not a per-kernel fact — it follows from the compute type you
pass at query time and is the same for every entry in one table.

### The five shapes of answer you will meet

Knowing which one you are looking at tells you how much is left to measure.

| structure | what has to be pinned | how hard |
|---|---|---|
| **plain** — one accumulator per output element, walk K once, never closed | **nothing** beyond "it is this one". A flat accumulator leaves no grouping choice; the output tile is irrelevant and so is the threadblock k step, because neither forms a rounding boundary | free |
| **split-K** — K cut into contiguous slices, one partial each, merged by a second kernel | the split count (stated in the config), the grain (from the stages enum), the cut positions (a closed form, see the CUTLASS source in Step 3), and the merge scheme | mostly free |
| **CUTLASS residue-first** — as above but the accumulator is closed **once per MMA**, and the `K % block_k` leftover goes at the **front** | the MMA's k, the threadblock k step, and therefore the leading group size | one reading |
| **SIMT chain** — no tensor core, an explicit multiply-add chain with two or three levels | how many threads share an output column, and how many k per inner accumulator. Encoded in an opaque config field | **must be measured** |
| **gemv** — one output element per row or column, several lanes cooperating on each | lane width, how k is dealt out to lanes, how often a lane closes its accumulator, and the merge-tree direction. All in one opaque config field | **must be measured, and this is most of the work** |

The split count does **not** need a table entry of its own — it is a runtime argument to the
closed form, so no split count, however large, adds an entry.

**The CUDA-core structures cannot be built with a tensor-core instruction.** A tensor-core dot
differs from an explicit multiply-add chain by one unit in the last place even at `K = 2`, and no
regrouping recovers it. Those twins have to be built from explicit outer products.

---

## Step 0 — pin the equivalence class before measuring anything

"Bit-identical to cuBLAS" is not a complete sentence. It is only true inside a class, and if you
do not name the class you will measure one thing and ship another.

NVIDIA's own Results Reproducibility section says the same bit-wise results are given at every
run when the run is on GPUs with **the same architecture and the same number of SMs**, with the
same library version, a single stream, and a provided workspace. Measurement here found exactly
the same list, independently. So the class is:

| axis | why it is in the class |
|---|---|
| architecture | different kernels are compiled per architecture |
| **SM count** | not the same as architecture. H100 SXM has 132 SMs, H100 PCIe has 114, both are sm_90 |
| cuBLASLt version | the `.so` is what holds the kernels; its own kernel names carry the architecture |
| workspace allowance | an **input** to the algorithm choice, see below |
| one stream | cuBLAS's own condition |

**The workspace allowance is an input, not an implementation detail.** cuBLAS reads
`CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` when it picks an algorithm. The same shape at 0, at
1 MiB and at 32 MiB can come back with a different split count and so a different addition
order. Make it a named parameter of your API next to the library version. If you hardcode it,
your code stays self-consistent with its own reference and quietly disagrees with every caller
who allows something else — that failure is silent, which is worse than loud.

It hides well. In one 8-shape sample only 1 shape differed, so a small sweep would have said
"no effect". The real failing shape needed 8 seeds to show up 1 time.

**The heuristic is a host cost model, not a benchmark.** Worth knowing, because it means a plan
derived on one machine can be replayed on another in the same class. This was proved rather than
assumed: breakpoints on `cuLaunchKernel`, `cuLaunchKernelEx`, `cuEventRecord`,
`cuEventElapsedTime`, `cuMemAlloc_v2` and `cuMemcpyAsync` over 1000 queries gave zero hits, and
the same program with one `cublasLtMatmul` added hit them. `cuDeviceGetAttribute` is called the
same number of times whether the program runs 0 queries or 200 — every device read happens once
at `cublasLtCreate`. Without the positive control the zero would have been worthless.

**Two caller-side knobs will break the class.** `cublasSetSmCountTarget()` /
`CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET`, which is documented as optimizing heuristics for a
different SM count and does change the split count; and Green Contexts through
`cublasLtMatmulAlgoGetHeuristicForStream`.

---

## Step 1 — read what cuBLAS chose

### The query

`cublasLtMatmulAlgoGetHeuristic` takes the full description of a matmul — shapes, input and
output dtypes, layouts, compute type, epilogue, workspace budget — and returns a ranked list of
candidates. The first is what it would run. **The query executes nothing**, so scanning millions
of shapes is cheap.

Each candidate carries an algo config: nine integers, read one at a time with
`cublasLtMatmulAlgoConfigGetAttribute`. Five of them bear on the order of addition:

| attr | name | what it selects |
|---|---|---|
| 0 | `ALGO_ID` | which family of kernel |
| 2 | `SPLITK_NUM` | how many pieces K is cut into |
| 3 | `REDUCTION_SCHEME` | how those pieces are merged |
| 5 | `CUSTOM_OPTION` | which variant inside the family |
| 6 | `STAGES_ID` | the threadblock's k step |

The other four are the output tile, the CTA swizzle and similar scheduling parameters, and they
do not change how the k axis is grouped.

**Read each attribute at the width the header declares for it.** `cublasLtMatmulAlgoConfigAttributes_t`
declares `INNER_SHAPE_ID` (7) and `CLUSTER_SHAPE_ID` (8) as `uint16`; every other one is 32 bits.
cuBLASLt checks the buffer size exactly, so reading a 16-bit attribute with 4 bytes returns
`INVALID_VALUE` and your reader silently records `None`. That hid the cluster shape for a while.

Two notes on those two fields, so you do not waste time on them:

- `INNER_SHAPE_ID` is `cublasLtMatmulInnerShape_t`, so it looks like it should hand you the MMA
  shape directly. It does not. It read `UNDEFINED` on every one of 2,448,266 shapes measured on
  one architecture, across every dtype and every algorithm the heuristic returned. cuBLAS never
  fills it in. Do not build on it.
- `CLUSTER_SHAPE_ID` *is* populated, but the cluster shape does not change the k-loop grouping.
  It is still worth reading, because it decodes a field of the kernel name (see Step 2) and can
  tell you a family apart without launching anything.

**A numeric field can hold something that is not a number.** The split-count field carries `-2`
on one family, and that is not a split count at all — it is the library's mark for the stream-K
tile scheduler, which cuts the k axis differently in every output tile. Read fields raw and look
at the whole range of values before you write any normalising code. See Step 7.

### `STAGES_ID` is not opaque

It is the public enum `cublasLtMatmulStages_t`, and the enum names spell the tile out — `32x1`,
`64x5`, `64xAUTO`. So `block_k = 16 << ((id - 1) // 6)` for ids 1..24. That rule reproduced every
key that had been measured one at a time, with no disagreement, which is what makes it safe to
trust for keys nothing has hit yet.

### Enumerate the space, do not sample it

`cublasLtMatmulAlgoGetIds` lists every algorithm the device has.
`cublasLtMatmulAlgoCapGetAttribute` per id gives the tile ids, the stages ids and the maximum
custom option. Together they bound how many table entries can ever exist, so you know when your
map is complete instead of hoping.

Then run a heuristic-only sweep over 10^6 to 10^7 shapes to see which of those keys actually come
back and how often. That decides the order to measure them in. Keys that are reachable but never
returned need no measurement — they just fail closed.

For any key the heuristic will not hand you, **force it**: `cublasLtMatmulAlgoInit` on a blank
config or a copy of a heuristic result, `cublasLtMatmulAlgoConfigSetAttribute` to write each
field, `cublasLtMatmulAlgoCheck` to confirm it is legal, then `cublasLtMatmul`.

Forcing is also a probe in its own right. Forcing `SPLITK_NUM` down to 1 removes the split and
the merge from the picture entirely, which is how one long-running "cuBLAS uses a cross-CTA
reduction tree" theory was killed: with the split forced away, the difference was still there, so
it was never the split.

### Turn on cuBLASLt's own logging

Three environment variables, all public:

```bash
CUBLASLT_LOG_LEVEL=5      # 0 off, 1 error, 2 trace, 3 hints, 4 heuristics, 5 API trace
CUBLASLT_LOG_MASK=31      # bitmask if you want to pick levels instead of stacking them
CUBLASLT_LOG_FILE=/tmp/lt_%i.log   # %i is replaced with the process id
```

At level 5 every API call prints its parameters, and the algo struct prints with its config
fields, which is a fast way to sanity-check that your ctypes reader agrees with the library.
Levels 3 and 4 print what cuBLAS thinks about the shape.

**Read those messages as hints to check, not as verdicts.** Some of them say a shape is not
supported and the call then goes on and runs it. Treat any such line as a pointer at something
worth measuring, never as an answer.

---

## Step 2 — read the launched kernel name

Run one real GEMM per representative shape and read back the launched kernel name, plus grid,
block and shared memory, from the profiler's CUDA activity.

cuBLAS's kernel names are structured. The nvjet family reads:

```
nvjet_<arch>_<dtypesig>_<T1>x<T2>_<BK>x<STG>_<Cx>x<Cy>_[2cta]_<h|v>_<beta>[_<epi>]_[splitK]_<lay3>
```

and CUTLASS names read like

```
cutlass3x_sm100_tensorop_s64x64x32gemm_..._64x64x128_..._align4_1sm_...
cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4
```

**Verify every field rather than assuming it.** The lever is the grid, which the profiler reports
next to the name: get the two tile dimensions the wrong way round and the grid formula stops
fitting. One real check — 818 launched kernels, the correct reading of the tile fits 818/818, the
transposed reading only 336/818. A second cross-check is shared memory:
`(T1 + T2) * BK * element_bytes * STG` has to land under the hardware limit, and a wrong decoding
produces impossible numbers.

Names carry three things worth having:

- **the k step per pipeline stage**, which on nvjet is `128 bytes / element size` — 64 for
  fp16/bf16, 128 for fp8. On the 651 split-K shapes where some other grain would give a different
  cut, the name's grain was the only one that matched, so it is the true grain and not a fitted
  constant.
- **the MMA's k**, from the `s16816` / `s64x64x32gemm` style token.
- **the alignment**, from `align16` / `align8` / `align4` / `align2` / `align1`.

### The alignment token was the single most valuable read

A name containing `align4` means the kernel only assumes 4-byte alignment. That is **below TMA's
16-byte minimum**, so CUTLASS cannot build that kernel on TMA and builds it on the `cp.async`
collective instead. And the `cp.async` mainloop shifts the whole k axis so the residue sits at
the origin:

```cpp
// cutlass/include/cutlass/gemm/collective/sm100_mma_cpasync_warpspecialized.hpp
auto k_residue = K - size<1>(gB_in) * size<2>(gA_in);
// Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
// This aligns the tensor with BLK_K for all but the 0th k_tile
Tensor gA = domain_offset(make_coord(0, k_residue, 0), gA_in);
```

So the MMA groups run `[0, K % 32)`, then `[K % 32, K % 32 + 32)`, and so on, ending exactly at
K: **a partial group at the front and every later one full**. A twin that starts its groups at 0
is the same computation only when `K % 32 == 0`, and wrong otherwise.

That reading explained a failure that byte comparison alone could not. It is worth making the
alignment token the first thing you look at on any name you do not recognise.

### What the name cannot tell you

Structure, yes. Addition order, no. The name tells you `block_k` is 64 and the MMA is `s16816`;
it does not tell you whether those 64 elements go into one accumulator or four.

It can also mislead outright. `internal::gemvx::kernel` takes its lane count from the runtime
`blockDim`, not from a template parameter, so **three different addition orders can share one
kernel signature**. On one architecture custom options 45, 62 and 63 all had the same signature
and three different recipes. The "read the name" shortcut works on the tensor-core side and fails
on the gemv side.

And a name field is a poor gate even when you read it right. A gate of "reject the `_v_` cluster
family" was 18% wrong before a bug was fixed and 22% wrong after — every rejection wrong.

---

## Step 3 — read the CUTLASS source

cuBLAS's CUTLASS kernels are open source. Read them.

**State the division of labour out loud, and keep to it:**

> **Source gives the hypothesis. Bytes confirm it.**
> The compiled library is not the source, so a reading is never the last word.

Four things were settled by source and could not have been settled any other way:

- **Where the residue sits** — the `domain_offset` line above.
- **The split-K partition grain.** `params_universal_base.h`, `init_grid_tiled_shape`, computes
  `kAlignK = max(max(128/bits(A), 128/bits(B)), cacheline_aligned ? cacheline_elements : 1)` and
  then `gemm_k_size = round_up(ceil_div(K, batch_count), kAlignK)`. For fp16 that is 64 when K
  divides by 64 and 8 otherwise. That is the closed form for the cut positions, sitting in the
  source, and it agrees exactly with the form that had been fitted by brute-force search.
- **The raster order, the swizzle size and the separate-reduction flag** for the stream-K
  scheduler. In `sm90_tile_scheduler_stream_k.hpp` and `tile_scheduler_params.h`, for an `N == 1`
  shape: `get_log_swizzle_size` keys off `min(tiles_m, tiles_n)` and so returns 0 whatever the
  maximum swizzle is; the default-Heuristic `get_rasterization_order` gives AlongN whenever
  `tiles_n <= tiles_m`, which pins the group count at 1; separate reduction is `return false`
  dead code; and the CTAs per wave are the plain SM count, because the stream-K `get_grid_shape`
  passes `truncate_by_problem_size = false`. Six host inputs looked unknowable and five of them
  are in the source.
- **The one case bytes could not decide at all.** A residue of `K % 32` and a residue of
  `K % 128` are byte-identical on every record tried, including the 82 whose grouping really
  differs, because they agree on everything past the leading group. Only the source separates
  them.

That last one is the reason to read source even when your byte test is green.

---

## Step 4 — probe for the reduction order when the source runs out

Some kernels are not open source, and some open-source kernels take their layout from a runtime
value. For those, stop guessing whole recipes and **measure which k land in the same
accumulator**.

### The construction

Make both operands almost entirely zero, with values only at chosen k positions:

| position | value |
|---|---|
| `P` | `+1024` (call it `+L`) |
| `Q` | `-1024` (`-L`) |
| probe position `r` | `2^-15` |

The products are exact, so the accumulator's contents are fully controlled.

### The readout

`1024 = 2^10`. fp32 carries 24 significand bits, so one unit in the last place in `[2^10, 2^11)`
is `2^-13` and half of that is `2^-14`. The probe, `2^-15`, is **smaller than half a unit in the
last place**:

- a probe added to an accumulator **currently holding an uncancelled L** is rounded away — it
  **vanishes**;
- a probe added to an accumulator **currently at zero** is kept exactly — it **survives**.

So the output element comes back non-zero if the probe survived and zero if it did not. "Did not"
has a precise meaning: at the moment the probe was added, its accumulator held an L that had not
yet been cancelled. Walk `r` and the survival pattern draws the accumulator boundaries directly.

This is cheap because every output element is an **independent** reduction, so one launch can
carry a different probe position per output element and read thousands of probes at once.

Full worked arithmetic — including how four scans give you lane width, lane stride, chunk length
and merge-tree direction — is in `reference/probes.md`.

### The correction that made it work on a tensor core

The first version of this probe placed its three marker values at **adjacent** k. On a CUDA-core
kernel that is fine. On a tensor core it is not: **one `tcgen05.mma` reduces 32 elements of k
internally before the accumulator ever sees a value.** The probe therefore met its own cancelling
value inside the instruction, and every output row read the same number no matter what the
structure was.

**Place one marker per MMA group, not at adjacent k.** Read the MMA's k off the kernel name
(`s64x64x32gemm` → 32) and space the markers by that.

### Elimination, not comparison

Comparison asks *is my guess right*. The probe asks *what is the structure*.

Comparison needs you to guess correctly up front, which is hopeless once the candidate space runs
to thousands — lane count times stride pattern times chunking times tree direction. Feed the
probe's structural constraints into a candidate grid instead (roughly 10^3 orders, simulated
exactly in fp32) and usually one survives.

The form of the conclusion changes with the method, and that matters:

- comparison yields **"this candidate passed"**;
- elimination yields **"only this recipe could have produced these readings"**.

A reading that matches no known layout announces itself. A pass rate does not.

---

## Step 5 — controls, in the same launch, always

**This is the step most easily skipped and least safely skipped.**

Run plans whose structure you already know alongside the unknown one, in the same launch, on the
same data. If a known plan does not read back its own structure, **the instrument is broken and
the run reports nothing** — throw the run away, do not interpret it.

That is exactly what caught the bad tensor-core probe above. The probe returned the same value
whatever the truth was; what made it obvious was that three control plans of known structure
returned the same value too. Without them the confident wrong answer would have shipped.

Every conclusion has to arrive with two numbers:

1. **A catch rate.** Build a plan perturbed the way a mis-ported table entry would be wrong — not
   perturbed randomly — and report the fraction of samples on which it is caught.
2. **A false-positive control.** Run an **unperturbed** plan under a different label and confirm
   its catch rate is zero. Any catch there is a false catch and voids everything above it.

A conclusion with a low catch rate **does not hold**. Go back to the probe.

One class of conclusion admits no control at all: where the twin exposes no knob whose change
would move the bits, "passed" is the only possible outcome. Write those up as *trivially
portable*, not as *verified*.

**Count errors separately from catches.** A control that crashes has caught nothing. See trap 2
below.

---

## Step 6 — verification that earns its claim

### Sweep the recipe space, not the shape space

cuBLAS's heuristic only ever hands a given shape a small corner of a plan's parameter space. No
number of random shapes will reach the rest of it — not a merge scheme the heuristic never picks
for that size, not a left-to-right tree when every shape it gives you is a butterfly.

So make the reconstruction take its recipe as an argument and **walk the parameters directly**,
calling the kernel with plans you build by hand. Cross every field value your table can name.

This is not a theoretical improvement. One such sweep ran 14,240 combinations with 0 differences;
run against a knowingly broken kernel on **one shape** it reported 112 differences across 46
recipe groups, including combinations shape sampling can never reach. The shape fuzz that found
the same defect needed 3,581 shapes to find 89 cases.

Do both. A recipe sweep with the wrong shapes still misses things — the same 14,240-combination
sweep missed two later defects because its shapes put only seven output elements in the sensitive
regime and it compared the narrow output instead of the wide intermediate.

### Use wide-exponent inputs

`randn`-style data is not enough. Narrow exponents make almost every regrouping round to the same
bits. Draw log-uniform magnitudes with random signs, spread over the dtype's usable range, and
run that set alongside the ordinary one.

Concretely: dropping an fp8 tile-height floor was byte-identical on every `randn/4` shape and
genuinely wrong on wide-exponent inputs, because the small tile moves fp8 off the native MMA and
rounds the accumulator twice as often.

Wide exponents are the right tool for a wrong **grouping of the bulk of K**. They are the wrong
tool for a reassociation that is real everywhere but only survives the output rounding
occasionally — that one needs many *shapes*, not extreme values. Know which you are hunting.

### Compare the widest intermediate the two sides share

If both sides produce fp32 partials before the final cast, compare the partials. One defect
showed on 40 draws out of 40 against the fp32 partials and about 5 draws out of 40 against the
fp16 output. The final cast throws away roughly a factor of ten of your detection power; do not
pay it if you do not have to.

### Count output row-draws, not shapes and not comparisons

A difference in the last fp32 bits only moves an fp16 output on a row whose exact value happens
to sit that close to a rounding boundary. Measured on one family, that is about **one row in
3,000**. So "5,650 comparisons, 0 differed" is almost meaningless on its own and very strong once
you notice those 5,650 comparisons carried **27,112,686 output row-draws**, 26.7 million of them
from shapes with M ≥ 1000. An error of last-fp32-bit size anywhere in that recipe would have
shown about 9,000 times, and it showed none.

Define it once and use it: an **output row-draw** is one output row compared, summed over every
input draw of every shape. Report it next to the shape count.

The same arithmetic cuts the other way, which is trap 4 below.

### State the space and the count

Write the sentence that a single counterexample would refute, number it, run it over 10^5 shapes,
and **report the violation rate, not the pass rate**. One real set:

| rule | checks | violations | verdict |
|---|---|---|---|
| fp8 chunk closed form | 131,533 | 0 | holds |
| fp8 grain is always 128 | 131,533 | 0 | holds |
| chunk depends only on (K, split count) | 58,324 groups | 0 | holds |
| **aligned fp16 is exact from the heuristic alone** | 56,999 | **453** | **falsified** |
| fp8 plain, tiles under 64 | 33,013 | 0 | holds — an existing gate was unnecessary |

442 of those 453 failed on every seed tried, so they were structural rather than marginal
rounding. This step is not a formality. It is the step that catches what you have already
shipped.

**Invariance is tested, never noticed.** "The chunk depends only on K and the split count, not on
M or N" is a claim; you test it by grouping on `(K, split count)` and counting distinct chunk
values per group.

**One clear case proves an effect. Nothing short of a sweep proves the absence of one.** For any
"X never moves the bits" claim, name the axes that could interact *before* you run, cover them,
and report the count. A big sweep that misses the one interacting axis is the same mistake with
more numbers.

---

## Step 7 — fail closed

A shape whose config the table cannot express must **decline**, and the caller falls back to
cuBLAS. An unfamiliar kernel should cost you **coverage**, never **wrong bits**.

Three kinds of decline, and the distinction belongs in both the code and the notes:

- **structural** — the reconstruction cannot express it at all (a gemv kernel handed a shape with
  both M and N above 1; a per-output-tile chunk list when every mode applies one chunk to every
  output element).
- **unmeasured** — the key is not in the table. More work removes this.
- **undecidable** — the config genuinely does not determine the order. One real case: a gemv
  variant whose lane count comes from occupancy, so two shapes with identical nine-field configs
  run different orders. No amount of work removes this.

Two rules that follow:

- **A narrower correct claim beats a broader unproven one.** "On these 40 shapes on this GPU, no
  difference" is a result. "No difference" is not.
- **Normalise carefully at the edges.** One split-count field carries `-2`, which is not a split
  count at all but the library's mark for the stream-K tile scheduler. A helper that clamped
  anything below 1 up to 1 would have turned that into a silent claim of a single unsplit
  accumulator. Read such fields raw, before any normalising.

---

## Step 8 — when you are allowed to stop

Not when the match rate is high enough. When:

> **every remaining item has a name.**

Each decline points at one of the three kinds above. Each mismatch belongs to a **named** family
whose boundary you can state — which algorithm id, which dtype, what range of K, what share of
the population. A single mismatch you cannot account for means an entry is still wrong, and that
on some other shape it is quietly returning the wrong bits.

---

## The four ways a measurement lied

Each of these first looked like a finding. Run this as a checklist before believing any result.

**1. A probe that returns the same value whatever the truth is.**
The tensor-core probe with adjacent markers. Every row read the same because one MMA reduced 32
k-elements before the accumulator saw anything. *Check:* did a control plan of known structure
read back its own structure in this run? If not, the run says nothing.

**2. A control whose 100% catch rate was a 100% crash rate.**
A perturbed plan was scored as caught 45 times out of 45. It had in fact raised 45 times — the
perturbed width did not compile. *Check:* count errors in their own column, never as catches. A
control that crashes has caught nothing.

**3. A random seed derived from the shape.**
Input data seeded as some function of M, N and K. Three shapes meant to be a controlled
comparison therefore each saw different data, so any difference between them was the data and not
the thing under test. *Check:* the seed comes from the run, not from the case. Fix a seed and
vary one axis at a time.

**4. An oracle with a 1-in-3,000 detection rate, reported as per-shape zeros.**
A prototype "matched everywhere except 4 elements out of 250,000, 9 of 12 shapes clean". Re-run
the clean shapes with 40 wide-exponent draws instead of 2 and three of them became 8, 9 and 12
differing elements; only one of six stayed at 0, over 131,280 row-draws. The zeros were sample
size, not correctness. *Check:* convert every zero into a row-draw count before you call it
clean, and re-run the zeros at ten times the draws.

**The standing lesson behind all four:**

> A **deliberately wrong** recipe once produced byte-identical output on **91.7%** of randomly
> drawn inputs.

A hypothesis surviving a few samples is the normal case, not evidence. Another real one: a family
reported "42/42 correct", and when a control was attached its catch rate was **0/42** — those 42
passes had established nothing at all.

---

## Which knobs move the bits, and which do not

This is what the table you build actually encodes, so it is worth knowing which axes you can
ignore.

**Measured bit-free** — free to tune for speed, on the shapes and dtypes noted:

| knob | evidence |
|---|---|
| output tile height and width (BM, BN) | 352 comparisons, 79 of them with the tile taller than M, 0 mismatches; BN 16/32/64/128 give identical results on fp8 |
| warp count | changes only how independent output-element reductions spread over warps |
| pipeline depth (`num_stages`) | identical results at 1, 2, 3, 5, 8 |
| tile swizzle / group-M | changes which output elements a CTA owns |
| TMA versus pointer loads | same numbers, different transport |
| warp specialization | same |
| CTA count / 2-CTA MMA pairing | 10 draws per shape over several tile configurations, no difference |
| index width (32-bit versus 64-bit) | workspaces byte-compared identical on 16 shapes |
| how many k groups are packed into one dot instruction | a dot of k extent `16*G` lowers to G chained k16 MMAs on one accumulator in increasing k, so it *is* the G narrow dots |

**Bit-defining** — these are the answer, and they are what the table has to carry: the k grouping
(which products share an accumulator), where the k axis is cut for split-K, where the first group
starts, and the merge scheme for the partials.

**Two exceptions that look free and are not:**

- **fp8 below a 64-row tile changes the MMA kind.** Under BM 64 the compiler cannot use
  `tcgen05.mma`, so for fp8 it upcasts the operands to f16 and runs the f16 MMA
  (`mma.sync.aligned.m16n8k16`). The conversion is exact but the instruction has a different k per
  rounding step, so the fp32 accumulator lands a few units in the last place away. For fp16 inputs
  the two paths happen to agree, which is exactly why only fp8 ever failed and why it was blamed
  on cuBLAS for weeks. Floor the tile height at 64.
- **gemv lane count and warp count are not bit-neutral above one element per lane.** The claim
  "every output element is its own reduction, so the launch shape cannot move a bit" is false. At
  one k per lane per tile all launch shapes agree; above that they do not, because an integer
  argument equal to 1 gets specialized into the generated code and a partly masked short tile then
  does not produce the same fp32 partial as any form carrying it as a runtime value. Measured over
  4,320 cases: 4 differ, all above one element per lane.

Also watch the compiler. A loop unroll attribute reassociated an accumulator chain — the PTX went
from 2 `fma.rn.f32` and 5 `add.f32` to 12 fma, 11 add and 3 separate `mul.f32`, which is a split
accumulator with a shallower add tree. Writing the unroll out by hand in source order kept the
chain and kept the speed.

---

## Is cuBLAS the one that is wrong?

Sometimes the twin is right and the library dropped a term. Test for it directly.

**All-ones operands make the answer exactly K**, so a short result means terms went missing. Two
refinements that this needs:

- **Put the ones inside a k-window and zeros elsewhere.** Then the reading is blind to how the
  partition falls and sensitive only to whether a term was summed at all. A plain all-ones input
  conflates the two.
- **Watch the output dtype.** fp16 cannot represent every whole number above 2048, and the step at
  16384 is 16, so a K of the form `q*64 + 8` rounds down on its own and looks exactly like a
  dropped tail. That produced a confident false "177 of 321 shapes drop". **Ask for an fp32
  output**, which is exact for every whole number up to 2^24, and check the change does not move
  the heuristic's pick (600/600 shapes returned a bit-identical nine-field config either way).

**And check your own API is telling cuBLAS the dtype you think it is.** One wrapper mapped every
non-bf16 output type to fp16 while the buffer was allocated from the requested type, so cuBLAS
wrote fp16 into an fp32 buffer and the caller read two halves back as one float. No error, no
warning, and an all-ones fp8 GEMM came back around 1e24 — which reads as a dramatic numerical
finding rather than as a wrong call. Make an unlisted output dtype raise.

For the record of what this found: on the shapes it covers, `cublasLtMatmul` returned the product
over only the first `K - (K % block_k)` elements, leaving the tail out. The condition is
`K % block_k != 0` and `(K // block_k) % SPLITK_NUM == 0` and `SPLITK_NUM > K % block_k`. It is
present on three architectures and three library versions. Two library versions meet the
condition on almost disjoint shape sets — over 12,282 shapes, 193 for one and 309 for the other,
with **zero** overlap — so "use an older cuBLAS" is not a fix, it just moves which shapes break.

That is an observation, not a verdict. Report it as one.

---

## Reference files

- **`reference/probes.md`** — the probe cookbook. The L-probe with its arithmetic written out, the
  four scans that give one gemv row, the MMA-group correction, the all-ones window, and how to
  build controls, with real catch rates so you know what to expect from a byte oracle.
- **`reference/porting-checklist.md`** — the ordered list to run on a new GPU or a new library
  version, with the six things that moved between the three architectures already measured and the
  four traps that were hit for real.
- **`reference/worked-example.md`** — one kernel family start to finish on one machine: heuristic
  reading, kernel-name reading, source reading, a probe that was wrong and how the controls caught
  it, verification, and a documented decline. It is an illustration of the *shape* of the work,
  **not a recipe to copy** — every number in it is a property of that machine and that library, and
  on yours each one is a hypothesis to test. Skippable.
