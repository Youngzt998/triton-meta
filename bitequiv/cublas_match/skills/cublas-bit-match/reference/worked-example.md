# One family, start to finish

**This is an illustration, not a recipe.** Every number here is a property of one machine — a
GB300 (sm_103) with cuBLASLt 13.1.1 and 13.2.2 — and one library. On your machine each one is a
hypothesis to test. Read it for the *shape* of the work: what got read, what got measured, what
got declined, and in what order.

You can skip this file entirely. Nothing in the main skill depends on it.

It happened in five commits in this repository, in this order:

| commit | what it did |
|---|---|
| `4340a4fe8` | added the family: heuristic reading, source reading, probe, first verification |
| `48069bd6f` | measured the second corner, and pinned down what the decline would take |
| `9d3cd3cc7` | fixed the output-dtype hole that had produced a false finding |
| `e354c255e` | corrected the decline note — most of it was not unknowable after all |
| `239a7cdd9` | recounted the evidence in output row-draws, which changed what it was worth |

---

## The gap

Some fp8 shapes were declining. The algorithm id was 74, which no table carried, so every shape
cuBLAS routed there fell back to the library.

It was not a rare corner. Over 6,000 random shapes drawn from the slice cuBLAS actually accepts
for fp8 — `K % 16 == 0`, N even, M free — algorithm 74 took **76%** of them.

**Why it had gone unnoticed** is the lesson worth taking. Every fp8 sweep the work was built on
rounded all three dimensions up to a multiple of 16 first, because "cuBLAS refuses fp8
otherwise". That is not what cuBLAS does. Algorithm 74 lives exactly in what that rounding
removes. The family was not new; it was invisible by construction.

## Reading the heuristic

Nine integers, as always. What mattered:

- `ALGO_ID` 74, which advertises exactly one `STAGES_ID` and is listed under e4m3 only.
- `SPLITK_NUM` is 1 or **-2**, and nothing else. -2 is not a split count. It is the library's mark
  for the stream-K tile scheduler, and the kernel it then launches carries a `_stream_k` suffix
  the `SPLITK_NUM` 1 kernel does not.

That second point needed care. A helper that normalised anything below 1 up to 1 would have
turned stream-K into a silent claim of a single unsplit accumulator. The field is read raw,
before any normalising, and -2 declines.

## Reading the kernel name

```
cutlass3x_sm100_tensorop_s64x64x32gemm_..._64x64x128_..._align4_1sm_...
```

Three readings, and the third is the one that mattered:

1. `cutlass3x` — CUTLASS 3.x, which is **not** the CUTLASS of the older algorithm ids in the same
   table. Those close their accumulator once per MMA. This one does not. It gets a family of its
   own rather than becoming a fifth member of an existing one.
2. `s64x64x32gemm` and the `64x64x128` tile — 128 is the threadblock's k step and **32 is the
   MMA's k**. The tile spells the same 32.
3. **`align4`** — the kernel assumes only 4-byte alignment.

## Reading the source, from the alignment token

4 bytes is below TMA's 16-byte minimum, so CUTLASS cannot build this kernel on TMA. It builds it
on the `cp.async` collective. And that mainloop shifts the k axis:

```cpp
// include/cutlass/gemm/collective/sm100_mma_cpasync_warpspecialized.hpp
auto k_residue = K - size<1>(gB_in) * size<2>(gA_in);
// Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
Tensor gA = domain_offset(make_coord(0, k_residue, 0), gA_in);
```

So the MMA groups run `[0, K % 32)`, then 32 apart, ending exactly at K. A partial group at the
**front**, every later one full.

That single reading explained a failure the byte test could see but not diagnose. A twin whose
groups start at 0 is the same computation only when `K % 32 == 0`.

## Reading the order with a probe

The source says where the groups start. It does not say whether each group closes an accumulator.
That took the k probe, and the probe read: **one accumulator, never closed, k blocks walked in
ascending order.**

**The first probe was wrong, and how it was caught is the most transferable part of this.** It
placed its three marker values at adjacent k. One `tcgen05.mma` reduces 32 elements of k
internally before the accumulator sees them, so the probe met its own cancelling value inside the
instruction, and **every output row read the same value no matter what the structure was**.

Three control plans of known structure read the same too. That is the only reason it was caught.
One marker per MMA group fixed it.

## Verifying it

Two parts, because there are two things to check and they need different shapes.

**Part 1, the flat-from-zero form**, which is the same computation wherever `K % 32 == 0`:
1,797 shapes, 16,173 comparisons, across two library versions, **0 differed**.

**Part 2, the shifted form across the boundary**: 846/846 at `K % 32 == 0` and 720/720 at
`K % 32 == 16`. The flat form manages only 587/720 at the second, and only 30 of 80 shapes. Nine
input draws per shape, three of them spanning the whole e4m3 exponent range.

**What the bytes could not settle.** A `K % 128` residue is byte-identical to the `K % 32` one on
all 160 records tried, **including the 82 whose grouping really differs**, because they agree on
everything past the leading group. Only the source separates them. This is the case that makes
"source gives the hypothesis, bytes confirm it" a two-way street rather than a slogan.

## The second corner

The sweep above was "N even, M free". It could not reach the other way cuBLAS gets to the same
family: **`N == 1` with M even and not a multiple of 8**, where it is the output column's
alignment rather than an operand's that lands the shape on `align4`.

A 92,699-shape random run declined 433 distinct shapes there. The recipe carried over unchanged,
so this was a measurement rather than a table change: 936 shapes and 5,650 byte comparisons, 0
differed, over those 433 plus an 816-shape grid crossing `M % 8` in {2, 4, 6} at seven magnitudes
against `K % 32` in {0, 16} at fourteen.

The M rule read clean over 1,215 heuristic queries: even and not a multiple of 8 gets algorithm
74, a multiple of 8 gets the other tensor-core family, odd gets no algorithm at all.

## The controls, and what they caught

The byte test does resolve the one free knob here — where the groups start:

| control | result |
|---|---|
| leading group moved by 16 | caught 45 of 45, at every M from 2 to 49,004 |
| flat-from-zero form | caught on all 30 shapes where it is wrong, survived all 15 where it is not |
| MMA k of 64 | survived 45 of 45 — **not a blind spot**: with the leading group at `K % 64` the boundaries land on the same multiples of 32, so it is the same arithmetic |
| MMA k of 16 | **45 errors, not 45 catches** — a 16-wide fp8 dot does not compile, and a control that crashes has caught nothing |

That last row is trap 2 from the main skill, caught in the act.

## Counting the evidence properly

The commit that changed what all of this was worth did not change any code.

A difference in the last fp32 bits only moves an fp16 output on a row whose exact value sits that
close to a rounding boundary. Measured on this family, about **one row in 3,000**.

So the 5,650 byte comparisons above are not 5,650 samples. They carry **27,112,686 output
row-draws**, 26,745,624 of them from shapes with M ≥ 1000. An error of last-fp32-bit size anywhere
in that recipe would have shown about **9,000 times**, and it showed none. The profile records the
row count next to the shape count for exactly that reason.

## The decline, and how it was earned

`SPLITK_NUM -2`, the stream-K tile scheduler, is about a quarter of the family and it declines.

**Why no plan can express it.** Stream-K walks one flat concatenation of every stream-K tile's k
range and cuts it into equal units, so the cut lands at a different k in each output tile, some
tiles are not split at all, and per-unit snapping shifts the boundaries again. Every plan mode in
this design applies one chunk to every output element. A structural decline, not an unmeasured
one.

**It is measured to be needed, not argued from the source.** Against the unsplit reconstruction,
16 of 1,218 stream-K shapes match. Against **every** uniform partition — each of the `Kt - 1`
chunk lengths at the 128-element k-tile grain, plus the unsplit form — 16 of 18 shapes with a long
enough output have no match at all, and the two that do have one and three survivors that their
own neighbours contradict.

**The sweep can find an answer when one exists.** On 12 `SPLITK_NUM` 1 shapes of the same size it
reports the unsplit form every time. A zero is a result, not a broken search.

**The sweep only means anything on a long output.** At `M == 2, N == 1` all 374 chunk lengths
tried reproduced cuBLAS's two fp16 numbers — re-associating an fp32 chain of similar-sized
partials does not move a value that is then rounded to fp16. On a short vector this byte test
cannot see the partition at all, and a wrong recipe validated there would look perfect.

**How close a twin got, and the reading that was wrong.** Five of the six host inputs the schedule
needs are settled by the source for an `N == 1` shape: the cluster is 1 (the name says `1sm`),
`get_log_swizzle_size` keys off `min(tiles_m, tiles_n)` and returns 0, the default-Heuristic
`get_rasterization_order` gives AlongN whenever `tiles_n <= tiles_m` and that pins the group count
at 1, separate reduction is `return false` dead code, and the CTAs per wave are the plain SM count
because the stream-K `get_grid_shape` passes `truncate_by_problem_size = false`. Only the SM count
is a device fact, and the device reports it.

A prototype built on that looked byte-identical on 12 shapes except for 4 elements out of about
250,000, with 9 of 12 shapes clean. **That reading was wrong.** Re-running the clean shapes with
40 wide-exponent draws instead of 2 turned three of them into 8, 9 and 12 differing elements; only
one of six stayed at 0, over 131,280 row-draws. So the twin carries a small persistent error
across nearly the whole family, and the per-shape zeros were sample size rather than correctness.

Two negative results are kept because they cost time and would cost it again:

- The residual is **not** the chunk boundaries. Many boundary perturbations "fix" any given
  element, and most of those violate the scheduler's own 8-k-tile minimum. That is what a
  one-unit-in-the-last-place coincidence looks like, not a correction.
- It is **not** the twin's N tile. BN 16, 32, 64 and 128 all give the same count — which
  incidentally extends the "the tile is bit-neutral" measurement to BN for fp8, where only the
  tile height had been checked.

## A false finding, and the API hole behind it

Partway through, an all-ones fp8 GEMM asked for in fp32 came back around 1e24. That reads as a
dramatic numerical result. It was a wrapper bug: the output-dtype mapping sent everything
non-bf16 to fp16 while the buffer was allocated from the requested dtype, so cuBLAS wrote fp16
into an fp32 buffer and the caller read two halves back as one float. No error, no warning. It
cost a reader an hour and a wrong conclusion.

The fix is one line of policy: an unlisted output dtype **raises**, the same way an unmeasured
operand dtype already did. Nothing else was ever supported — it was only silently accepted.

## What was and was not claimed at the end

The row went into **one** architecture's profile. The same gap is very likely live on a sister
architecture — the study that concluded the heuristic never returns 74 there filtered fp8 shapes
on `M % 16 or N % 16 or K % 16` in both of its sweeps, so it excluded the family by construction —
but that is unverified and was not testable from that box.

**An unverified table entry is worse than a decline.** So it was left out.
