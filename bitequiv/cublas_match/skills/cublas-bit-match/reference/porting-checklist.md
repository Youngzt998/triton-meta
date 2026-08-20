# Porting to a new GPU, or a new cuBLASLt version

The ordered list. It has been run three times — on an sm_90 part, an sm_100 part and an sm_103
part — and the order below is the one that turned out to cost least.

Nothing here assumes a particular codebase. Where it says "your table", that is whatever you use
to record which kernel adds up in which order.

---

## 0. Decide what you are porting to

Write down the class before you measure: architecture, **SM count**, cuBLASLt version, workspace
allowance, one stream. See Step 0 of the main skill. A table is valid for one class. If you are
moving to a new machine you are building a new one, not reusing the old.

Note two library facts while you are here:

- A box can carry several `libcublasLt` versions and the newest is often not the one your
  framework was built against. Pin the one you mean, by path or by version prefix.
- Two versions of cuBLAS can meet a given condition on **almost disjoint** sets of shapes. Over
  12,282 shapes, one version's bad set was 193 shapes and the other's was 309, with **zero**
  overlap. So "the older library is fine" is not a conclusion you can draw from a sweep on the
  newer one.

## 1. Point the new machine at an existing table and run your full evaluation

Do this before any measurement. Do not edit the table for it — patch it in place at runtime so
you cannot forget.

Whatever byte-matches is free. **The decline and mismatch breakdown, grouped by algorithm id, is
your work list.** On one port this answered the entire question in one run; on another it
produced a work list of exactly twelve gemv config values.

## 2. Enumerate the reachable space, do not sample it

`cublasLtMatmulAlgoGetIds`, then `cublasLtMatmulAlgoCapGetAttribute` per id for the tile ids, the
stages ids and the maximum custom option. That bounds how many entries can ever exist, so you
know when the map is complete.

Then a heuristic-only sweep — no GEMM runs, so it is nearly free — to see which of those keys the
heuristic **actually returns**, and how often. Measure in that order.

For the gemv side, close the table with a log-uniform scan over vector lengths to about 10^6 and
K to about 4×10^6. On one architecture an 8.25 million-query scan turned up 52 reachable config
values, seven of which only appear at very long vectors or very deep K and which no ordinary
sample had ever produced.

### Your sweep's own filter decides what you can find

The single most expensive mistake in the whole effort: every fp8 sweep rounded all three
dimensions up to a multiple of 16 first, because "cuBLAS refuses fp8 otherwise". It does not — it
wants `K % 16 == 0` with N even and M free. An entire kernel family lived exactly in what that
rounding removed, and it took **76%** of the shapes in the slice cuBLAS actually accepts. It was
not a rare corner. It was invisible by construction.

Before trusting a sweep, write down what its shape generator cannot produce, and go and look
there.

The same family was reachable a second way that the first fix still could not see — `N == 1` with
M even and not a multiple of 8, where it is the **output** column's alignment rather than an
operand's that decides. Two different corners of one family. Neither generator could reach the
other.

## 3. Carry what carries, but re-read it

Across three architectures, these carried unchanged, and each was re-read on the machine rather
than assumed:

- the set of algorithm ids the heuristic returns (byte-identical between two of the three, for
  fp16, bf16 and fp8)
- the `STAGES_ID` to `block_k` rule, which is the public `cublasLtMatmulStages_t` enum:
  `block_k = 16 << ((id - 1) // 6)` for ids 1..24
- which stages id the main tensor-core family uses per dtype
- the CUTLASS stages-id key space
- the small-M CUDA-core rows

An earlier attempt to carry rules the other way, from a newer part to an older one, was wrong.
Another architecture's values are a hypothesis to test, never a starting point to copy. Each of
the differences below would otherwise have shipped wrong bits silently.

## 4. The six things that moved, in the order they are worth checking

**a. The single-lane gemv rows.** They moved on two architectures out of two. On one part the
single-lane configs close an accumulator every 512 k and every 256 k where another part records
them as never closing. Check these first.

Note *why* one architecture recorded "never closing" and was not wrong: it only ever reached that
config at K under 80, and below one chunk length the two forms are the same kernel. A row can be
correct on its own machine and wrong on yours for reasons that have nothing to do with error.

**b. The occupancy cap.** One gemv variant keeps its config but changes its lane count from
occupancy above a length threshold. The formula ports even though the constant does not:
`sm_count * threads_per_sm / W`. Confirmed on a 132-SM part giving exactly 8448 where a 152-SM
part gave 9728. Bisect the output length to find it — one length still runs the old lane count
and the next does not.

**c. A second accumulation level appearing.** On one part the fp8 tensor-core kernel closes an
accumulator every 128 k and adds the block totals afterwards, where the same key on two other
parts runs one flat accumulator over the whole slice. That is a different sum, not a refinement,
so it needs a new kernel rather than a new table entry.

How it was found, and this is the general recipe: **walk K in 16-element steps against the flat
form.** The flat form matched at K 16 through 128 — the whole range that fits one 128-element
step — and missed every K from 160 up, **including exact multiples of 128**, which rules out a
residue explanation. Check the instruction did not change underneath you at the same time; here
the PTX showed the native fp8 MMA either way, so it was the library's grouping and not a
compiler fallback.

**d. New gemv rows.** One port added twelve config values another never returns. None was a new
family — each extended an existing even/odd pairing, where an even config is the
contiguous-slice-per-lane form at some lane count and the next odd one is the strided form at the
same count. Look for the pattern before assuming new structure.

**e. Algorithm ids only one part reaches.** One part returns two extra tensor-core ids at about
24 hits in 400,000 draws. No model reproduced them, so they decline, at a cost of one shape in
648,720. That is a fine outcome. Do not invent a recipe for a 24-hit family.

**f. Whole families the previous sweep could not see.** See the filter warning in step 2.

## 5. Four traps that were hit for real

**The output dtype is a weak oracle, and for gemv it is nearly blind.** A deliberately wrong lane
recipe still byte-matched on **91.7%** of shapes. Use the probe, settle by elimination over a
candidate grid, and always run a perturbed control plus a no-op control.

**The all-ones oracle lies above 16384 with an fp16 output.** The fp16 step there is 16, so a K of
the form `q*64 + 8` rounds down on its own and looks exactly like a dropped tail. Ask for an fp32
output and check the heuristic still returns the same config.

**A framework's `matmul` cannot reach these kernels.** Checked across two major framework
versions, eight shape pairs, four operand layouts, three entry points and a large workspace
setting: all correct, none reproducing the kernels under study. **Call cuBLASLt directly, and give
it a workspace** — at a zero-size allowance split-K is never offered and nothing reproduces.

**Structure changes at boundaries you did not cross.** One small-M CUDA-core kernel grows a second
accumulation level only above K = 512, and on one part it reaches K = 31,253 at M 2–3 with N
around 22,000–29,000 — a corner no M × N × K grid with N ≤ 4096 ever visits. Probe both sides of
anything that looks like an edge.

## 6. Verify, and say what the verification covered

- Sweep the **recipe space** directly, not only shapes. The heuristic only hands a shape a small
  corner of a plan's parameters.
- Run a **wide-exponent** input set alongside the ordinary one.
- Compare the **widest intermediate the two sides share**, not the narrow output, when you can.
- Report **output row-draws** next to shape counts.
- Attribute every decline and every mismatch **by algorithm id**.
- Run against the table you will actually ship, not a runtime patch.

## 7. Stop when every remaining item has a name

Each decline points at structural, unmeasured, or undecidable. Each mismatch belongs to a named
family whose boundary you can state. A single unexplained mismatch means an entry is still wrong.

---

## What three ports cost

For scale, so you can plan. Across three architectures: 1.77 million distinct shapes, 6.74
million byte comparisons.

| part | shapes | reconstructed | declined | bit-identical |
|---|---|---|---|---|
| first (sm_100) | 626,522 | 100.00% | 0 | 99.8055% |
| second (sm_103) | 496,906 | 100.00% | 0 | 99.791% |
| third (sm_90) | 648,720 | 99.9995% | 3 | 99.817% |

The second port needed **one** changed table entry. The third needed twelve new gemv entries,
three changed ones, a re-measured occupancy constant and one new kernel. On all three, essentially
every non-matching shape is the same named residual family. The first port is the expensive one;
the rest is a checklist.
