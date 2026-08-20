# The probe cookbook

Every probe here is built the same way: put values in the operands so that the **output tells you
where the accumulator boundaries are**, instead of telling you whether a guess was right.

Nothing in this file depends on a particular codebase. You need a way to write a kernel with an
explicit k loop, and a way to call `cublasLtMatmul` on the algorithm you want to study.

---

## 1. The L-probe

### Construction

Both operands are almost entirely zero. Values sit only at chosen k positions:

| position | value | name |
|---|---|---|
| `P` | `+1024` | `+L` |
| `Q` | `-1024` | `-L` |
| `r` | `2^-15` | the probe |

fp16 times fp16 lands in fp32 with no loss, so the products are exact and you control the
accumulator's contents completely.

### Why it reads out

```
1024 = 2^10
fp32 keeps 24 significand bits
so one unit in the last place inside [2^10, 2^11) is 2^(10-23) = 2^-13
half of that is 2^-14
the probe is 2^-15, which is smaller
```

Three facts follow, and they are the whole instrument:

```
probe into an accumulator holding 1024:   fl(1024 + 2^-15) = 1024     swallowed
probe into an accumulator at zero:        fl(0    + 2^-15) = 2^-15    kept exactly
the two L values meeting:                 1024 + (-1024)   = 0        exact
```

So the output element is **non-zero if the probe survived** and **zero if it did not**, and "did
not" means: at the moment the probe was added, its accumulator held an L that had not yet been
cancelled.

### Why it is cheap

Every output element is an independent reduction, so each output element can carry a **different**
probe position. One launch reads thousands of probes.

If the shape has few output elements, read it the other way round: set the probe value to `2^-24`
and place probes at a **set** of positions per output element. The output is then exactly
`(number of survivors) * 2^-24`, which stays exact while the count is under 2048.

### The tensor-core correction

**One `tcgen05.mma` reduces 32 elements of k internally before the accumulator sees anything.**
A probe placed at k adjacent to its cancelling L therefore meets that L *inside the instruction*,
and the output reads the same value no matter what the real structure is.

Rule: **one marker per MMA group, not at adjacent k.** Read the MMA's k off the launched kernel
name — `s16816` is 16, `s64x64x32gemm` is 32 — and space the markers by that.

The same applies to any kernel with an internal reduction step you have not accounted for. If the
readings look flat, suspect this before suspecting the kernel.

---

## 2. Four scans that give one gemv row

A gemv row is four numbers. `V` is how many consecutive k a lane takes at a time, `W` is how many
lanes share one output element, `CC` is how many tiles form a chunk (how often a lane closes its
accumulator, `0` meaning never), and `DOWN` says whether the lane totals combine as a count-down
butterfly or left to right.

```
W=4, V=2:   k: 0 1 | 2 3 | 4 5 | 6 7 || 8 9 | 10 11 | 12 13 | 14 15
          lane: 0 0 | 1 1 | 2 2 | 3 3 || 0 0 |  1  1 |  2  2 |  3  3
                └────── tile 0 ─────┘  └────── tile 1 ──────────┘
```

There is one primitive throughout: **given `+L` at P and `-L` at Q, did the probe at r survive?**

### Scan 1 — are two positions in the same accumulator?

Take `K = 4`, one operand all ones, the other non-zero only here:

```
k:       0        1        2        3
value: +1024    2^-15    -1024      0
```

One accumulator, left to right:

```
acc = 0
  += 1024    -> 1024
  += 2^-15   -> 1024        swallowed
  += -1024   -> 0
  += 0       -> 0
output = 0
```

Two lanes, each taking every other k:

```
lane 0 (k=0,2):  1024 + (-1024) = 0
lane 1 (k=1,3):  2^-15 + 0      = 2^-15      the probe never met an L
combined:        0 + 2^-15      = 2^-15
output = 2^-15
```

One launch, one bit. **Zero means one accumulator. Non-zero means they were separated.**

### Scan 2 — V and W

Take `K = 16`. Put `+L` at `k = 0` and `-L` at **the last position belonging to the same lane as
0**, so that lane holds an L throughout. Then walk `r` from 1 to K-1 and record survival.

If the real layout is `W=4, V=1` (lane `l` owns `k = l, l+4, l+8, ...`), lane 0 owns
`{0, 4, 8, 12}`, so `-L` goes at `k = 12`:

```
r:         1  2  3  4  5  6  7  8  9 10 11
survives?  y  y  y  n  y  y  y  n  y  y  y
```

The swallowed positions are `{4, 8}` — **even spacing 4, so V=1 and W=4**.

If the layout is `W=4, V=2` (tile = 8, lane 0 owns `{0,1}` and `{8,9}`), `-L` goes at `k = 9`:

```
r:         1  2  3  4  5  6  7  8
survives?  n  y  y  y  y  y  y  n
```

The swallowed positions are `{1, 8}` — **in pairs, spaced 8**. A completely different figure.

So the shape of the pattern gives V and W together: even spacing `g` means `V=1, W=g`; runs of
`V` consecutive positions spaced `V*W` apart give both numbers directly.

Those 11 probe positions can sit on 11 different output elements and be read in one launch.

### Scan 3 — CC, the accumulator boundary

Take the simplest case, `V=1, W=1`, a single lane. The only remaining question is whether that
lane closes its accumulator part way through.

Pin `+L` at `k = 0`, put `-L` far away at the end, and walk the probe forward:

```
CC = 0 (one accumulator throughout):
r:         1  2  3 ... 255  256  257 ... 1000
survives?  n  n  n      n    n    n       n      swallowed forever

CC = 256 (closed every 256 elements of k):
r:         1  2  3 ... 255  256  257 ... 1000
survives?  n  n  n      n    y    y       y      flips at 256
                             ^
                     the accumulator holding the L at k=0 was closed after 255;
                     256 begins a fresh one, starting at zero
```

**The flip is the boundary.** Move `+L` to 256 and scan again: the flip appears at 512. Again:
768. The boundary list comes out `{255, 511, 767, 1023, ...}`, spaced 256, so `CC = 256`.

**The scan must run out to large K.** Below one chunk length the two layouts produce identical
bytes and no comparison can separate them. This is a real difference between two architectures
running the same config value: on one the scan finds no flip at all, on the other it flips every
256. The architecture that recorded "no flip" had only ever reached that config at K under 80,
where the question does not exist.

### Scan 4 — DOWN, the shape of the merge tree

The first three scans fixed which products share an accumulator. What is left is how the lane
totals meet each other.

Take `W = 4` and arrange the four lane totals as:

```
lane:      0        1      2        3
total:   2^-15      0    +1024    -1024
        (probe)          the L pair, in an adjacent pair of lanes
```

Left to right:

```
((2^-15 + 0) + 1024) + (-1024)
   = fl(fl(2^-15 + 1024) - 1024)
   = fl(1024 - 1024)
   = 0                          the probe was swallowed at the second step
```

Pairwise, folding (0,1) and (2,3):

```
(2^-15 + 0) + (1024 + (-1024))
   = 2^-15 + 0
   = 2^-15                      the probe never met an L
```

Zero or `2^-15` separates the two tree shapes directly.

### Putting it together

```
scan 2 -> V, W      how k is dealt out to lanes
scan 3 -> CC        how often a lane closes its accumulator
scan 4 -> DOWN      how lanes combine
```

Then filter a candidate grid — on the order of 10^3 orders, simulated exactly in fp32 — through
those constraints. The conclusion is **"only this layout could have produced these readings"**,
not "this candidate happened to pass". On one architecture 24 of 35 config values came back as
the unique survivor of a 1,355-recipe elimination; the rest tied only with forms that are the
same operation, or with chunk lengths longer than the k range, which a boundary scan at
K = 262,144 then settled.

---

## 3. The all-ones window: did cuBLAS drop a term?

**Question:** not "is our order right" but "did the library sum every product at all".

**Construction.** Both operands all ones, so the exact answer is K. A short answer means terms
went missing.

**Refinement 1 — window it.** Put the ones inside a chosen k window and zeros everywhere else.
The reading is then blind to how the partition falls and sensitive only to whether a term was
summed. A plain all-ones input conflates the two questions.

**Refinement 2 — ask for an fp32 output.** fp16 cannot represent every whole number above 2048,
and its step at 16384 is 16, so a K of the form `q*64 + 8` rounds down on its own and looks
exactly like a dropped tail. That produced a confident false reading of "177 of 321 shapes drop".
fp32 is exact for every whole number to 2^24.

**Check the change is free.** Ask both ways and confirm the heuristic returns a bit-identical
nine-field config — it did, on 600/600 shapes — so you are studying the same kernel.

**Check the API is not lying to you.** If your wrapper maps output dtypes to cuBLAS enums, make
sure an unlisted dtype raises. One wrapper mapped everything non-bf16 to fp16 while the buffer was
allocated from the requested dtype, so cuBLAS wrote fp16 into an fp32 buffer and the caller read
two halves back as one float. An all-ones fp8 GEMM then came back around 1e24, which reads as a
dramatic numerical finding rather than as a wrong call.

---

## 4. Controls

Every probe run carries controls **in the same launch**, on the same data.

### Known-structure controls

Include plans whose structure you already know, and check they read back their own structure. If
they do not, the instrument is broken and the run reports nothing. Throw it away; do not
interpret it. This is what caught the flat tensor-core probe.

### Perturbed controls, with a catch rate

Build a plan perturbed **the way a mis-ported table entry would be wrong** — the residue moved by
one MMA group, a lane count halved, a merge scheme swapped, a tree direction flipped — and report
the fraction of samples on which the perturbation is caught. Do not perturb randomly; a random
perturbation answers a question nobody asked.

Some real catch rates, so you know what to expect from the byte oracle:

| perturbation | caught |
|---|---|
| an extra accumulation level added or removed | ~100% |
| threadblock k step 128 -> 64 | ~100% |
| split-K merge scheme swapped | 100% |
| MMA k 16 -> 8 | 98.1–99.9% |
| residue moved to 0 | 90.7% and 93.9% |
| gemv lane count halved | 57.7–96.0% |
| gemv lane tree flipped | 27.6% and 90.7% |
| **gemv lane vector width** | **11.1% and 13.3%** |

The bottom row is why the gemv tables were never settled by a pass rate on any architecture.

### The no-op control

Run an **unperturbed** plan under a different label. Its catch rate must be zero. Any catch there
is a false catch and voids every number above it. In one full run the only non-zero no-op counts
were exactly the shapes where the correct plan does not match either, which is the right answer.

### Count errors in their own column

A control that raises has caught nothing. One perturbed plan was scored 45 catches out of 45 and
had in fact raised 45 times, because the perturbed width did not compile. Record `caught`,
`missed` and `error` as three separate counters and never fold the third into the first.

### When no control is possible

Where the reconstruction exposes no knob whose change would move the bits, "passed" is the only
possible outcome. Write those up as **trivially portable** (the plan reads no measured value),
not as **verified**.
