# Parked Triton reports — waiting for an owner ruling

**Nothing here is decided.** These 19 reports differ only in NaN payload bits or
only in the sign of a zero. Whether that counts as a real finding is a policy
question the owner has not settled, so every reviewer pass has left them alone.
This file is the material for that call, not the call.

None of them was moved, re-decided, or edited. The ledger entries still read
`parked`.

## What was asked, and what was found

Two questions were put to this audit.

### Q1. Does the harness's "NaN payload only" banner lie on any of them?

**No. The lie exists, but it did not bite this set.**

The banner is real: the harness prints

```
* of those, real value differences: 0; NaN-payload-only: 0
* NaN-PAYLOAD-ONLY difference (no computed value differs anywhere)
```

on **integer** slots too, where the value/payload split has no meaning — there
is no NaN in an `i64` and no payload bits to differ in. `r2-corpus/HIT-0159`
carries that banner on a `torch.int64` slot and is a real wrong answer, off by
as much as 13 on 60 of 64 rows. It was folded into `r2-corpus/HIT-0024`, not
parked.

For all 19 parked reports the differing slot is a **float** slot — 14 `float32`,
4 `float16`, 1 `bfloat16`. On a float slot the banner means what it says. So the
label is trustworthy on all 19 and none of them is a hidden `HIT-0159`.

This was checked properly rather than by reading the first dtype in the file:

* the dtype was taken from the **per-slot** `Difference measurements` block, and
  only from slots with `ndiff > 0`;
* the per-slot `ndiff` values sum to `ndiff_total` on all 19, so no differing
  slot was missed;
* each slot index was cross-checked against the kernel's own pointer argument
  type in `repro.json` (`*fp32` / `*fp16` / `*bf16`) — two independent sources,
  agreeing on all 19;
* as a positive control the same reader was run on `r2-corpus/HIT-0159` and
  correctly returns `torch.int64`, so a method that can see the integer case
  found none here.

As it happens no parked report mentions more than one dtype anywhere in its
text, so on this set the shortcut would have given the same answer. That is
luck, not a reason to trust it next time.

One thing the banner does get wrong in the other direction, worth knowing: the
4 **signed-zero** reports are counted by the harness as **value** differences
(`ndiff_value` 3 to 35) with `max_abs_diff: 0.0`. They carry no NaN-payload
banner at all. They were parked by a human reading `max_abs_diff 0.0`, not by
the banner.

### Q2. Are some of them the `HIT-0099` lane-split mechanism rather than a floating-point-policy question?

**On 9 of the 19 the cue fires, so yes, possibly — and on those the usual
floating-point story is not available.**

`r2-corpus/HIT-0125` showed the `r2-inductor/HIT-0099` mechanism — a `tt.reduce`
result that is not the same in every lane the layout calls a copy — **can
present as signed-zero-only**: its two disagreeing copies were always
`0x80000000` against `0x00000000`, reference uniform. So "only a signed zero" is
not by itself evidence of a licensed floating-point change.

The cue for that class is the one in `HIT-0099`'s own duplicate screen: **is the
`tt.reduce` operand layout byte-identical between the two arms?** If it is, the
pass did not reorder the reduction, so "a reordered reduction moved the payload"
is not available as an explanation, and criterion B1 cannot be applied to it.

Measured on all 19: **9 identical, 10 changed.**

Two qualifications, both important, both measured:

* **The full `HIT-0099` screen has three legs and no parked report passes all
  three.** Leg (a) is the identical reduce operand layout; leg (b) is a combine
  that is not order-independent; leg (c) is the pass **deleting** the
  `ttg.convert_layout` sitting on the reduce result, which is the step that
  changes which copy a consumer reads. On all 9 with leg (a), the
  `convert_layout` count on the reduce result is **unchanged** between the arms
  (1→1, 2→2, 3→3 or 0→0), so leg (c) does not fire. Conversely 5 reports do
  delete it (`HIT-0103`, `HIT-0116`, `HIT-0140`, `HIT-0176` at 1→0, `HIT-0146`
  at 2→1) but all 5 also changed the layout, so for them reorder and lane-split
  are confounded — which is exactly the shape `HIT-0125` had.
* So leg (a) firing is **necessary, not sufficient**. It does not say "this is
  the lane-split class". It says "the reduce-reorder explanation is off the
  table for this report", which is what makes the park worth a second look.

The report that most deserves the owner's eye is **`r2-oracle/HIT-0009`**: it is
signed-zero-only *and* its reduce operand layout is byte-identical — the same
pair of properties that made `HIT-0125` worth writing up.

## The table

`layout` compares the `tt.reduce` operand type between the two arms **with
layout aliases expanded** (see the method warning below). `cvt` is the number of
`ttg.convert_layout` ops reading a `tt.reduce` result, reference → candidate.
`combine` says whether the reduce body picks an operand (`cmpf`+`select`, so not
order-independent when a NaN is present) or is a plain add.

### Group 1 — reduce operand layout IDENTICAL (9). B1 reorder is not available.

| id | slot dtype | label OK for dtype? | kind | ndiff | ref/cand bits | layout | cvt | combine | culprit | suspicion |
|---|---|---|---|---|---|---|---|---|---|---|
| `r2-oracle/HIT-0009` | float16 | yes | signed zero | 76 | `0x0`/`0x8000` | **IDENTICAL** | 0→0 | plain `addf` | `--tritongpu-coalesce` | **Highest interest.** Signed-zero-only *and* identical reduce layout — the `HIT-0125` pair of properties. A reordered sum can flip the sign of a zero, but no reorder is available here. Leg (c) cannot fire: there is no `convert_layout` on the reduce result at all, so if a lane split is behind this, some other step exposes it. Needs the one-arm uniformity probe. |
| `r2-inductor/HIT-0026` | float32 | yes | NaN payload | 7171 | `0xffea5349`/`0x7fffffff` | **IDENTICAL** | 1→1 | picks operand | `--tritongpu-coalesce` | By far the largest parked report. Leg (a) and leg (b) both fire, leg (c) does not. The reference bits `0xffea5349` are a signalling-ish payload against a plain quiet `0x7fffffff`, i.e. the payload is being *replaced*, not permuted. Worth the probe. |
| `r2-inductor/HIT-0149` | float32 | yes | NaN payload | 67 | `0xffffffff`/`0x7fffffff` | **IDENTICAL** | 1→1 | picks operand | `--tritongpu-coalesce` | Legs (a) and (b) fire. Only the NaN **sign bit** differs. Same shape as HIT-0026 on a smaller kernel. |
| `r2-inductor/HIT-0186` | float32 | yes | NaN payload | 1 | `0x7fffffff`/`0xffffffff` | **IDENTICAL** | 3→3 | picks operand | `--tritongpu-remove-layout-conversions` | Legs (a) and (b) fire on all **three** reduces. Only the NaN sign bit differs, on 1 element of 557056. The culprit is the same pass as `HIT-0099`, but it deletes none of the three `convert_layout` ops here. |
| `r2-corpus/HIT-0071` | float32 | yes | NaN payload | 28 | `0x7fffffff`/`0x7fc00000` | **IDENTICAL** | 1→1 | picks operand | `--loop-invariant-code-motion` | Legs (a) and (b) fire. The culprit is a **bit-class** pass with no numeric licence anywhere — LICM should not be able to touch a reduce at all, which makes "licensed floating-point change" a hard argument to make. `0x7fffffff` against the canonical quiet NaN `0x7fc00000`. Screened at the compiled-TTGIR level (this is a TTIR report; see the method note). |
| `r2-corpus/HIT-0142` | float32 | yes | NaN payload | 33 | `0x7fc00000`/`0x7fffffff` | **IDENTICAL** | 1→1 | picks operand | `--triton-licm` | Same as HIT-0071 with the arms the other way round and the other LICM pass. Same argument: a bit-class pass with no licence. |
| `r2-corpus/HIT-0164` | float32 | yes | NaN payload | 25 | `0x7fc00000`/`0x7fffffff` | **IDENTICAL** | 1→1 | picks operand | `--cse` | Same family as HIT-0071 / HIT-0142; culprit `--cse`, again a bit-class pass with no numeric licence. These three look like one mechanism seen three times. |
| `r2-inductor/HIT-0150` | float16 | yes | NaN payload | 6 | `0xffff`/`0x7fff` | **IDENTICAL** | 1→1 | `mulf` | `--tritongpu-coalesce` | Leg (a) fires. The combine is a **product** reduce, so leg (b) needs its own argument: fp16 `mul` also propagates a NaN payload, and which payload survives depends on operand order. Only the sign bit differs. |
| `r2-inductor/HIT-0114` | bfloat16 | yes | NaN payload | 1 | `0x7fff`/`0xffff` | **IDENTICAL** | 2→2 | plain `addf` | `--triton-nvidia-gpu-plan-cta` | Leg (a) fires but leg (b) does **not** — a plain `addf` is order-independent except for payload/sign choice. 1 element of the pool, sign bit only. Weakest of the nine. Note this is one of the two reports where a naive text compare would have said CHANGED. |

### Group 2 — reduce operand layout CHANGED (10). A B1 reorder is available.

| id | slot dtype | label OK for dtype? | kind | ndiff | ref/cand bits | layout | cvt | combine | culprit | suspicion |
|---|---|---|---|---|---|---|---|---|---|---|
| `r2-inductor/HIT-0146` | float32 | yes | signed zero | 3 | `0x80000000`/`0x0` | CHANGED | **2→1** | picks operand | `--tritongpu-remove-layout-conversions` | **The closest match to `HIT-0125` in the set**: signed-zero-only, layout changed, and the pass deletes one of the two `convert_layout` ops on the reduce results. Both explanations are live and confounded, which is precisely what `HIT-0125` had to untangle by probe rather than by label. |
| `r2-inductor/HIT-0103` | float32 | yes | NaN payload | 1 | `0x7fd7477a`/`0x7fd9b1a3` | CHANGED | **1→0** | picks operand | `--tritongpu-remove-layout-conversions` | Leg (c) fires (the only `convert_layout` on the reduce result is deleted) but leg (a) fails. Reorder and lane-split confounded. Both payloads are ordinary quiet NaNs with different bodies, which fits a reorder. |
| `r2-inductor/HIT-0116` | float32 | yes | NaN payload | 64 | `0x7fffffff`/`0x7fc7ed03` | CHANGED | **1→0** | `mulf` | `--tritongpu-remove-layout-conversions` | Same confound as HIT-0103, on 64 elements and a product reduce. |
| `r2-inductor/HIT-0140` | float32 | yes | NaN payload | 1 | `0x7feb90a8`/`0x7ffdcfbe` | CHANGED | **1→0** | picks operand | `--tritongpu-remove-layout-conversions` | Same confound as HIT-0103, 1 element. |
| `r2-inductor/HIT-0176` | float32 | yes | NaN payload | 1 | `0x7fde5127`/`0x7fe45416` | CHANGED | **1→0** | picks operand | `--tritongpu-remove-layout-conversions` | Same confound as HIT-0103, 1 element of 98304. The ledger already notes the layout is CHANGED and the exact-integer control goes equal. |
| `r2-inductor/HIT-0095` | float16 | yes | signed zero | 5 | `0x0`/`0x8000` | CHANGED | 1→1 | picks operand | `--tritongpu-remove-layout-conversions` | Reorder available and leg (c) does not fire. A reassociated fp16 sum can change the sign of a zero, so the ordinary B1 reading is the simple one. |
| `r2-inductor/HIT-0098` | float16 | yes | signed zero | 4 | `0x8000`/`0x0` | CHANGED | 1→1 | picks operand | `--tritongpu-remove-layout-conversions` | Same as HIT-0095, arms the other way round. |
| `r2-corpus/HIT-0104` | float32 | yes | NaN payload | 16 | `0x7fa20da4`/`0x7facd408` | CHANGED | 1→1 | picks operand | `--tritongpu-remove-layout-conversions` | Reorder available, leg (c) does not fire. Two unrelated quiet-NaN payloads — the pattern a reorder produces. |
| `r2-corpus/HIT-0160` | float32 | yes | NaN payload | 90 | `0x7fab26d3`/`0x7fa93131` | CHANGED | 1→1 | picks operand | 4 passes incl. `--tritongpu-coalesce`, `--tritongpu-remove-layout-conversions` | Same as HIT-0104. Multi-pass culprit, so the reorder has more than one possible source. |
| `r2-corpus/HIT-0165` | float32 | yes | NaN payload | 112 | `0x7fa65b48`/`0x7fa446c8` | CHANGED | 1→1 | picks operand | 3 passes incl. `--tritongpu-coalesce`, `--tritongpu-remove-layout-conversions` | Same as HIT-0160. |

## A warning about the screen itself

**The reduce-operand-layout screen must expand the layout aliases before
comparing. Run as a text compare it is wrong on 10 of these 19 reports.**

The operand type is written with an alias, `tensor<64x16xf32, #blocked>`, and a
pass may redefine what `#blocked` means or renumber it. Both directions happen
here:

* **Text says identical, layout actually changed — 8 reports**
  (`HIT-0104`, `HIT-0160`, `HIT-0165`, `HIT-0095`, `HIT-0098`, `HIT-0103`,
  `HIT-0116`, `HIT-0146`). In `r2-inductor/HIT-0103` both files say
  `tensor<64x16xf32, #blocked>`, but `#blocked` is
  `threadsPerWarp = [2, 16], warpsPerCTA = [4, 1]` in the reference and
  `threadsPerWarp = [32, 1], warpsPerCTA = [4, 1]` in the candidate. A text
  compare would have called this the lane-split cue when it is a reduce reorder.
* **Text says changed, layout actually identical — 2 reports**
  (`HIT-0114`, `HIT-0186`), where only the alias number moved.

The four parked reports whose ledger entries already state a layout verdict
(`HIT-0149`, `HIT-0150`, `HIT-0176`, `HIT-0186`) all agree with the
alias-expanded result, so those reviewers did expand. The warning is for anyone
re-running the screen from scratch.

A second note: `r2-corpus/HIT-0071`, `HIT-0142` and `HIT-0164` are **TTIR-level**
reports. TTIR has no `#ttg` layouts at all, so comparing the `.ttir` pair is not
evidence either way. They were screened on the `input.compiled.ttgir` /
`output.compiled.ttgir` pair, which is where layouts exist.

## What would settle each group

Not run here, because deciding these reports is the owner's call:

* **Group 1** — the one-arm uniformity probe from `HIT-0099` §1 and `HIT-0125`:
  inside a single build, store the reduce result broadcast along the row and
  check that every lane the layout calls a copy holds the same bits. If the
  reference arm is uniform and the candidate is not, it is the lane-split
  mechanism and not a floating-point-policy question. `r2-oracle/HIT-0009`
  first.
* **Group 2** — the reorder is available, so the probe cannot separate the two
  causes on its own; `HIT-0125` needed the extra step of showing the split was
  invisible downstream while the reassociated sum was what produced the recorded
  difference.
* **Both groups** — the policy question stays: is a difference in NaN payload
  bits, or in the sign of a zero, with `max_abs_diff 0.0` and `max_ulp 0` and no
  one-sided NaN or Inf, a finding or not? REVIEW-CRITERIA §2 keeps "NaN or Inf
  appears on **one side only**"; on all 19 of these `nan_one_side_only` is 0, so
  that rule does not fire and there is no written rule that does.

## Bookkeeping

* The wrap-up brief said 18 parked reports; the id list it gave has **19**
  (6 `r2-corpus`, 12 `r2-inductor`, 1 `r2-oracle`). All 19 were audited.
* 15 are NaN-payload-only, 4 are signed-zero-only
  (`r2-inductor/HIT-0095`, `HIT-0098`, `HIT-0146`, `r2-oracle/HIT-0009`).
* All 19 have `max_abs_diff 0.0`, `max_ulp 0`, `nan_one_side_only 0`,
  `inf_one_side_only 0`, and `ndiff_ref_unwritten = ndiff_cand_unwritten = 0`.
* Scripts used, kept next to the ledger so the table can be regenerated:
  `/home/youngzt/fuzz/reviewer-triton/audit_parked.py` (dtype and banner) and
  `audit_parked2.py` (alias-expanded layout screen, `convert_layout` count,
  combine kind).
