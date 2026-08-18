# Write-write race check on this line's corpus — nothing to retract

Line `ttgir-broad` found that kernels which store to an address computed from
loaded data (scatter stores) carry a write-write race whenever two lanes compute
the same address, and that the determinism pre-screen cannot see it. It
retracted 8 reports. This is the same check run against the `ttgir-deep`
corpus.

**Result: clean. No finding on this line is affected, and nothing is retracted.**

## What was checked

`ubscan.py` (in this directory, run from `/home/youngzt/fuzz/ttgir-deep/`) walks
the def-use chain backwards from the address side of every global write and asks
whether it reaches a load. It is run in two passes:

1. the TTIR of **all 818 cases** in the corpus;
2. the full-pipeline TTGIR of one case per family — that is the program that
   actually runs, with `tt.descriptor_store` already lowered to
   `ttng.async_tma_copy_local_to_global`.

`ttgir-broad/fz/irscan.py` is run alongside it on the same text and agrees on
every family. Raw output is in `ubscan.json`.

```
== pass 1: TTIR of all 818 cases ==
  cases whose TTIR has a data-dependent store address or an atomic: 0

== pass 2: full-pipeline TTGIR, one case per family ==
  attn             stores=  4  scatter=no  atomics=no   [line-B scan: scatter=False, atomic=False]
  loop_reduce      stores=  1  scatter=no  atomics=no   [line-B scan: scatter=False, atomic=False]
  mm_epilogue      stores=  2  scatter=no  atomics=no   [line-B scan: scatter=False, atomic=False]
  mm_ptr           stores=  1  scatter=no  atomics=no   [line-B scan: scatter=False, atomic=False]
  mm_tma           stores=  2  scatter=no  atomics=no   [line-B scan: scatter=False, atomic=False]
  mm_tma_persist   stores=  2  scatter=no  atomics=no   [line-B scan: scatter=False, atomic=False]
  nested           stores=  1  scatter=no  atomics=no   [line-B scan: scatter=False, atomic=False]
  tma_copy         stores=  1  scatter=no  atomics=no   [line-B scan: scatter=False, atomic=False]

VERDICT: CLEAN
```

This is expected: the corpus is synthesized rather than harvested, every store
address is an affine function of `tl.program_id`, `tl.arange` and constexpr
block sizes, and no kernel uses an atomic.

## Two ways this scan is stricter than line B's

* Line B walks back from the **first** SSA operand of the store. For
  `tt.descriptor_store %desc[%x, %y], %val` the first operand is the descriptor,
  so a coordinate computed from loaded data would be missed. This scan walks the
  descriptor **and** every coordinate. That matters here because 722 of the 818
  cases store through a TMA descriptor.
* It also covers the post-lowering ops
  (`ttng.async_tma_copy_local_to_global`, `ttng.async_tma_scatter`) and the
  gather/scatter descriptor ops, which only exist after TTGIR lowering.

## The other race shape, and why the pre-screen already covers it

A static scan cannot decide whether two *different program instances* write the
same address. It does not need to here, and the reason is worth writing down
because it is exactly the asymmetry line B ran into:

* **Two lanes of one instance** racing: within a single compiled binary the
  write order is fixed, so three runs of that binary agree and the pre-screen
  sees nothing. Only a different binary reorders the writes. This class needs
  the static scan — hence this document.
* **Two program instances** racing: the order in which the hardware schedules
  blocks varies from run to run, so the same binary gives different answers and
  the determinism pre-screen *does* catch it. All 818 cases passed that
  pre-screen (3 launches on one input, 2 on another, all bit-identical).

Independently, every kernel here writes one tile selected by `tl.program_id`,
and the persistent one walks `tile_id` in strides of `NUM_SMS` from a distinct
`start_pid`, so instances write disjoint tiles by construction.

## Aliasing

The other thing a static scan cannot follow is two kernel arguments pointing at
the same buffer. That cannot happen here: `make_inputs` allocates every tensor
separately for each launch.

## Line B's generic net is still installed

Since aliasing is the case a scan cannot follow, line B's net was adopted
anyway: when three or more **distinct** culprit pass sets fire on the same case,
every finding from that case carries `suspect-unstable: true` and
`distinct-culprits-on-this-case: N`, and after the fourth report the case stops
producing new `.md` files (repeat sightings still go to the JSONL, so nothing is
discarded — PLAN section 4). It is a label plus a cap rather than a corpus
filter, because dropping a case from a synthesized corpus that is provably free
of both scatter stores and argument aliasing would cost coverage for no gain.

As of this writing no case on this line has reached the threshold.
