# TileLang differential fuzzing -- every difference signature seen

One row per distinct (minimal pass set, pass_configs keys, strength class).
`n` counts every (kernel, input mode, variant) occurrence, not just the
reported one.  The full per-occurrence records are in
`/home/youngzt/fuzz-tilelang/run/results.jsonl`.

updated 2026-08-17 17:50:20

| n | report | minimal pass set | pass_configs | strength | example kernels |
|---|---|---|---|---|---|
| 2 | HIT-0001 | `-` | `tl.config_index_bitwidth` | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | cap_081a26caff856329, cap_401ab8ec5f03638a |
| 1 | HIT-0002 | `-` | `tl.disable_wgmma` | exact-int-accum:reassociation-excluded, gross, pass-class:bit-preserving | cap_0b881ee73fbdf83c |
| 1 | HIT-0003 | `-` | `tl.disable_wgmma` | gross, pass-class:bit-preserving | cap_2bae46f31c2f27cd |
