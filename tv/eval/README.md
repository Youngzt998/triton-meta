# tv evaluation suite

Automated evaluation for the `triton-tv` translation validator. One Python
runner drives the built binary over three kinds of evaluation, one per folder.

```bash
python tv/eval/run_eval.py pairs            # curated pairs gate
python tv/eval/run_eval.py inequal          # inequality-detection gate
python tv/eval/run_eval.py compile-options  # unopt-vs-variant gate
python tv/eval/run_eval.py solver-cost      # timing report
python tv/eval/run_eval.py all              # gates, then the report
```

The runner finds the binary automatically (the same build dir the Triton build
uses). Override with `TRITON_TV_BIN=/path/to/triton-tv` (and `TRITON_OPT_BIN`
for `triton-opt`). Build first if needed:

```bash
ninja -C <build_dir> triton-tv triton-opt
```

`triton-tv src tgt` reports `EQUIVALENT` / `NOT EQUIVALENT` / `UNKNOWN` with
exit code `0` / `1` / `2`. The runner reads the verdict from the exit code.

## `pairs/` — curated validation pairs

Hand-written before/after `.ttir` pairs, each tagged with the verdict the
validator must produce. The manifest is `pairs/cases.tsv`, tab-separated:

```
<src.ttir>    <tgt.ttir>    <EQUIV|NEQ>    optional note
```

`EQUIV` means the pair must be proven equivalent; `NEQ` means it must be proven
not equivalent. Any mismatch is a FAIL and the runner exits non-zero.

**Add a case:** drop the two `.ttir` files in `pairs/` and add one manifest line.

## `inequal/` — inequality detection (soundness)

Pairs that are **genuinely not equivalent** (a kernel vs a mutated version: `x-y`
or `x*y` instead of `x+y`, a wrong `pid*5` offset, …). The validator must report
`NEQ` for every one — this checks it actually *catches* real differences, the
opposite worry from the EQUIV gates. An `EQUIV`/`UNKNOWN` here is a FAIL. Manifest
is `inequal/cases.tsv` (same TSV format; all rows tagged `NEQ`).

**Add a case:** drop a reference and a mutated `.ttir` under `inequal/<kernel>/`
and add one `NEQ` manifest line.

## `compile-options/` — unoptimized standard vs variants

Kernels compiled under different options. For each kernel,
`compile-options/<kernel>/standard.ttir` is the unoptimized reference; every
other `*.ttir` in that folder is a variant. The runner validates each variant
against the standard and expects `EQUIV` — a compiler pass must not change
observable behavior. A `NEQ`/`UNKNOWN` here is a real finding.

**Add a kernel:** generate its folder from a Triton kernel script:

```bash
python tv/eval/compile-options/generate.py python/tutorials/01-vector-add.py
```

This dumps the unoptimized `standard.ttir` and runs `triton-opt` pass pipelines
(`canonicalize`, `cse`, `canonicalize-cse`) to produce the variants.

## `solver-cost/` — timing

Runs every `pairs/` query plus every `compile-options/` variant query, records
each into `solver-cost/results.csv`:

```
case, src, tgt, fp_mode, verdict, interp_s, solver_s
```

`interp_s` / `solver_s` are parsed from the binary's own timing output.
`fp_mode` is `Abstract` for now — the only FP encoding the validator
implements. The column is ready for the planned Real / IntegerRange / FPA
modes: once `triton-tv` grows a `--fp-mode` flag, the same runner sweeps modes
with no schema change. This folder is reporting only; it never gates.
