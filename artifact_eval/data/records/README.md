# The raw records

Every row every step wrote, one gzipped file per table, as it stood when the artifact was frozen.
3.1 MB compressed, 46 MB open.

These are what `data/*.csv` and `data/summary.csv` are made from. They are here because the
summaries cannot answer the questions a per-row file can: the spread behind a geometric mean, the
single worst shape and what was odd about it, which model and which layer a row came from, which
draw of the ten differed. While the runs were happening these lived in `artifact_eval/cache/`,
which is not committed because it is regenerated — regenerating it needs this machine, and the
machine goes away.

**Read `../FORMAT.md` before you compute anything from them.** These are append logs, not tables:
a shape measured twice is in here twice, and every step's own report de-duplicates on a key before
it averages. `FORMAT.md` names the key per table.

## Reading one

One JSON object per line, no header, keys matching the columns in `../FORMAT.md`.

```
zcat data/records/gemm.fusion.jsonl.gz | head -1 | python3 -m json.tool
zcat data/records/inner_tree.layout.jsonl.gz | wc -l
```

## Putting them back so the steps can use them

`--export` reads `cache/`, so unpack there and every CSV in `data/` regenerates exactly:

```
cd <repo root>
mkdir -p artifact_eval/cache
for f in artifact_eval/data/records/*.gz; do
    case "$f" in *.tar.gz) tar xzf "$f" -C artifact_eval/cache ;;
                 *) gunzip -c "$f" > "artifact_eval/cache/$(basename "${f%.gz}")" ;; esac
done
python3 artifact_eval/artifact.py --export
```

Any Python 3 will do for that last line. `--export` reads the unpacked records and writes the
CSVs; it imports neither torch nor Triton, needs no GPU and no `PYTHONPATH`. Verified with this
machine's stock `/usr/bin/python3` (3.9, no torch installed): all nine CSVs and `FORMAT.md` came
back byte-identical to the committed ones.

A step re-run against an unpacked `cache/` also resumes from it rather than starting over, which
is the cheap way to add to a partial sweep instead of repeating it.

## Remaking this directory after a further run

```
cd artifact_eval
for f in cache/*.jsonl cache/gemm.perf.random.shapes.json cache/inner_tree.layout.header.txt; do
    gzip -9 -c "$f" > "data/records/$(basename "$f").gz"
done
tar czf data/records/gemm.fusion.kernels.tar.gz -C cache gemm.fusion.kernels
```

## What is in each file

| file | rows | one row is |
|---|---|---|
| `gemm.bitmatch.jsonl.gz` | 155,310 | one random shape. The **union of two runs** under the six-regime draw — the 62,025-shape rerun, an earlier 91,581-shape run, and 1,704 declines. Not comparable shape for shape across the two; read one run's report, not the union, unless you mean the union |
| `gemm.bitmatch.pre-regimes.jsonl.gz` | 67,306 | the same, from the **earlier four-regime draw**, before `gemv` and `simt` were added. Superseded, and kept because it is the record of what the narrower draw covered: over these 67,306 shapes only 57 reached a vector kernel and none at all reached the CUDA-core chain GEMM, which is why the two regimes were added |
| `gemm.perf.random.jsonl.gz` | 6,995 | one (shape, arm). 1,382 of the 1,400 drawn shapes, five arms each |
| `gemm.perf.random.oldfloor.jsonl.gz` | 6,995 | the same 6,910 (shape, arm) keys **before the replay floor was re-measured**. Superseded by the file above; kept so the two can be diffed |
| `gemm.perf.random.shapes.json.gz` | 1,400 shapes | the shape list itself. Regenerated deterministically from seed 20260816, so this is a convenience, not evidence |
| `gemm.perf.static.jsonl.gz` | 5,345 | one (model, layer, pairing, M, N, K, dtype, arm). 698 cases over 590 distinct shapes |
| `gemm.perf.static.contaminated.jsonl.gz` | 220 | rows taken while **another process was on the GPU**, moved aside and re-measured. Kept so the re-take can be checked against what it replaced |
| `gemm.fusion.jsonl.gz` | 96 | one (case, dtype), with all four arms on the row |
| `gemm.fusion.kernels.tar.gz` | 31 files | the fused kernel **source** spliced for each row, as it was compiled. Regenerated on every run from `fusion_oracle/inductor_kernel.EPI_SRC`; here so you can read what ran without running it |
| `gemm.cublas-bug.jsonl.gz` | 13 | one probe of the cuBLAS split-K tail defect: `K`, what cuBLAS summed, whether it fired |
| `gemm.perf.jsonl.gz` | 8 | the **superseded `gemm.perf` step**, whose unconstrained arm was a Triton configuration sweep we had written ourselves. Replaced by `gemm.perf.random` and `gemm.perf.static`, whose unconstrained arms are `torch.compile`. Kept only as the record that it existed and what it said |
| `inner_tree.bitmatch.jsonl.gz` | 104 | one cell — a fixed (kernel, dtype, `enable_fp_fusion`, ordering) — with the 36-configuration sweep summarised on it |
| `inner_tree.layout.jsonl.gz` | 1,728 | one (kernel, dtype, configuration), three arms timed on it |
| `inner_tree.layout.header.txt.gz` | — | the header block of that step's report, kept beside its rows |
| `checker.corpus.jsonl.gz` | 103 | 93 group rows at `cap = 0`, 4 at `cap = 48`, and the six recomputed total rows. **Filter on `cap` first**: a capped row is a strided sample of its group, not the group |
| `env.jsonl.gz` | 83 | one invocation: GPU, cuBLASLt version, Triton and torch versions, commit, and which steps it ran. `data/env.csv` is this file as CSV |

## What is deliberately not here

* `cache/gemm.perf.random.claims/` and `cache/gemm.perf.static.claims/` — 7.9 MB of lock files for
  the protocol that lets several workers share a sweep. They are coordination state, not results,
  and a stale claim in a fresh tree would only confuse a resumed run.
* Anything bulky the steps regenerate: PTX dumps, per-draw tensors, the 11 GB checker corpus.
  `checker.corpus`'s corpus is an input, not a result, and `corpus_builder/` rebuilds it.
