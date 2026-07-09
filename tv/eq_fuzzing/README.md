# eq_fuzzing — quickstart

Fuzzer-based check of Triton **compilation correctness**: compare two
compilations of the same kernel on many random inputs, bitwise. Lives at
`tv/eq_fuzzing/` — the empirical GPU counterpart to the SMT translation
validator in `tv/`.

## Setup (once)
Build the tv checkout so `import triton` and `triton-opt` come from it:

```bash
cd /home/youngzt/tv/triton
pip install -e . --no-build-isolation      # build in parallel (-j100)
```

(See `env.sh` for the exact CUDA/LLVM env used on this box.)

## Run
Command-line paths are relative to the **triton repo root**, so you can run from
any directory. `env.sh` sets up the interpreter, PYTHONPATH, and CUDA tools.

```bash
source tv/eq_fuzzing/env.sh   # sets $EQF_PY, PYTHONPATH, CUDA tool paths

# TTGIR: does an optimization pipeline change results vs the unoptimized root?
"$EQF_PY" -m eq_fuzzing.runner \
  --level ttgir --mode candidate-vs-reference \
  --kernels tv/eq_fuzzing/kernels/basic.py \
  --passes tv/eq_fuzzing/passes/ttgir_opt.txt \
  -R 200 --target cuda:90 \
  --checkpoint tv/eq_fuzzing/checkpoints/ttgir.json
```

`-R N` = number of random launches required to declare equivalence.
`--kernels` defaults to `tv/eq_fuzzing/kernels/basic.py` if omitted.

### Modes
- `candidate-vs-reference` (default): candidate pipeline (`--passes`) vs the
  unoptimized, straight-lowered root.
- `ablation`: `--passes` is one pipeline; compares full vs full-minus-one-pass
  for each pass, to pinpoint which pass changes results.
- `pairwise`: `--passes` (A) vs `--passes-b` (B).

### Levels
- `--level ttgir` — varies TTGIR passes.
- `--level ttir` — varies TTIR passes (convert-to-ttgpu appended automatically).
- `--level triton` — varies compile options instead of passes: use
  `--triton-opts "num_warps=8,num_stages=3"` (and `--triton-opts-b` for
  pairwise).

See `pass_classification.md` for which passes are expected to preserve bits vs.
change them by design, plus the ready-made `passes/bit_preserving.txt` and
`passes/numerics_changing.txt`.

## Pause / resume
Press Ctrl-C to stop; a checkpoint is written. Resume with the same command plus
`--resume`. Because inputs are seeded by `seed + iteration`, resume replays the
exact sequence.

## When a mismatch is found
The run stops (unless `--keep-going`) and writes, under `--out-dir`
(default `tv/eq_fuzzing/inequiv/<kernel>__<task>/`):
`ref.ttgir`, `cand.ttgir`, `input.pt`, `out_ref.pt`, `out_cand.pt`, `meta.json`.
Load them to see which pass/option changed the numerics.

Example self-check (expects a mismatch on matmul from tf32 tensor cores):

```bash
"$EQF_PY" -m eq_fuzzing.runner \
  --level ttgir --mode candidate-vs-reference \
  --only matmul \
  --passes tv/eq_fuzzing/passes/ttgir_matmul_tf32.txt \
  -R 50 --checkpoint tv/eq_fuzzing/checkpoints/mm.json
```

## Add your own kernels
Copy `kernels/basic.py`, fill in `KERNELS` with `KernelSpec` entries, and point
`--kernels` at your file. A spec gives the `@triton.jit` fn, its non-constexpr
`signature`, `constexprs`, a seeded `gen_inputs`, a `grid`, and `out_args`.
