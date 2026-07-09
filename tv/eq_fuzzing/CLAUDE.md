# eq_fuzzing — compiler equivalence fuzzing

## What this is
A fuzzer that checks Triton **compilation correctness**: it takes two different
compilations of the *same* kernel, feeds them many random inputs, launches both,
and compares outputs **bitwise**. All `R` launches match => "empirically
equivalent". First mismatch => stop and save everything for a human to inspect.

## Where this lives
This is `tv/eq_fuzzing/` on the `tv` branch — the empirical, GPU-runtime
counterpart to the SMT translation validator in `tv/` (the validator proves
equivalence with Z3; this fuzzer checks it empirically by launching real
kernels). Per `tv/CLAUDE.md`, all `tv/` work stays inside `tv/` and is pushed
only to `tv` / `tv-trials`.

It depends only on the public Triton compiler API + the `triton-opt` binary (not
on the SMT validator code), so it stays self-contained and portable.

## The three levels
`fuzzer.check_eq(kernel1, kernel2, level, spec)`, `level` = ext of the IR you
pass:
- `triton`: `kernel1/2` are `(fn_ignored, options_dict)`; kernel comes from
  `spec.fn`. Full-pipeline comparison driven by compile options.
- `ttir` / `ttgir`: `kernel1/2` are IR strings; they are compiled straight to
  cubin.

## Key trick — isolating the compared stage
Compiling from a `.ttgir` runs only `make_llir/ptx/cubin` (pure lowering, no
TTGIR optimization). So the runner always hands **TTGIR** to `triton.compile`
and lets it only lower; both sides get identical lowering and the *only*
difference is the `triton-opt` passes run before. This is why the runner calls
the fuzzer with `"ttgir"` even for a `--level ttir` experiment (it lowers the
ttir root to ttgir with `convert-triton-to-tritongpu` first).

- ttgir experiment: reference = unoptimized root ttgir straight-lowered;
  candidate = root ttgir + passes from the `.txt`.
- ttir experiment: reference = root ttir + only `convert-triton-to-tritongpu`;
  candidate = root ttir + txt passes (+ convert appended if absent).

## Files
- `fuzzer.py` — `EquivalenceFuzzer` (the improvable core: input gen + compare).
- `compile_utils.py` — root-IR extraction, `triton-opt` driver, `CompiledVariant`,
  bitwise compare.
- `runner.py` — CLI: modes, checkpoint/resume, artifact dumping.
- `kernel_spec.py` — `KernelSpec` dataclass.
- `kernels/basic.py` — starter kernel registry (`KERNELS`).
- `passes/*.txt` — one `triton-opt` flag per line; `#` comments allowed.

## Prerequisites
- Python-only code (no rebuild for changes here). The one hard requirement is a
  built `triton-opt` matching the installed `triton`: build the tv checkout with
  `pip install -e . --no-build-isolation` (see `env.sh` for the CUDA/LLVM build
  env used on this box). Build with `-j100`.
- Binary lookup order for `triton-opt`: `$TRITON_OPT`, then `build/*/bin/triton-opt`,
  then `python/build/*/bin/triton-opt`, then `$PATH`.

## Determinism caveats
- Bitwise diffs from float reassociation (reduction order, FMA contraction,
  tf32) are an **expected signal to inspect**, not automatically a bug. e.g.
  `--tritongpu-accelerate-matmul` on `matmul` will differ.
- Only test deterministic kernels: no atomics, no cross-program reductions.
- The ttir root is raw AST->TTIR; the root generator runs the MLIR inliner so
  standard helpers (`tl.zeros`, etc.) and other `@jit` callees are inlined before
  `convert-triton-to-tritongpu`.

## Running
See `README.md`. Default target here is `cuda:90` (H100). If a run hangs, run
`third_party/tlx/killgpu.sh`.
