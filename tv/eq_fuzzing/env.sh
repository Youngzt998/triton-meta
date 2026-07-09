# Machine-specific runtime glue for eq_fuzzing on this box.
# Source it, then run: "$EQF_PY" -m eq_fuzzing.runner ...
#
# Why this exists: the tv checkout's .venv has triton (built here) but NOT torch,
# and the offline build did not fetch the CUDA toolchain. We borrow torch from
# the fbsource `beta` venv (via PYTHONPATH, with this checkout's python/ dir
# FIRST so `import triton` resolves to the tv checkout) and point Triton at the
# shared CUDA tools under ~/.triton. Runtime-only glue; the eq_fuzzing code
# itself only uses the public Triton API + the triton-opt binary.

# Triton repo root (this script is at <repo>/tv/eq_fuzzing/env.sh).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# External deps (outside the triton tree, so necessarily absolute on this box).
BETA_SP=/data/users/youngzt/fbsource/third-party/triton/beta/triton/.venv/lib/python3.12/site-packages

export EQF_PY="$ROOT/.venv/bin/python"
# $ROOT/python -> triton (this checkout, wins over beta's .pth);
# $ROOT/tv     -> makes the `eq_fuzzing` package importable;
# $BETA_SP     -> torch + numpy.
export PYTHONPATH="$ROOT/python:$ROOT/tv:$BETA_SP"

# CUDA toolchain (same archives the beta env uses)
export TRITON_PTXAS_PATH="/home/youngzt/.triton/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.9.86-archive/bin/ptxas"
export TRITON_CUOBJDUMP_PATH="/home/youngzt/.triton/nvidia/cuobjdump/cuda_cuobjdump-linux-x86_64-13.1.80-archive/bin/cuobjdump"
export TRITON_NVDISASM_PATH="/home/youngzt/.triton/nvidia/nvdisasm/cuda_nvdisasm-linux-x86_64-13.1.80-archive/bin/nvdisasm"
export TRITON_CUDACRT_PATH="/home/youngzt/.triton/nvidia/nvcc/cuda_crt-linux-x86_64-13.1.80-archive/include"
export TRITON_CUDART_PATH="/home/youngzt/.triton/nvidia/cudart/cuda_cudart-linux-x86_64-13.1.80-archive/include"
