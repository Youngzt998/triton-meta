"""Low-level helpers shared by the fuzzer and the experiment runner.

This module knows how to:
  * turn a Triton @jit kernel into a *root* unoptimized IR string (ttir/ttgir),
  * run a list of ``triton-opt`` passes on an IR string,
  * compile an IR string (or a Triton kernel) into a launchable variant,
  * compare two output tensors bitwise.

It is deliberately agnostic to the ``tv`` branch: it only uses the public
Triton compiler API (`triton.compile`, `ASTSource`, `IRSource`) plus the
`triton-opt` command line tool.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend
from triton._C.libtriton import ir, passes

# Repo root = the triton checkout (this file lives at <repo>/tv/eq_fuzzing/).
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Target parsing
# --------------------------------------------------------------------------- #
def parse_target(target_str: str, warp_size: int = 32) -> GPUTarget:
    """Turn ``"cuda:90"`` into a ``GPUTarget``."""
    backend_name, arch = target_str.split(":")
    return GPUTarget(backend_name, int(arch), warp_size)


# --------------------------------------------------------------------------- #
# Locate the triton-opt binary
# --------------------------------------------------------------------------- #
def find_triton_opt() -> str:
    """Find the ``triton-opt`` binary.

    Order: ``$TRITON_OPT`` env var, then the worktree build dir, then $PATH.
    """
    env = os.environ.get("TRITON_OPT")
    if env and Path(env).exists():
        return env
    for pattern in ("build/*/bin/triton-opt", "python/build/*/bin/triton-opt"):
        candidates = sorted(REPO_ROOT.glob(pattern))
        if candidates:
            return str(candidates[-1])
    import shutil
    found = shutil.which("triton-opt")
    if found:
        return found
    raise FileNotFoundError(
        "triton-opt not found. Build the worktree with "
        "`pip install -e . --no-build-isolation`, or set $TRITON_OPT.")


def run_triton_opt(ir_text: str, in_ext: str, flags: List[str]) -> str:
    """Run ``triton-opt <flags>`` on ``ir_text`` and return the result text."""
    binary = find_triton_opt()
    with tempfile.TemporaryDirectory() as d:
        in_path = Path(d) / f"in.{in_ext}"
        out_path = Path(d) / "out.mlir"
        in_path.write_text(ir_text)
        cmd = [binary, str(in_path), *flags, "-o", str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"triton-opt failed (exit {proc.returncode})\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr}")
        return out_path.read_text()


def convert_flag(target: GPUTarget, num_warps: int, num_ctas: int = 1) -> str:
    """The ``--convert-triton-to-tritongpu`` flag string (ttir -> ttgir)."""
    tgt = f"{target.backend}:{target.arch}"
    return (f"--convert-triton-to-tritongpu="
            f"target={tgt} num-warps={num_warps} "
            f"num-ctas={num_ctas} threads-per-warp={target.warp_size}")


# --------------------------------------------------------------------------- #
# Root IR generation
# --------------------------------------------------------------------------- #
def make_root_ttir(spec, target: GPUTarget, num_warps: int) -> str:
    """Raw AST -> TTIR for ``spec.fn`` (unoptimized: before make_ttir passes)."""
    src = ASTSource(fn=spec.fn, signature=spec.signature, constexprs=spec.constexprs)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": num_warps})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = src.make_ir(target, options, codegen_fns, module_map, context)
    # Inline helper calls (e.g. tl.zeros -> standard.zeros) so the module is a
    # single function. This is required before convert-triton-to-tritongpu and
    # is structural, not a numeric optimization.
    pm = ir.pass_manager(module.context)
    passes.common.add_inliner(pm)
    pm.run(module, "eq_fuzzing_root_inline")
    return module.str()


def make_root_ir(spec, level: str, target: GPUTarget, num_warps: int) -> Tuple[str, str]:
    """Return ``(ext, ir_text)`` for the unoptimized root at ``level``.

    ``level == "ttir"``  -> raw AST TTIR.
    ``level == "ttgir"`` -> raw TTIR run through only ``convert-triton-to-tritongpu``.
    """
    ttir = make_root_ttir(spec, target, num_warps)
    if level == "ttir":
        return "ttir", ttir
    if level == "ttgir":
        ttgir = run_triton_opt(ttir, "ttir", [convert_flag(target, num_warps)])
        return "ttgir", ttgir
    raise ValueError(f"make_root_ir does not apply to level {level!r}")


# --------------------------------------------------------------------------- #
# Compiled variant (a launchable kernel)
# --------------------------------------------------------------------------- #
class CompiledVariant:
    """Wraps one CompiledKernel plus the ordered non-constexpr arg names."""

    def __init__(self, compiled, arg_order: List[str], label: str):
        self.compiled = compiled
        self.arg_order = arg_order
        self.label = label

    def run(self, inputs: Dict[str, Any], grid) -> None:
        args = [inputs[name] for name in self.arg_order]
        self.compiled[grid](*args)


def _write_ir(ir_text: str, ext: str, workdir: Path, name: str) -> Path:
    path = workdir / f"{name}.{ext}"
    path.write_text(ir_text)
    return path


def build_variant_from_ir(ir_text: str, ext: str, spec, target: GPUTarget,
                          workdir: Path, label: str) -> CompiledVariant:
    """Compile an IR string straight to a launchable kernel (lowering only)."""
    path = _write_ir(ir_text, ext, workdir, label)
    compiled = triton.compile(str(path), target=target)
    return CompiledVariant(compiled, list(spec.signature.keys()), label)


def build_variant_from_triton(spec, options: Dict[str, Any], target: GPUTarget,
                              label: str) -> CompiledVariant:
    """Compile the Triton kernel (full pipeline) with given options."""
    src = ASTSource(fn=spec.fn, signature=spec.signature, constexprs=spec.constexprs)
    compiled = triton.compile(src, target=target, options=options or None)
    return CompiledVariant(compiled, list(spec.signature.keys()), label)


# --------------------------------------------------------------------------- #
# Bitwise comparison
# --------------------------------------------------------------------------- #
_FLOAT_TO_INT = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}


def _int_view(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a tensor's bytes as a same-width integer tensor."""
    t = t.contiguous()
    if t.is_floating_point():
        int_dt = _FLOAT_TO_INT.get(t.dtype)
        if int_dt is None:  # fp8 and friends: 1 byte
            return t.view(torch.int8)
        return t.view(int_dt)
    return t


def tensors_bitwise_equal(a: torch.Tensor, b: torch.Tensor, nan_equal: bool = True) -> bool:
    """True iff ``a`` and ``b`` have identical bits (NaN payloads optional)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    ai, bi = _int_view(a), _int_view(b)
    if nan_equal and a.is_floating_point():
        nan_a, nan_b = torch.isnan(a), torch.isnan(b)
        return bool(((ai == bi) | (nan_a & nan_b)).all().item())
    return torch.equal(ai, bi)
