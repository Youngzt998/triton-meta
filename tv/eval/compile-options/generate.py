"""
Generate a compile-options test folder for one Triton kernel.

For each kernel we produce a folder  compile-options/<kernel>/  holding:
  standard.ttir       - the UNOPTIMIZED TTIR (the correct reference)
  <pipeline>.ttir     - the standard run through a triton-opt pass pipeline

The unoptimized kernel is the ground-truth standard: every variant is later
validated against it (run_eval.py compile-options) and must come back EQUIV,
because a compiler pass must not change observable behavior.

Usage:
    python tv/eval/compile-options/generate.py <kernel_script.py>
    python tv/eval/compile-options/generate.py <unoptimized.ttir> [name]

Two ways to get the standard:
  - <kernel_script.py>: compile the kernel and dump its raw (unoptimized) TTIR.
    The capture trick (monkey-patching make_ttir to dump the module before any
    pass runs) is the same one used by tv/test/make_unoptimized_ttir.py. This
    path needs a working Triton backend (GPU driver / ptxas).
  - <unoptimized.ttir>: use an already-captured unoptimized TTIR directly as the
    standard. Works fully offline (no GPU, no ptxas) — only triton-opt is run.

Either way we then run triton-opt pass pipelines on the standard to produce the
variant files, each of which must later validate EQUIV against the standard.
"""

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so we can import common
import common  # noqa: E402

# Pass pipelines to apply to the unoptimized standard. Each becomes one variant
# file named after its key. These are plain MLIR passes always registered in
# triton-opt, so they work without a GPU and never change TTIR semantics.
PIPELINES = {
    "canonicalize": ["-canonicalize"],
    "cse": ["-cse"],
    "canonicalize-cse": ["-canonicalize", "-cse"],
}


def _extract_kernel_name(mod_str):
    m = re.search(r"tt\.func\s+public\s+@(\w+)", mod_str)
    if m:
        return m.group(1)
    m = re.search(r"func\.func\s+(?:public\s+)?@(\w+)", mod_str)
    return m.group(1) if m else "unknown"


def _capture_standard(script_path):
    """Compile the kernel(s) in `script_path`, dumping raw TTIR per kernel.

    Returns a dict {kernel_name: Path to its <kernel>/ folder}.
    """
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    import triton
    from triton.backends import backends
    from triton.compiler.compiler import ASTSource, compile as triton_compile
    from triton.runtime import driver

    made = {}

    def make_dump_wrapper(original_make_ttir):
        @staticmethod
        def _dump_raw(mod, metadata, opt, capability):
            mod_str = str(mod)
            name = _extract_kernel_name(mod_str)
            out_dir = HERE / name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "standard.ttir").write_text(mod_str)
            made[name] = out_dir
            print(f"  standard -> {out_dir / 'standard.ttir'}")
            return original_make_ttir(mod, metadata, opt, capability)
        return _dump_raw

    for _name, backend in backends.items():
        cls = backend.compiler
        cls.make_ttir = make_dump_wrapper(cls.make_ttir)

    # Load only imports + kernel defs from the user script.
    tree = ast.parse(Path(script_path).read_text())
    tree.body = [n for n in tree.body
                 if isinstance(n, (ast.Import, ast.ImportFrom,
                                   ast.FunctionDef, ast.AsyncFunctionDef))]
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, script_path, "exec"), namespace)
    kernels = {n: o for n, o in namespace.items()
               if isinstance(o, triton.JITFunction)}
    if not kernels:
        sys.exit(f"No triton.jit kernels found in {script_path}")

    # Use the active GPU target if there is one; otherwise fall back to a fixed
    # CUDA target. We only need TTIR (captured before any pass), so the exact
    # target does not affect the dumped standard.
    try:
        target = driver.active.get_current_target()
    except Exception:
        from triton.backends.compiler import GPUTarget
        target = GPUTarget("cuda", 80, 32)
        print("  (no active GPU driver; using fixed cuda/80 target)")

    for name, kernel in kernels.items():
        print(f"Compiling kernel: {name}")
        sig, constexprs = {}, {}
        for param in kernel.params:
            if param.is_constexpr:
                constexprs[param.name] = 1024
            elif "ptr" in param.name.lower():
                sig[param.name] = "*fp32"
            else:
                sig[param.name] = "i32"
        triton_compile(ASTSource(kernel, signature=sig, constexprs=constexprs),
                       target=target)

    return made


def _make_variants(kernel_dir):
    """Run triton-opt pipelines on standard.ttir, writing one file each."""
    triton_opt = common.find_triton_opt()
    standard = kernel_dir / "standard.ttir"
    for key, passes in PIPELINES.items():
        out = kernel_dir / f"{key}.ttir"
        proc = subprocess.run(
            [str(triton_opt), str(standard), *passes],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  [skip] {key}: triton-opt failed:\n{proc.stderr}")
            continue
        out.write_text(proc.stdout)
        print(f"  variant -> {out}")


def _standard_from_ttir(ttir_path, name=None):
    """Use an already-captured unoptimized TTIR as the standard (offline path)."""
    src = Path(ttir_path)
    if name is None:
        # add_kernel_unoptimized.ttir -> add_kernel
        name = re.sub(r"_unoptimized$", "", src.stem)
    out_dir = HERE / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "standard.ttir").write_text(src.read_text())
    print(f"  standard -> {out_dir / 'standard.ttir'}")
    return {name: out_dir}


def main():
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <kernel_script.py | unoptimized.ttir> [name]")
    inp = os.path.abspath(sys.argv[1])

    if inp.endswith(".ttir"):
        name = sys.argv[2] if len(sys.argv) > 2 else None
        made = _standard_from_ttir(inp, name)
    else:
        made = _capture_standard(inp)

    for name, kernel_dir in made.items():
        print(f"Building variants for {name}")
        _make_variants(kernel_dir)


if __name__ == "__main__":
    main()
