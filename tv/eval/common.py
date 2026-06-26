"""
Shared glue for the tv evaluation suite.

This module knows two things:
  1. Where the built `triton-tv` / `triton-opt` binaries live.
  2. How to run one validation and read its result.

The `triton-tv` binary contract (see tv/triton-tv.cpp):
  - exit code 0  -> prints "EQUIVALENT"
  - exit code 1  -> prints "NOT EQUIVALENT ..."
  - exit code 2  -> prints "UNKNOWN ..."
  - it also prints timing lines:  "Interpretation: <x> s"  and  "Solver: <x> s"
We read the verdict from the exit code and the timings from stdout.
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Repo root = three levels up from this file: <repo>/tv/eval/common.py
REPO_ROOT = Path(__file__).resolve().parents[2]

# Verdict names, keyed by the binary's exit code.
VERDICT_BY_CODE = {0: "EQUIV", 1: "NEQ", 2: "UNKNOWN"}


@dataclass
class Result:
    """Outcome of one `triton-tv src tgt` run."""
    verdict: str        # EQUIV | NEQ | UNKNOWN | ERROR | TIMEOUT
    exit_code: int      # raw process exit code (-1 on timeout/error)
    interp_s: float     # seconds spent interpreting both programs (nan if absent)
    solver_s: float     # seconds spent in Z3 (nan if absent)
    stdout: str


def _build_dir():
    """The CMake build directory, the same one the Triton build uses."""
    # Prefer the project's own helper so we match the build exactly.
    sys.path.insert(0, str(REPO_ROOT / "python"))
    try:
        import build_helpers  # type: ignore
        return Path(build_helpers.get_cmake_dir())
    except Exception:
        return None


def _glob_one(pattern):
    hits = sorted(REPO_ROOT.glob(pattern))
    return hits[0] if hits else None


def find_binary(name, env_var):
    """Locate a built binary by name.

    Order: explicit env var -> the build_helpers build dir -> glob under build/.
    `name` is "tv/triton-tv" or "bin/triton-opt" (path under the build dir).
    Returns a Path or raises a clear error telling the user how to build it.
    """
    override = os.getenv(env_var)
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise FileNotFoundError(f"{env_var}={override} is not a file")

    bd = _build_dir()
    if bd is not None and (bd / name).is_file():
        return bd / name

    hit = _glob_one(f"build/cmake.*/{name}")
    if hit:
        return hit

    target = Path(name).name
    raise FileNotFoundError(
        f"could not find '{target}'. Build it first, e.g.:\n"
        f"    ninja -C {bd or '<build_dir>'} {target}\n"
        f"or point {env_var} at the binary."
    )


def find_triton_tv():
    return find_binary("tv/triton-tv", "TRITON_TV_BIN")


def find_triton_opt():
    return find_binary("bin/triton-opt", "TRITON_OPT_BIN")


def _parse_time(label, text):
    m = re.search(rf"{label}:\s*([0-9.eE+-]+)\s*s", text)
    return float(m.group(1)) if m else float("nan")


def run_validation(binary, src, tgt, timeout=120):
    """Run `triton-tv src tgt` and return a Result."""
    try:
        proc = subprocess.run(
            [str(binary), str(src), str(tgt)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return Result("TIMEOUT", -1, float("nan"), float("nan"),
                      f"timed out after {timeout}s")

    out = proc.stdout + proc.stderr
    verdict = VERDICT_BY_CODE.get(proc.returncode, "ERROR")
    return Result(
        verdict=verdict,
        exit_code=proc.returncode,
        interp_s=_parse_time("Interpretation", out),
        solver_s=_parse_time("Solver", out),
        stdout=out,
    )
