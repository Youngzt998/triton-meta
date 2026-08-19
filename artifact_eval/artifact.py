#!/usr/bin/env python3
"""Artifact evaluation driver.

One entry point. Pick what to run with --run; every step appends its raw records to
`cache/<table>.jsonl`, and `--export` turns those into the CSV tables in `data/`, so the tables
in the paper can be rebuilt without a GPU.

    python artifact.py --list
    python artifact.py --run gemm.bitmatch --minutes 20
    python artifact.py --run gemm.perf.random,gemm.fusion
    python artifact.py --run gemm                # every gemm step
    python artifact.py --run all                 # every step
    python artifact.py --export                  # cache/*.jsonl -> data/*.csv + FORMAT.md

Environment. Put the repo root on PYTHONPATH (`bitequiv` is not installed) and pin one GPU:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(git rev-parse --show-toplevel) \
        python artifact_eval/artifact.py --run gemm.bitmatch

The numbers in the paper were taken on an NVIDIA GB300 (sm_103) with cuBLASLt 13.1.1. The driver
records the architecture and the cuBLAS version it actually ran against and warns when they
differ, because which kernel cuBLAS picks -- and therefore the bits it returns -- is a function
of both.

Layout. This file holds only what every step shares: the measurement primitives, the streaming
record writer, the CSV/FORMAT.md export, and the CLI. Each step lives in its own file under
`steps/` and declares its own table columns there, so two people can implement two steps without
touching the same file. `steps/_common.py` is the read-only helper library for step authors and
its docstring is the contract a step module has to meet.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")  # committed: results only, as CSV, plus FORMAT.md
CACHE = os.path.join(HERE, "cache")  # local only, gitignored: streaming jsonl, dumps, sweeps

# Step modules import the primitives below as `from artifact import digest, ...`, and so do the
# standalone probes in fusion*/. Run as a script this file is `__main__`, so without this alias
# that import would load a SECOND copy of it -- two `_W` caches, two of everything. Registering
# the alias up front means there is always exactly one.
if __name__ == "__main__":
    sys.modules.setdefault("artifact", sys.modules["__main__"])

# One declaration per table, used for BOTH the CSV header and the format document, so the two
# cannot drift apart. Filled by discover() from each step module's own TABLES, so a step owns its
# columns and adding one is a change to that step's file alone. Only results go here. Anything
# bulky -- PTX dumps, full config sweeps, per-draw records -- stays in cache/ and is regenerated
# by the script rather than committed.
TABLES: dict = {}

# Filled by discover(): step name -> the module implementing it, and group name -> step names.
# A step's group is the part of its name before the first dot, so `--run gemm` picks up a new
# `gemm.*` step with no edit here.
STEPS: dict = {}
GROUPS: dict = {}

PAPER_ARCH = "sm_103"
PAPER_CUBLASLT = (13, 1, 1)

# --------------------------------------------------------------------------------------------
# Measurement primitives. Kept in this file on purpose: a reviewer should be able to read the
# whole measurement in one place. Step modules reach them through `steps/_common.py`, which
# re-exports them unchanged.
# --------------------------------------------------------------------------------------------


def hot_cublas(L, torch, a, b, kind, out_dtype):
    """cuBLAS with everything but the matmul call set up once.

    `cublas_matmul` rebuilds the handle, the four layouts, the preference object and re-runs the
    heuristic on every call; timing it measures host setup, not the GEMM -- up to 7x slower than
    cuBLAS itself on small shapes. The heuristic `results` array must stay alive on the object:
    as a local it is freed when this function returns, the algo pointer dangles, and
    `cublasLtMatmul` then returns non-zero and does nothing, which times as a petaflop.
    """

    class Hot:
        pass

    h = Hot()
    lib = h.lib = L._load_lt()
    M, K = a.shape
    N = b.shape[1]
    h.handle, h.desc = ctypes.c_void_p(), ctypes.c_void_p()
    pref = ctypes.c_void_p()
    L._ck("create", lib.cublasLtCreate(ctypes.byref(h.handle)))
    L._ck("desc", lib.cublasLtMatmulDescCreate(ctypes.byref(h.desc), L._CUBLAS_COMPUTE_32F, L._CUDA_R_32F))
    h.A, h.B, h.C, h.D = L._make_layouts(lib, h.desc, M, N, K, kind, out_dtype, 1.0, 1.0)
    L._ck("pref", lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)))
    ws = ctypes.c_size_t(L._workspace_bytes())
    L._ck("pref-set", lib.cublasLtMatmulPreferenceSetAttribute(pref, L._PREF_MAX_WS, ctypes.byref(ws), 8))
    h.results = (L._HeurResult * 16)()
    ret = ctypes.c_int(0)
    lib.cublasLtMatmulAlgoGetHeuristic(h.handle, h.desc, h.A, h.B, h.C, h.D, pref, 16,
                                       ctypes.cast(h.results, ctypes.c_void_p), ctypes.byref(ret))
    if ret.value == 0:
        raise RuntimeError("cuBLAS returned no algorithm for this shape")
    h.algo = ctypes.cast(ctypes.byref(h.results[0]), ctypes.c_void_p)
    h.out = torch.empty(M, N, device="cuda", dtype=out_dtype)
    h.wsbuf = L._ws_buffer(L._workspace_bytes())
    alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
    pa, pb, pd = (ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(h.out.data_ptr()))

    def run():
        return lib.cublasLtMatmul(h.handle, h.desc, ctypes.cast(ctypes.pointer(alpha), ctypes.c_void_p), pa, h.A, pb,
                                  h.B, ctypes.cast(ctypes.pointer(beta), ctypes.c_void_p), pd, h.C, pd, h.D, h.algo,
                                  ctypes.c_void_p(h.wsbuf.data_ptr()), ctypes.c_size_t(L._workspace_bytes()),
                                  ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))

    h.run = run
    rc = run()
    torch.cuda.synchronize()
    if rc != 0:
        raise RuntimeError(f"cublasLtMatmul returned {rc}; the timed call would be a no-op")
    return h


def graph_ms(torch, fn, flush, reps=25):
    """Device time: capture once, then time replays with L2 flushed between them.

    Timing the Python call instead would bracket it with CUDA events on the stream, so every
    microsecond the GPU idles waiting on the host is counted -- and the two arms here are wildly
    asymmetric on the host side (one ctypes call against a Triton launcher). On a 30 us GEMM that
    measures the launcher. Returns None if the call cannot be captured.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
    except Exception:
        return None
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(reps)]
    for x, y in ev:
        flush.zero_()
        x.record()
        g.replay()
        y.record()
    torch.cuda.synchronize()
    t = sorted(x.elapsed_time(y) for x, y in ev)
    return t[len(t) // 2]


_W = {}


def digest(torch, t):
    """A 64-bit hash computed on the device, so a full device-to-host copy is not needed per
    draw. Bilinear: every byte is weighted by its column and its row, so a change anywhere moves
    the total and two changes cancel only if they align in both weights at once."""
    v = t.contiguous().view(torch.uint8).view(-1)
    n = v.numel()
    w = 1024
    pad = (-n) % w
    if pad:
        v = torch.cat([v, torch.zeros(pad, dtype=torch.uint8, device=v.device)])
    v = v.view(-1, w)
    rows = v.shape[0]
    if "col" not in _W:
        g = torch.Generator(device=v.device).manual_seed(0x5EED)
        _W["col"] = torch.randint(1, 2**30, (w, ), generator=g, device=v.device, dtype=torch.int64)
        _W["row"] = torch.randint(1, 2**30, (1 << 20, ), generator=g, device=v.device, dtype=torch.int64)
    while _W["row"].numel() < rows:
        _W["row"] = torch.cat([_W["row"], _W["row"]])
    return int(((v.to(torch.int64) * _W["col"]).sum(1) * _W["row"][:rows]).sum().item())


def make_inputs(torch, M, N, K, kind, rep, seed):
    """Deterministic in (shape, rep). Even reps are ordinary gaussian; odd reps spread the
    exponents across the dtype's usable range, which is what makes a change in the ORDER of the
    additions visible -- with narrow exponents almost every regrouping rounds to the same bits."""
    g = torch.Generator(device="cuda").manual_seed(seed + 104729 * rep)
    wide = rep % 2 == 1

    def draw(r, c, lo, hi, dt):
        sign = torch.where(torch.rand(r, c, generator=g, device="cuda") < 0.5, -1.0, 1.0)
        return (sign * torch.exp2(torch.rand(r, c, generator=g, device="cuda") * (hi - lo) + lo)).to(dt)

    if kind == "fp8":
        f8 = torch.float8_e4m3fn
        if wide:  # e4m3 normals run to about 448
            return draw(M, K, -5.0, 5.0, f8), draw(N, K, -5.0, 5.0, f8).t()
        a = (torch.randn(M, K, generator=g, device="cuda") / 4).to(f8)
        return a, (torch.randn(N, K, generator=g, device="cuda") / 4).to(f8).t()
    if wide:
        return draw(M, K, -12.0, 8.0, torch.float16), draw(K, N, -12.0, 8.0, torch.float16)
    a = (torch.randn(M, K, generator=g, device="cuda") / 8).to(torch.float16)
    return a, (torch.randn(K, N, generator=g, device="cuda") / 8).to(torch.float16)


# --------------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------------


def env_info():
    import torch

    from bitequiv.cublas_match import cublaslt_version
    from bitequiv.cublas_match.arch import platform
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()[:12]
    except Exception:
        commit = "unknown"
    import triton
    return {
        "arch": platform().name.split()[0],
        "device": torch.cuda.get_device_name(0),
        "cublaslt": ".".join(map(str, cublaslt_version())),
        "triton": triton.__version__,
        "torch": torch.__version__,
        "commit": commit,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def banner(env):
    print(f"  device      {env['device']}  ({env['arch']})")
    print(f"  cuBLASLt    {env['cublaslt']}")
    print(f"  triton      {env['triton']}   torch {env['torch']}   commit {env['commit']}")
    want = ".".join(map(str, PAPER_CUBLASLT))
    if env["arch"] != PAPER_ARCH:
        print(f"  WARNING: the paper's numbers are {PAPER_ARCH}; this is {env['arch']}. Which kernel cuBLAS\n"
              f"           picks depends on the architecture, so both the bit results and the speed will differ.")
    if not env["cublaslt"].startswith(want):
        print(f"  WARNING: the paper's numbers used cuBLASLt {want}; this is {env['cublaslt']}. cuBLAS may pick a\n"
              f"           different algorithm, which changes the bits it returns and what we must reproduce.")


def writer(step):
    """Streaming jsonl into cache/. Kill-safe and never committed; `--export` turns it into the
    CSV that ships. `step` is the TABLE name, so a step that fills two tables opens two."""
    os.makedirs(CACHE, exist_ok=True)
    return open(os.path.join(CACHE, f"{step}.jsonl"), "a")


def export():
    """cache/*.jsonl -> data/*.csv plus data/FORMAT.md, both generated from TABLES."""
    import csv as _csv
    os.makedirs(DATA, exist_ok=True)
    made = []
    for step, spec in TABLES.items():
        src = os.path.join(CACHE, f"{step}.jsonl")
        if not os.path.exists(src):
            continue
        cols = [c for c, _, _ in spec["cols"]]
        dst = os.path.join(DATA, f"{step.replace('.', '_')}.csv")
        n = 0
        with open(dst, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(cols)
            for line in open(src):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                row = []
                for c in cols:
                    v = r.get(c, "")
                    if isinstance(v, (list, tuple)):
                        v = " ".join(map(str, v))
                    elif isinstance(v, dict):
                        v = json.dumps(v, separators=(",", ":"))
                    row.append("" if v is None else v)
                w.writerow(row)
                n += 1
        made.append((dst, n))
        print(f"  {os.path.basename(dst):28} {n} rows")
    doc = os.path.join(DATA, "FORMAT.md")
    with open(doc, "w") as f:
        f.write("# Data format\n\nOne CSV per experiment, first line is the header. Generated by\n"
                "`artifact.py --export` from the streaming records in `cache/`, which are not\n"
                "committed. Empty cell means the field does not apply to that row.\n\n"
                "`env.csv` records the machine each run was taken on. Which kernel cuBLAS picks --\n"
                "and therefore the bits it returns -- depends on the architecture and the cuBLASLt\n"
                "version, so a row is only comparable to another row with the same two.\n\n"
                "A table listed here whose CSV is absent belongs to a step that has not been\n"
                "implemented yet; the columns are declared so the shape is fixed in advance.\n")
        for step, spec in TABLES.items():
            f.write(f"\n## {step.replace('.', '_')}.csv\n\n{spec['doc']}\n\n")
            f.write("| column | type | meaning |\n|---|---|---|\n")
            for c, t, d in spec["cols"]:
                f.write(f"| `{c}` | {t} | {d} |\n")
        f.write("\n## env.csv\n\nOne row per invocation.\n\n| column | type | meaning |\n|---|---|---|\n")
        for c, d in (("arch", "compute capability, e.g. sm_103"), ("device", "GPU name"),
                     ("cublaslt", "libcublasLt version actually loaded"), ("triton", "Triton version"),
                     ("torch", "PyTorch version"), ("commit", "repository commit"),
                     ("when", "local time the run started"), ("steps", "steps run in this invocation")):
            f.write(f"| `{c}` | str | {d} |\n")
    print(f"  {os.path.basename(doc):28} written")
    src = os.path.join(CACHE, "env.jsonl")
    if os.path.exists(src):
        rows = [json.loads(x) for x in open(src) if x.strip()]
        with open(os.path.join(DATA, "env.csv"), "w", newline="") as fh:
            w = _csv.writer(fh)
            cols = ["arch", "device", "cublaslt", "triton", "torch", "commit", "when", "steps"]
            w.writerow(cols)
            for r in rows:
                w.writerow([" ".join(r[c]) if isinstance(r.get(c), list) else r.get(c, "") for c in cols])
        print(f"  {'env.csv':28} {len(rows)} rows")
    return made


# --------------------------------------------------------------------------------------------
# Step discovery. One step, one file under steps/ -- see steps/_common.py for the contract.
# --------------------------------------------------------------------------------------------


def _step_modules():
    """Import every `steps/<x>.py` and return them in listing order.

    Discovery is by file, so adding or implementing a step touches that one file and nothing
    else. The file name has to be the step name with `.` and `-` turned into `_`; checking it
    here turns "one step, one file" from a convention people drift from into an error.
    """
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import steps
    folder = os.path.dirname(os.path.abspath(steps.__file__))
    mods = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        modname = fname[:-3]
        mod = importlib.import_module(f"steps.{modname}")
        want = mod.NAME.replace(".", "_").replace("-", "_")
        if want != modname:
            raise RuntimeError(f"steps/{fname} declares NAME={mod.NAME!r}; the file must be steps/{want}.py")
        mods.append(mod)
    mods.sort(key=lambda m: (getattr(m, "ORDER", 1000), m.NAME))
    return mods


def discover():
    """Fill STEPS, GROUPS and TABLES from the modules in steps/. Idempotent."""
    if STEPS:
        return STEPS
    for mod in _step_modules():
        STEPS[mod.NAME] = mod
        GROUPS.setdefault(mod.NAME.split(".")[0], []).append(mod.NAME)
        for table, spec in getattr(mod, "TABLES", {}).items():
            if table in TABLES:
                raise RuntimeError(f"table {table!r} is declared by two steps; a table has one owner")
            TABLES[table] = spec
    return STEPS


def print_listing():
    """`--list`. Says outright which steps are implemented and which are still a design, so a
    reviewer sees the gaps instead of wondering why a step printed nothing."""
    print("steps:")
    group = None
    for name, mod in STEPS.items():
        g = name.split(".")[0]
        if g != group:
            group = g
            print()
        state = "implemented" if getattr(mod, "IMPLEMENTED", False) else "PLACEHOLDER"
        print(f"  {name:24} {state:12} {mod.DESCRIPTION}")
    todo = [n for n, m in STEPS.items() if not getattr(m, "IMPLEMENTED", False)]
    print(f"\n  {len(STEPS) - len(todo)}/{len(STEPS)} implemented. A PLACEHOLDER prints its measurement design,")
    print("  writes no records and exits 0; its table columns are already declared in data/FORMAT.md.")
    print("\ngroups:")
    for k, v in GROUPS.items():
        print(f"  {k:24} {' '.join(v)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="", help="comma-separated step or group names; 'all' for everything")
    p.add_argument("--list", action="store_true")
    p.add_argument("--export", action="store_true", help="cache/*.jsonl -> data/*.csv + FORMAT.md")
    p.add_argument("--minutes", type=float, default=20, help="budget for the sampling steps")
    p.add_argument("--reps", type=int, default=10, help="input draws per shape")
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--max-bytes", type=int, default=6 * 2**30)
    args = p.parse_args()

    discover()

    if args.export:
        sys.path.insert(0, ROOT)
        print("exporting results to", DATA)
        export()
        return 0

    if args.list or not args.run:
        print_listing()
        return 0

    wanted = []
    for tok in args.run.split(","):
        tok = tok.strip()
        if tok == "all":
            wanted += list(STEPS)
        elif tok in GROUPS:
            wanted += GROUPS[tok]
        elif tok in STEPS:
            wanted.append(tok)
        else:
            print(f"unknown step {tok!r}; --list to see them")
            return 2

    sys.path.insert(0, ROOT)
    env = env_info()
    print("artifact evaluation")
    banner(env)
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, "env.jsonl"), "a") as f:
        f.write(json.dumps({**env, "steps": wanted}) + "\n")
    for name in wanted:
        STEPS[name].run(args, env)
    print("\nstreaming records in", CACHE)
    export()
    print("\nresults in", DATA)
    return 0


if __name__ == "__main__":
    sys.exit(main())
