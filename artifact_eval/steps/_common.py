"""Everything a step module is allowed to share. Read this; do not edit it.

WHAT A STEP MODULE LOOKS LIKE
-----------------------------
A step is one file, `steps/<name>.py`, where `<name>` is the step name with `.` and `-` turned
into `_` (`gemm.cublas-bug` -> `steps/gemm_cublas_bug.py`). `artifact.py` checks that, so the
file name and the step name cannot drift apart. The module declares:

    NAME         str   the step name, e.g. "gemm.perf.random". The part before the first dot is
                       the group, so `--run gemm` picks the step up with no edit anywhere else.
    ORDER        int   position in `--list` and in FORMAT.md. Sections have ranges; see the
                       numbers already in use rather than picking a fresh one at random.
    DESCRIPTION  str   one line, shown by `--list`.
    IMPLEMENTED  bool  False means the step prints its design and writes nothing.
    TABLES       dict  table name -> {"doc": str, "cols": [(name, type, description), ...]}.
                       `artifact.py` merges these into the one global declaration that generates
                       both the CSV header and `data/FORMAT.md`, so the two cannot disagree. A
                       table has exactly one owning step. Declare the columns even while the step
                       is a placeholder: the shape is then fixed before anyone measures anything,
                       and whoever implements the step never has to edit a shared file.
    run(args, env)     do the work. `args` is the parsed CLI namespace (`minutes`, `reps`,
                       `seed`, `max_bytes`); `env` is the dict from `artifact.env_info()`.

IMPORT AT CALL TIME, NOT AT IMPORT TIME
---------------------------------------
`artifact.py` imports every step module just to build `--list`, and `--list` must work on a
machine with no GPU. So import torch, triton and bitequiv INSIDE `run()`, never at module level.
Nothing in this file imports them either.

WRITING RECORDS
---------------
    out = writer("gemm.perf.random")        # cache/gemm.perf.random.jsonl, opened for append
    out.write(json.dumps(rec) + "\\n")
    out.flush()                             # flush per record: a killed run keeps what it had

One line per row, keys matching the declared columns. A key that is missing exports as an empty
cell, so a row does not have to carry every column. `--export` reads these back into
`data/<table>.csv`.

WHAT IS ALREADY MEASURED FOR YOU
--------------------------------
    hot_cublas(L, torch, a, b, kind, out_dtype)   cuBLAS with the handle, layouts, preference and
                                                  heuristic built once; `.run()` is the bare call
    graph_ms(torch, fn, flush, reps=25, batch=1)  device time for ONE call: `batch` of them
                                                  captured into one CUDA graph, median replay
                                                  with L2 flushed between replays, divided by
                                                  `batch`. `fn` may be a list, and then the
                                                  batch cycles through it -- one operand copy
                                                  per call, so no call warms the next one's L2
    pick_batch(torch, calls, flush, floor, ...)   what `batch` this shape should use. 1 when the
                                                  kernel is already far above the replay floor
    digest(torch, tensor)                         64-bit device-side hash of the raw bytes; equal
                                                  digests means byte-identical output
    make_inputs(torch, M, N, K, kind, rep, seed)  deterministic operands; even `rep` is ordinary
                                                  gaussian, odd `rep` spreads the exponents wide
    draw_shape(rng)                               the random shape draw `gemm.bitmatch` uses
    draw_shape_with_family(rng)                   the same draw, also returning which family the
                                                  shape came from
    print_design(...)                             what a placeholder prints

Re-implementing any of these in a step is the mistake this file exists to prevent -- above all
the two in `artifact.py` whose docstrings explain why they look the way they do. Timing cuBLAS
without the hot closure measures host setup, and timing on the host instead of the device
measures the Triton launcher.
"""
from __future__ import annotations

import textwrap

# Re-exported unchanged from the driver so a step has one place to import from. The alias
# `artifact.py` registers for itself means this resolves to the same module object whether the
# driver was run as a script or imported.
from artifact import (CACHE, DATA, GRAPH_BATCH_CAP, GRAPH_FLOOR_SHARE, HERE, ROOT, digest, graph_ms, hot_cublas,
                      make_inputs, pick_batch, writer)

__all__ = [
    "CACHE", "DATA", "GRAPH_BATCH_CAP", "GRAPH_FLOOR_SHARE", "HERE", "ROOT", "digest", "graph_ms", "hot_cublas",
    "make_inputs", "pick_batch", "writer", "SHAPE_FAMILIES", "draw_shape", "draw_shape_with_family", "print_design"
]

# The regimes `draw_shape_with_family` draws from, in the order it weights them. `gemm.perf.*`
# groups its rows by these, so the name a row carries and the name the draw used are the same
# string rather than two lists that fall out of step.
SHAPE_FAMILIES = ("square", "thin", "deepk", "decode")


def draw_shape_with_family(rng):
    """A random GEMM shape, plus which family it came from.

    Deliberately includes very deep K. The cuBLAS split-K tail defect only appears there, and a
    sweep that never reaches it cannot support the claim "every mismatch is cuBLAS's own bug"
    -- it would simply never see one.
    """
    kind = rng.choice(["fp16", "fp16", "fp8"])
    regime = rng.choice(["square", "square", "thin", "deepk", "deepk", "decode"])
    if regime == "square":
        M, N, K = rng.randint(256, 8192), rng.randint(256, 16384), rng.randint(256, 16384)
    elif regime == "thin":
        M, N, K = rng.randint(16, 256), rng.randint(1024, 20000), rng.randint(1024, 20000)
    elif regime == "deepk":
        M, N, K = rng.randint(16, 512), rng.randint(64, 2048), rng.randint(20000, 300000)
    else:
        M, N, K = rng.randint(1, 64), rng.randint(1024, 16384), rng.randint(1024, 16384)
    if kind == "fp8":  # cuBLAS refuses fp8 unless every dimension is a multiple of 16
        M, N, K = (max(16, x // 16 * 16) for x in (M, N, K))
    elif rng.random() < 0.5:
        M, N, K = (max(16, x // 16 * 16) for x in (M, N, K))
    return M, N, K, kind, regime


def draw_shape(rng):
    """The four-value form `gemm.bitmatch` has always used. Same draw, same random stream, so
    the committed bitmatch results stay reproducible from the same seed."""
    M, N, K, kind, _ = draw_shape_with_family(rng)
    return M, N, K, kind


_WIDTH = 96


def _wrap(text, indent="    "):
    """Reflow a design paragraph to the terminal, but leave a hand-laid-out block alone.

    A block where some line is indented further than the rest is a small table or a list that
    was written that way on purpose; reflowing it would run the columns together.
    """
    blocks = []
    for block in textwrap.dedent(str(text).strip("\n")).split("\n\n"):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if any(ln.startswith(" ") for ln in lines):
            blocks.append("\n".join(indent + ln for ln in lines))
        else:
            joined = " ".join(block.split())
            blocks.append(textwrap.fill(joined, width=_WIDTH, initial_indent=indent, subsequent_indent=indent))
    return "\n\n".join(blocks)


def print_design(name, claim, method, cost, see, judge, notes=None):
    """What an unimplemented step prints instead of measuring.

    The five headings are the ones `README.md` uses for a finished step, so a placeholder reads
    like the section it will become and a reviewer can tell the design apart from the result.
    Prints, writes no records, returns None.
    """
    print(f"\n[{name}] PLACEHOLDER -- the measurement is designed, the code is not written yet.")
    for title, body in (("Claim", claim), ("Method", method), ("Cost", cost), ("What you should see", see),
                        ("How to judge it", judge), ("Notes", notes)):
        if not body:
            continue
        print(f"\n  {title}")
        print(_wrap(body))
    print("\n  Nothing was measured and no records were written.")
