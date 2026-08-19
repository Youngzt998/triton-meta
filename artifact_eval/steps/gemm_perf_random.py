"""gemm.perf.random -- what does the bit constraint cost on a bare GEMM, on random shapes?

PLACEHOLDER. Replaces the old `gemm.perf`, whose unconstrained arm was a Triton configuration
sweep we wrote ourselves. A ceiling we built out of our own kernel is not a credible ceiling: it
is only as good as the configuration space we happened to type in. The ceiling here is
`torch.compile(mode="max-autotune-no-cudagraphs")` instead -- the thing a user would actually
reach for.
"""
from __future__ import annotations

from ._common import print_design

NAME = "gemm.perf.random"
ORDER = 20
DESCRIPTION = "price of the bit constraint on random shapes, by shape family"
IMPLEMENTED = False

# The three arms are the same in `gemm.perf.random` and `gemm.perf.static`, so the columns that
# describe them are written once here and reused there. Only the columns that say WHICH shape a
# row is differ: a family here, a model and a layer there.
ARM_COLS = [
    ("dtype", "str", "operand dtype: fp16 or fp8 (e4m3)"),
    ("cublas_ms", "float", "arm 1: cuBLAS through the hot closure. Device time, median of "
     "CUDA-graph replays with L2 flushed between them"),
    ("torch_ms", "float", "arm 2 as it ships: torch.compile(mode='max-autotune-no-cudagraphs'), "
     "timed on whatever its autotuner actually picked"),
    ("torch_pick", "str", "what that autotuner picked: the extern cuBLAS call or a Triton template"),
    ("torch_triton_ms", "float", "arm 2 with the extern call excluded, so the number is the best "
     "Triton template torch.compile can produce rather than cuBLAS wearing a Triton hat"),
    ("torch_triton_cfg", "str", "the winning template configuration, BLOCK_M/BLOCK_N/BLOCK_K/"
     "num_warps/num_stages"),
    ("ours_ms", "float", "arm 3: our GEMM, tuned, and byte-identical to arm 1"),
    ("ours_cfg", "str", "the configuration ours was tuned to"),
    ("bit_ok", "int", "draws of arm 3 that came back byte-identical to arm 1"),
    ("bit_total", "int", "draws compared. bit_ok < bit_total makes the timings on this row "
     "meaningless, because the two arms are then not computing the same thing"),
    ("price_of_constraint", "float", "torch_triton_ms / ours_ms. Above 1 means the constraint "
     "cost nothing; below 1 is what it cost"),
    ("ours_over_cublas", "float", "cublas_ms / ours_ms, for reference. This is a different and "
     "much larger term -- it is Triton against cuBLAS, not the price of the constraint"),
    ("declined", "str", "non-empty if the shape is out of scope and no comparison was made"),
    ("error", "str", "non-empty if an arm failed to run"),
]

TABLES = {
    "gemm.perf.random": {
        "doc":
        "One row per random shape, drawn from the same four shape families as "
        "`gemm.bitmatch` (square, thin, deep K, decode) so the price of the constraint "
        "can be read per family rather than as one average that hides the split. Three "
        "arms per row, same inputs and same timing method.",
        "cols": [
            ("family", "str", "shape family the draw came from: square, thin, deepk or decode"),
            ("M", "int", "rows of A"),
            ("N", "int", "columns of B"),
            ("K", "int", "contraction length"),
        ] + ARM_COLS,
    },
}


def run(args, env):
    print_design(
        NAME,
        claim="""
        Requiring the output to be byte-identical to cuBLAS costs very little speed, and that
        stays true across the whole shape space rather than on a handful of shapes chosen by us.
        This is a smaller quantity than the gap between Triton and cuBLAS, and the two are easy
        to conflate.
        """,
        method="""
        Draw random shapes with `draw_shape_with_family` from `steps/_common.py` -- the same draw
        `gemm.bitmatch` uses, so the two steps cover the same space and a row here can be lined
        up with a row there. Keep the family name on the row.

        Three arms per shape, same operands, same timing (`graph_ms`: CUDA-graph capture, median
        replay, L2 flushed between replays). Arm 1 is cuBLAS through `hot_cublas`; it is both the
        baseline and the reference the bits are compared against. Arm 2 is torch with no numerics
        requirement, `torch.compile(mode="max-autotune-no-cudagraphs")`, reported twice: once as
        its autotuner actually picks (which is often the extern cuBLAS call, and saying so is
        part of the result) and once with the extern excluded, so the second number is the best
        Triton template torch.compile can build. Arm 3 is ours, tuned, and byte-identical to
        arm 1 -- checked with `digest` over several draws before any time is recorded.

        Budget with `--minutes`, like `gemm.bitmatch`. Stream one record per shape so a killed
        run keeps what it had.
        """,
        cost="Set by `--minutes`. Much slower per shape than `gemm.bitmatch`, because each row "
        "pays for a torch.compile autotune of the shape.",
        see="Per shape: the family, the three times, the price of the constraint, and what "
        "torch.compile picked. Then a geometric mean per family and one overall.",
        judge="""
        The claim rests on `price_of_constraint` = torch_triton_ms / ours_ms. `ours_over_cublas`
        is printed for reference only; it also contains Triton against cuBLAS, which is a
        separate and much larger term and is not what this step is about.

        Check `bit_ok` equals `bit_total` first. If it does not, arm 3 is not byte-identical on
        that shape and its time means nothing -- the row is a bug report, not a measurement.

        Read the families separately. A single geometric mean over a mixed draw hides the case
        where the constraint is free on square shapes and expensive on deep K, which is exactly
        the thing worth knowing.
        """,
        notes="""
        Why not the old arm 3. `gemm.perf` autotuned a Triton GEMM we wrote over a 108-entry
        space of our own choosing and called the best of it the ceiling. That is not a ceiling a
        reviewer has any reason to trust. torch.compile's max-autotune is the honest stand-in for
        what a user gets when they ask for speed and nothing else.
        """,
    )
