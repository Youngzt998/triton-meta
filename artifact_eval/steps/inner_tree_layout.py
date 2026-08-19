"""inner_tree.layout -- the reduction layout-optimization pass: is it bit-safe, and what does it
buy?

PLACEHOLDER.

Attribution. The ordering path this pass sits on top of is not ours: `reduction_ordering` /
`inner_tree` (D100027220) and `TRITON_STRICT_REDUCTION_ORDERING` (D101872700) were written by
Nick Riasanovsky, a co-author. The pass under test here is
`tritongpu-optimize-reduction-layout` (D110978754).
"""
from __future__ import annotations

from ._common import print_design

NAME = "inner_tree.layout"
ORDER = 70
DESCRIPTION = "is the reduction layout pass bit-safe, and what does it buy"
IMPLEMENTED = False

TABLES = {
    "inner_tree.layout": {
        "doc":
        "One row per (kernel, dtype, configuration group). The same kernel compiled "
        "without and with `tritongpu-optimize-reduction-layout`, answering three "
        "independent questions in order: does it still compile, do the bits change, and "
        "is it faster. A speedup on a row whose bits changed is not a result.",
        "cols": [
            ("kernel", "str", "kernel from the bitequiv evaluation registry"),
            ("dtype", "str", "element dtype: f16, bf16, f32 or fp8"),
            ("ordering", "str", "reduction_ordering the row was compiled with; the pass targets inner_tree"),
            ("config", "str", "the configuration, e.g. num_warps=4 num_stages=3 block_n=1024"),
            ("pass_name", "str", "the pass under test, in triton-opt vocabulary"),
            ("pass_available", "int", "1 if the build has a binding for that pass; 0 means the row "
             "is a baseline-against-baseline control and must show no change"),
            ("compiles", "int", "1 if the kernel still compiles with the pass applied"),
            ("seeds", "int", "random input draws the two builds were compared on"),
            ("bit_changed", "int", "draws whose bytes differ from the baseline build. Must be 0"),
            ("baseline_ms", "float", "device time without the pass, median of CUDA-graph replays"),
            ("optimized_ms", "float", "device time with the pass"),
            ("speedup", "float", "baseline_ms / optimized_ms; above 1 means the pass helped"),
            ("verdict", "str", "bit-safe and faster, bit-safe and neutral, BITS CHANGED, or why "
             "the row was out of scope"),
            ("error", "str", "non-empty if the row failed to produce a result"),
        ],
    },
}


def run(args, env):
    print_design(
        NAME,
        claim="""
        `tritongpu-optimize-reduction-layout` never changes the output bits, and on the
        reductions it targets it makes them faster. Both halves are needed: a pass that is fast
        but moves the bits is useless here, because the whole point of the ordering switch is
        that the bits stop moving.
        """,
        method="""
        Do not build a new harness. `bitequiv/evaluation/evaluate_opt.py` is exactly this ruler
        and it is deliberately pass-agnostic -- you name the pass on the command line in
        triton-opt vocabulary and it compiles every in-scope configuration without and with it,
        in three independent stages:

            support      does it still compile? compile only, no launches
            correctness  do the bytes change against the baseline build? the gate is 0 changes
            performance  is it faster? baseline against optimized

        It injects the pass by patching the NVIDIA backend's `make_ttgir` to append it at
        end-of-TTGIR, so the kernel file and the launch path are shared with the baseline and
        only the compile pipeline differs. This step calls it and writes its three answers into
        one table per row.

        Two controls are worth the time. First, run it with no pass at all: optimized must equal
        baseline, 0 bit changes. If that self-check does not come out clean the harness is
        broken and nothing after it means anything. Second, keep the rows where the build has no
        binding for the pass (`pass_available` = 0); they are treated as baseline and must show
        no change, which is how you tell "the pass did nothing" apart from "the pass was never
        applied".

        GOTCHA. `TRITON_ALWAYS_COMPILE=1` has to be set when the process starts. Triton's compile
        cache is not keyed on the injected pass, so without it the optimized build is served from
        cache as the baseline and the run reports a perfect, meaningless 0 bit changes.
        """,
        cost="The correctness and performance stages launch kernels; budget on the order of tens "
        "of minutes for the full registry. Support alone is compile-only and quick.",
        see="Per row: whether it compiled, how many draws changed bits, the two times and the "
        "speedup. Then the bit-change total across all rows, which is the gate.",
        judge="""
        Read `bit_changed` first and only then the speedup. `bit_changed` must be 0 on every row;
        one non-zero row means the pass is not bit-safe and its speedup is not a result.

        Then check `pass_available` is 1. A row with 0 measured nothing, however clean it looks.

        The speedup is per configuration and it should be read that way. A geometric mean over
        rows where the pass does not apply drags toward 1 and hides both the wins and the
        regressions.
        """,
        notes="""
        The pass is in scope only for `reduction_ordering=inner_tree` configurations, which is
        what `evaluate_opt.py` defaults to; a kernel with no such configuration reports out of
        scope rather than passing silently.
        """,
    )
