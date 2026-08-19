"""gemm.fusion -- for an epilogue cuBLAS cannot fuse, how much faster is our bit-exact GEMM with
the epilogue folded in than the two-kernel path a user gets today?

PLACEHOLDER. The design below is settled and measured data now exists; what is missing is the
wiring that turns that data into these tables through the driver.
"""
from __future__ import annotations

from ._common import print_design

NAME = "gemm.fusion"
ORDER = 40
DESCRIPTION = "what the bit constraint costs on a fused GEMM"
IMPLEMENTED = False

# Both tables are owned by this step, so a third one (for the three_way oracle below) is added
# here and nowhere else.
_FUSION_COLS = [
    ("model", "str", "model the shape comes from"),
    ("layer", "str", "layer within that model"),
    ("pairing", "str", "which fusion case: lora, moe_up, moe_down, lm_head, attn"),
    ("epi", "str", "the epilogue actually fused, as it appears in that model"),
    ("M", "int", "rows"),
    ("N", "int", "columns"),
    ("K", "int", "contraction length"),
    ("dtype", "str", "operand dtype"),
    ("mode", "str", "cuBLAS plan mode the shape resolves to"),
    ("baseline_ms", "float", "eager unfused: hot cuBLAS then one separate epilogue kernel"),
    ("fused_ms", "float", "ours, fused and byte-identical to the baseline"),
    ("free_fused_ms", "float", "the same fusion on an ordinary autotuned Triton matmul"),
    ("speedup", "float", "baseline_ms / fused_ms; above 1 means fusion pays"),
    ("bit_ok", "int", "draws byte-identical to the baseline"),
    ("bit_total", "int", "draws compared"),
    ("verdict", "str", "win, loss, or why the case was not covered"),
    ("tma", "str", "whether the fused kernel used TMA"),
    ("fused_cfg", "str", "launch configuration of the fused kernel"),
]

TABLES = {
    "gemm.fusion.dense": {
        "doc":
        "One row per (model, layer, epilogue). Fusing a byte-identical epilogue into the GEMM, on dense Llama-family layers. `speedup` above 1 means fusion beats the two-kernel path a user gets today.",
        "cols": list(_FUSION_COLS),
    },
    "gemm.fusion.moe": {
        "doc":
        "The same measurement on the layer dimensions of open-weight models current in August 2026, read from each model's own config.json. Includes the MoE expert GEMMs, whose epilogue is SwiGLU on gate/up and the routing-weight multiply on down.",
        "cols": list(_FUSION_COLS),
    },
}


def run(args, env):
    print_design(
        NAME,
        claim="""
        For an epilogue cuBLAS cannot express, folding the epilogue into a GEMM that is still
        byte-identical to cuBLAS beats the two-kernel path a user gets today -- and the bit
        constraint is not what decides whether it wins.
        """,
        method="""
        Three arms, same shapes, same inputs, device time by CUDA-graph replay.

        Arm 1, the baseline a user actually gets: cuBLAS through the hot closure, then the
        epilogue as its own Triton kernel. Note it is TWO kernels even for a four-op chain,
        because Inductor fuses a pointwise chain into one kernel even when it cannot fuse into
        the GEMM.

        Arm 2, ours: the bit-exact GEMM with the epilogue applied to the accumulator before the
        store. It must be BYTE-IDENTICAL to arm 1, which means rounding to the output dtype at
        the op boundary exactly where the unfused version rounds -- what Inductor calls
        "emulate unfused numerics". If it is not byte-identical the timing is meaningless.

        Arm 3, the ceiling: the same fusion on an unconstrained autotuned Triton matmul, which is
        what torch.compile with max-autotune would pick. arm2/arm3 is the price of the
        constraint.

        The epilogues have to be ones cuBLASLt cannot express -- its whole list is bias, relu,
        gelu and their backward and bias-gradient variants. So: silu, a two-tensor gate
        `x * sigmoid(g)`, `(x + residual) * scale`, and a four-op chain.

        Report per shape as well as a geometric mean, because the win grows as K shrinks and that
        trend matters more than one number.
        """,
        cost="Fixed by the case list, not by `--minutes`.",
        see="Per (model, layer, epilogue): the three times, the speedup, and the bit check. Then "
        "a geometric mean per pairing and one overall.",
        judge="""
        Check `bit_ok` equals `bit_total` before reading any time on that row. `speedup` =
        baseline_ms / fused_ms is the headline; `fused_ms / free_fused_ms` is the price of the
        bit constraint, and the two must not be mixed up.
        """,
        notes="""
        MEASURED DATA EXISTS AND IS NOT WIRED IN. `artifact_eval/fusion_oracle/three_way.jsonl`
        holds the three-way oracle run: per (M, N, K, epilogue, dtype) it records the eager
        unfused arm, what Inductor produces, Inductor's own Triton template, and ours both
        untuned and tuned, each with its bit check against the reference and its device time,
        plus the ready-made `r_*_over_ours` ratios. `artifact_eval/data/summary.csv` already
        carries the `gemm.fusion.oracle` rows summarising it. None of that reaches these tables
        yet: nothing in this step reads that file. Wiring it up is the work, and it is a change
        to THIS file -- a new table is declared in `TABLES` above and nowhere else.

        The older per-model runs in `fusion_real/` and `fusion_moe/` are what filled
        `gemm.fusion.dense` and `gemm.fusion.moe`, and they too are run by hand rather than
        through the driver.

        The number to re-earn. On the fbsource checkout fusing was a LOSS, 0.802x over 136 pairs,
        and the split was arm1/arm2-unconstrained = 0.810 (our GEMM), arm2u/arm2 = 0.990 (the
        fusion itself), arm2/arm3 = 0.996 (the bit constraint). So the loss was our GEMM, not the
        constraint -- and that is the distinction this step exists to make.
        """,
    )
