"""gemm.perf.static -- the same three arms as `gemm.perf.random`, on the fixed layer dimensions
of real models.

PLACEHOLDER. Random shapes say the constraint is cheap on average; a reviewer still wants to know
what it costs on the GEMMs that actually run in a model. The shapes are read from the two model
tables already in this tree, so a row can be traced back to a weight that exists.
"""
from __future__ import annotations

from ._common import print_design
from .gemm_perf_random import ARM_COLS

NAME = "gemm.perf.static"
ORDER = 30
DESCRIPTION = "price of the bit constraint on fixed shapes from real models"
IMPLEMENTED = False

TABLES = {
    "gemm.perf.static": {
        "doc":
        "One row per (model, layer). The same three arms as `gemm.perf.random`, on the "
        "layer dimensions of real models instead of random draws, so the price of the "
        "constraint can be read on the GEMMs that actually run.",
        "cols": [
            ("model", "str", "model the shape comes from"),
            ("layer", "str", "layer within that model, e.g. qkv_proj, gate_up_proj, lm_head"),
            ("M", "int", "rows of A; the token count the row was measured at"),
            ("N", "int", "columns of B"),
            ("K", "int", "contraction length"),
        ] + ARM_COLS,
    },
}


def run(args, env):
    print_design(
        NAME,
        claim="""
        On the GEMMs real models actually run, requiring the output to be byte-identical to
        cuBLAS costs very little speed.
        """,
        method="""
        Same three arms, same inputs, same timing as `gemm.perf.random` -- arm 1 cuBLAS through
        `hot_cublas` and the bit reference, arm 2 `torch.compile(mode="max-autotune-no-cudagraphs")`
        reported both as its autotuner picks and as its best Triton template with the extern
        excluded, arm 3 ours, tuned and byte-identical to arm 1. The only difference is where the
        shapes come from.

        Do not type a new shape list. Two tables in this tree already hold layer dimensions with
        the model and the layer named, so a row can be traced back to a weight that exists:
        `fusion_real/models.py` (Llama-2-7B, Llama-3-8B, Llama-3-70B, with the fused `qkv_proj`
        and `gate_up_proj` that vLLM and TensorRT-LLM ship) and `fusion_moe/models_2026.py` (the
        open-weight models current in August 2026, every number read out of that model's own
        config.json, including the MoE expert GEMMs). Import from those; they are also the
        shape source `gemm.fusion` uses, so the two steps stay on the same shapes.

        Sweep M over a small fixed set of token counts, because the answer moves with it: decode
        is M=1, and prefill and training are not.
        """,
        cost="Fixed, not budgeted by `--minutes`: it is the shape count times a torch.compile "
        "autotune per shape.",
        see="Per (model, layer): the three times, the price of the constraint, and what "
        "torch.compile picked. Then a geometric mean per model and one overall.",
        judge="""
        The claim rests on `price_of_constraint` = torch_triton_ms / ours_ms. Check `bit_ok`
        equals `bit_total` first; if not, arm 3 is not byte-identical on that shape and its time
        means nothing.

        Read it per layer kind. `lm_head` and an expert `down_proj` are different shapes with
        different answers, and one mean over all of them is not the claim.
        """,
        notes="""
        `gemm.perf.random` and `gemm.perf.static` answer different questions and both are needed.
        The random draw covers a space and can support a statement about the space; the fixed
        shapes cannot, but they are the ones a reader will care about. Neither substitutes for
        the other.
        """,
    )
