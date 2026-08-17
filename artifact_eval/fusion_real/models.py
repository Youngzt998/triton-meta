"""Layer dimensions of real models, and the case list built from them.

Every case names a model and a layer, so a row in the result table can be traced back to a
weight that exists.  The dimensions are the published Llama configurations:

    Llama-3-8B    hidden 4096  ffn 14336  vocab 128256  32 heads x 128, 8 kv heads
    Llama-2-7B    hidden 4096  ffn 11008  vocab  32000  32 heads x 128, 32 kv heads (no GQA)
    Llama-3-70B   hidden 8192  ffn 28672  vocab 128256  64 heads x 128, 8 kv heads

A fused `qkv_proj` has out_features = hidden + 2 * kv_heads * head_dim; a fused `gate_up_proj`
has out_features = 2 * ffn.  Both fusions are what vLLM and TensorRT-LLM ship, so those are the
GEMMs that actually run.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    hidden: int
    ffn: int
    vocab: int
    heads: int
    kv_heads: int
    head_dim: int = 128

    @property
    def qkv_out(self) -> int:
        return self.hidden + 2 * self.kv_heads * self.head_dim

    @property
    def gate_up_out(self) -> int:
        return 2 * self.ffn


LLAMA3_8B = Model("Llama-3-8B", hidden=4096, ffn=14336, vocab=128256, heads=32, kv_heads=8)
LLAMA2_7B = Model("Llama-2-7B", hidden=4096, ffn=11008, vocab=32000, heads=32, kv_heads=32)
LLAMA3_70B = Model("Llama-3-70B", hidden=8192, ffn=28672, vocab=128256, heads=64, kv_heads=8)
MODELS = [LLAMA3_8B, LLAMA2_7B, LLAMA3_70B]

# Token counts. The first three are a training step or a prefill; the rest are decode with the
# batch size as the token count.
TOK_BIG = (16384, 8192, 4096)
TOK_DECODE = (256, 64, 8, 1)

RANKS = (8, 16, 32, 64)


@dataclass(frozen=True)
class Case:
    """One (model, layer, epilogue) triple at one token count.

    `M x N x K` is the GEMM the epilogue is fused into: M tokens, N out_features, K in_features.
    """
    model: str
    layer: str  # the weight this GEMM multiplies by, named as in the checkpoint
    pairing: str  # which of the five workloads this row belongs to
    M: int
    N: int
    K: int
    epi: str  # epilogue name, see fused_real.EPI
    note: str = ""

    @property
    def key(self):
        return (self.model, self.layer, self.pairing, self.M, self.N, self.K, self.epi)

    def label(self):
        return f"{self.model} {self.layer} {self.M}x{self.N}x{self.K} {self.epi}"


# ------------------------------------------------------------------------------------------- #
# 1. LoRA merge.  y = base + (x @ A^T) @ B^T * (alpha / r)
#
# The GEMM fused into is the SECOND one, `lora_B`: (tokens, out_features, rank), so K = rank.
# The epilogue is the residual add of the frozen layer's output plus the adapter scalar --
# exactly what `peft`'s `Linear.forward` computes, and exactly what a merge-free multi-adapter
# server has to do on every token.
# ------------------------------------------------------------------------------------------- #


def lora_cases(toks=TOK_BIG + TOK_DECODE, ranks=RANKS):
    out = []
    for m in MODELS:
        for layer, n in (("q_proj.lora_B", m.hidden), ("up_proj.lora_B", m.ffn)):
            for r in ranks:
                for t in toks:
                    out.append(Case(m.name, f"{layer} r={r}", "lora", t, n, r, "lora", note="base + delta*(alpha/r)"))
    return out


# ------------------------------------------------------------------------------------------- #
# 2. Quantization boundary.  A linear layer whose output is scaled and cast to fp8 for the next
# layer -- static (delayed) scaling, as in TransformerEngine fp8 training and in fp8 inference.
# K is hidden or ffn and is therefore large; the epilogue halves the bytes written.
# ------------------------------------------------------------------------------------------- #


def fp8q_cases(toks=(16384, 8192, 4096, 256, 64, 8, 1)):
    out = []
    for m in MODELS:
        layers = (("qkv_proj", m.qkv_out, m.hidden), ("o_proj", m.hidden, m.hidden),
                  ("gate_up_proj", m.gate_up_out, m.hidden), ("down_proj", m.hidden, m.ffn))
        for layer, n, k in layers:
            for t in toks:
                out.append(Case(m.name, layer, "fp8q", t, n, k, "fp8q", note="scale then cast to fp8"))
    return out


# ------------------------------------------------------------------------------------------- #
# 3. lm_head then argmax -- greedy decode.  (batch, vocab, hidden).  K = hidden is large, so the
# GEMM handicap is against us, but the epilogue collapses `vocab` values to one, which is the
# largest output-shrinking effect there is.
# ------------------------------------------------------------------------------------------- #


def lmhead_cases(toks=(256, 64, 8, 1)):
    return [
        Case(m.name, "lm_head", "lmhead", t, m.vocab, m.hidden, "argmax", note="greedy decode")
        for m in MODELS
        for t in toks
    ]


# ------------------------------------------------------------------------------------------- #
# 4 and 5.  The main trunk at its real K, where fusion is expected to lose.  Reported so the
# boundary is stated honestly rather than left out.
#
#   SwiGLU          silu(gate_proj(x)) * up_proj(x).  Fused into up_proj; gate_proj's output is
#                   the second M x N tensor.  Both GEMMs have K = hidden.
#   residual add    o_proj(attn) + residual, fused into o_proj.  K = hidden.
# ------------------------------------------------------------------------------------------- #


def trunk_cases(toks=(16384, 8192, 4096, 256, 64, 8, 1)):
    out = []
    for m in MODELS:
        for t in toks:
            out.append(Case(m.name, "up_proj", "swiglu", t, m.ffn, m.hidden, "swiglu",
                            note="silu(gate_proj) * up_proj"))
            out.append(
                Case(m.name, "o_proj", "resid", t, m.hidden, m.hidden, "resid", note="attn out + residual stream"))
    return out


GROUPS = {
    "lora": lora_cases,
    "fp8q": fp8q_cases,
    "lmhead": lmhead_cases,
    "trunk": trunk_cases,
}


def build(names):
    out = []
    for n in names:
        out.extend(GROUPS[n]())
    return out
