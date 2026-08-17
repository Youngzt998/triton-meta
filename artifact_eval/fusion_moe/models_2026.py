"""Layer dimensions of the open-weight models that matter in August 2026, and the case list
built from them.

Every number below was read out of that model's own `config.json` on Hugging Face on
2026-08-16, by fetching

    https://huggingface.co/<repo>/raw/main/config.json

and each field records the config key it came from.  Nothing here is from memory, and a model
whose number could not be read that way is not in the table.  The picks are the models with the
highest trending score and the highest 30-day download count in the Hugging Face
`text-generation` listing on the same day, plus the two Qwen MoEs that are still heavily
downloaded from the previous generation.

`models.py` in `fusion_real/` (the previous run) held Llama-3-8B, Llama-2-7B and Llama-3-70B.
Those are gone from the list here: none of them is in the current top-40 by trending score, and
none is an MoE.

Which epilogue really follows which GEMM
----------------------------------------
The rule this file exists to satisfy (`DEV_NOTES.md`) is that the epilogue must be the operation
that actually follows that GEMM in that model.  For the MoE expert FFN the epilogues were read
out of official modeling code, not guessed:

  down_proj -> multiply by the routing weight, one scalar per token, before the expert outputs
  are summed.  Verified in three independent official implementations:

    modeling_minimax_m2.py     `self[expert_idx](current_state) * top_k_weights[top_x, idx, None]`
    modeling_kimi_linear.py    `new_x.view(...).mul_(topk_weight.unsqueeze(-1)).sum(dim=1)`
    modeling_bailing_moe_v3.py `.mul_(topk_weight.unsqueeze(dim=-1))` after `expert_out`

  DeepSeek-V4 is the documented exception.  Its official `inference/model.py` applies the weight
  one GEMM earlier, on the SwiGLU output:

    Expert.forward:  x = F.silu(gate) * up ; if weights is not None: x = weights * x ; return w2(x)

  so for DeepSeek-V4 the routing-weight multiply belongs to the gate/up epilogue and the down
  projection has NO pointwise epilogue at all -- its output goes straight into an index_add.
  That is why DeepSeek-V4 has no `moe_down` row.

  gate/up -> the activation.  `hidden_act` in each config:
    silu, no clamp     Qwen3.x, GLM, Kimi-K2.6, MiniMax-M2, Ling-3.0     -> `swiglu`
    silu + swiglu_limit DeepSeek-V4 (10.0 Flash / 10.0 Pro)              -> `swiglu_cw`
    relu2 (`mlp_hidden_act`) Nemotron-3.5-Lightning                      -> `relu2`
    situ (custom)      Kimi-K3                                           -> no gate/up row
    silu + swiglu_limit 7.0 but the alpha constant is not in the config
                       gpt-oss                                           -> no gate/up row

  A model with no verifiable gate/up epilogue still gets its `moe_down` row, because the routing
  weight multiply is the same operation everywhere it appears.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    """One model.  `None` means the key is absent from that model's config.json, which for
    `intermediate_size` means the model has no dense FFN layer at all."""
    name: str
    repo: str  # the Hugging Face repo the config.json came from
    hidden_size: int
    intermediate_size: int | None  # the DENSE ffn; None when every layer is MoE
    moe_intermediate_size: int | None
    num_experts: int | None
    experts_per_token: int | None
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    head_dim: int | None  # None for the MLA models, where q/k/v head widths differ
    expert_in: int | None = None  # the width the routed experts read; hidden_size unless stated
    up_epi: str | None = None  # the epilogue on the gate/up GEMM, None if not verifiable
    down_epi: str | None = None  # the epilogue on the down GEMM
    swiglu_limit: float = 0.0
    gqa: bool = True  # a fused qkv is only well defined for plain grouped-query attention
    # False when the output projection is NOT a single matrix.  DeepSeek-V4's is a grouped
    # low-rank pair -- inference/model.py builds
    #     wo_a: (n_heads * head_dim // o_groups) -> n_groups * o_lora_rank
    #     wo_b: (n_groups * o_lora_rank)         -> dim
    # so `num_attention_heads * head_dim -> hidden_size` is a shape that does not exist in the
    # model.  A row for it would be an invented dimension, so there is none.
    one_o_proj: bool = True

    @property
    def ein(self) -> int:
        return self.expert_in if self.expert_in is not None else self.hidden_size

    @property
    def q_out(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def qkv_out(self) -> int:
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim

    def tokens_per_expert(self, total_tokens: int, balance: float) -> int:
        """`balance` = 1.0 is perfectly even routing, which never happens; the sweep also runs a
        hot expert and a cold one.  The factors are chosen, not measured -- they are named in the
        layer string so no row can be read as a measured routing statistic."""
        return max(1, round(total_tokens * self.experts_per_token / self.num_experts * balance))


# ------------------------------------------------------------------------------------------- #
# The table.  Field -> config.json key:
#   hidden_size            hidden_size
#   intermediate_size      intermediate_size            (absent => no dense FFN layer)
#   moe_intermediate_size  moe_intermediate_size        (gpt-oss: intermediate_size, see below)
#   num_experts            num_experts | n_routed_experts | num_local_experts
#   experts_per_token      num_experts_per_tok | num_experts_per_token | experts_per_token
#   num_attention_heads    num_attention_heads
#   num_key_value_heads    num_key_value_heads
#   vocab_size             vocab_size
#   head_dim               head_dim
# ------------------------------------------------------------------------------------------- #

MODELS = [
    # Qwen/Qwen3.8-2.4T-A95B -- the highest trending score in the HF text-generation listing on
    # 2026-08-16.  No `intermediate_size` key: every layer is MoE, with a shared expert of
    # shared_expert_intermediate_size=2048 beside the 512 routed ones.
    Model("Qwen3.8-2.4T-A95B", "Qwen/Qwen3.8-2.4T-A95B", hidden_size=8192, intermediate_size=None,
          moe_intermediate_size=2048, num_experts=512, experts_per_token=10, num_attention_heads=64,
          num_key_value_heads=4, vocab_size=248320, head_dim=256, up_epi="swiglu", down_epi="rweight"),

    # deepseek-ai/DeepSeek-V4-Pro -- 1.28M downloads in 30 days.  No dense FFN key.  Attention is
    # not plain GQA (head_dim 512, num_key_value_heads 1, q_lora_rank 1536, o_lora_rank 1024), so
    # no fused-qkv row.  swiglu_limit 10.0 and the routing weight is applied on the SwiGLU output,
    # both from the repo's own inference/model.py.
    Model("DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro", hidden_size=7168, intermediate_size=None,
          moe_intermediate_size=3072, num_experts=384, experts_per_token=6, num_attention_heads=128,
          num_key_value_heads=1, vocab_size=129280, head_dim=512, up_epi="swiglu_cw", down_epi=None, swiglu_limit=10.0,
          gqa=False, one_o_proj=False),

    # deepseek-ai/DeepSeek-V4-Flash -- 2.05M downloads in 30 days, the most-downloaded frontier
    # open-weight model in the listing.
    Model("DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash", hidden_size=4096, intermediate_size=None,
          moe_intermediate_size=2048, num_experts=256, experts_per_token=6, num_attention_heads=64,
          num_key_value_heads=1, vocab_size=129280, head_dim=512, up_epi="swiglu_cw", down_epi=None, swiglu_limit=10.0,
          gqa=False, one_o_proj=False),

    # zai-org/GLM-5.2 -- 2.71M downloads in 30 days.  MLA (qk_nope 192 / qk_rope 64 / v 256), so
    # head_dim is not one number and there is no fused-qkv row.
    Model("GLM-5.2", "zai-org/GLM-5.2", hidden_size=6144, intermediate_size=12288, moe_intermediate_size=2048,
          num_experts=256, experts_per_token=8, num_attention_heads=64, num_key_value_heads=64, vocab_size=154880,
          head_dim=None, up_epi="swiglu", down_epi="rweight", gqa=False),

    # zai-org/GLM-4.7-Flash -- 2.02M downloads in 30 days.  MLA as above.
    Model("GLM-4.7-Flash", "zai-org/GLM-4.7-Flash", hidden_size=2048, intermediate_size=10240,
          moe_intermediate_size=1536, num_experts=64, experts_per_token=4, num_attention_heads=20,
          num_key_value_heads=20, vocab_size=154880, head_dim=None, up_epi="swiglu", down_epi="rweight", gqa=False),

    # moonshotai/Kimi-K2.6.  Fields come from config.json's `text_config`.  MLA.
    Model("Kimi-K2.6", "moonshotai/Kimi-K2.6", hidden_size=7168, intermediate_size=18432, moe_intermediate_size=2048,
          num_experts=384, experts_per_token=8, num_attention_heads=64, num_key_value_heads=64, vocab_size=163840,
          head_dim=None, up_epi="swiglu", down_epi="rweight", gqa=False),

    # moonshotai/Kimi-K3.  `text_config`.  This one is a LATENT MoE: modeling_kimi_linear.py sets
    # the expert width to routed_expert_hidden_size=3584, not hidden_size=7168, so the expert
    # GEMMs have K=3584 on the way up and N=3584 on the way down.  hidden_act is "situ", a custom
    # activation, so there is no gate/up row.  num_experts_per_token (not ..._tok) = 16.
    Model("Kimi-K3", "moonshotai/Kimi-K3", hidden_size=7168, intermediate_size=33792, moe_intermediate_size=3072,
          num_experts=896, experts_per_token=16, num_attention_heads=96, num_key_value_heads=96, vocab_size=163840,
          head_dim=None, expert_in=3584, up_epi=None, down_epi="rweight", gqa=False),

    # openai/gpt-oss-120b -- 4.48M downloads in 30 days.  No `moe_intermediate_size`: every layer
    # is MoE and `intermediate_size`=2880 IS the expert width.  num_local_experts=128,
    # experts_per_token=4.  swiglu_limit 7.0 is in the config but the alpha constant of its
    # clamped SwiGLU is not, so there is no gate/up row.
    Model("gpt-oss-120b", "openai/gpt-oss-120b", hidden_size=2880, intermediate_size=None, moe_intermediate_size=2880,
          num_experts=128, experts_per_token=4, num_attention_heads=64, num_key_value_heads=8, vocab_size=201088,
          head_dim=64, up_epi=None, down_epi="rweight", swiglu_limit=7.0),

    # openai/gpt-oss-20b -- 7.97M downloads in 30 days.  Same widths, 32 experts.
    Model("gpt-oss-20b", "openai/gpt-oss-20b", hidden_size=2880, intermediate_size=None, moe_intermediate_size=2880,
          num_experts=32, experts_per_token=4, num_attention_heads=64, num_key_value_heads=8, vocab_size=201088,
          head_dim=64, up_epi=None, down_epi="rweight", swiglu_limit=7.0),

    # nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 -- 4th by trending score.  A hybrid
    # mamba/attention model; `mlp_hidden_act` is relu2, NOT swiglu, so the gate/up GEMM is a
    # single up projection whose epilogue reads no second tensor.
    Model("Nemotron-3.5-L-30B-A3B", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", hidden_size=2688,
          intermediate_size=1856, moe_intermediate_size=1856, num_experts=128, experts_per_token=6,
          num_attention_heads=32, num_key_value_heads=2, vocab_size=131072, head_dim=128, up_epi="relu2",
          down_epi="rweight"),

    # MiniMaxAI/MiniMax-M2.  No `moe_intermediate_size`: modeling_minimax_m2.py builds every
    # expert as MiniMaxM2MLP with ffn_dim = intermediate_size = 1536.  num_local_experts=256.
    # `shared_intermediate_size` is 0, so there is no shared expert.
    Model("MiniMax-M2", "MiniMaxAI/MiniMax-M2", hidden_size=3072, intermediate_size=None, moe_intermediate_size=1536,
          num_experts=256, experts_per_token=8, num_attention_heads=48, num_key_value_heads=8, vocab_size=200064,
          head_dim=128, up_epi="swiglu", down_epi="rweight"),

    # Qwen/Qwen3-30B-A3B -- 2.63M downloads in 30 days, the previous generation but still the
    # most-downloaded MoE that fits on one card.
    Model("Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B", hidden_size=2048, intermediate_size=6144, moe_intermediate_size=768,
          num_experts=128, experts_per_token=8, num_attention_heads=32, num_key_value_heads=4, vocab_size=151936,
          head_dim=128, up_epi="swiglu", down_epi="rweight"),

    # inclusionAI/Ling-3.0-flash.  512 experts of 768 -- the narrowest expert in the table
    # together with Qwen3-30B-A3B.  expert_swiglu_limit_list is 0 for 35 of its 42 layers, so the
    # unclamped SwiGLU is the majority case.
    Model("Ling-3.0-flash", "inclusionAI/Ling-3.0-flash", hidden_size=2560, intermediate_size=6144,
          moe_intermediate_size=768, num_experts=512, experts_per_token=8, num_attention_heads=32,
          num_key_value_heads=32, vocab_size=157184, head_dim=128, up_epi="swiglu", down_epi="rweight", gqa=False),

    # Qwen/Qwen3.6-35B-A3B.  `text_config`.  512-wide experts, the narrowest here.
    Model("Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B", hidden_size=2048, intermediate_size=None,
          moe_intermediate_size=512, num_experts=256, experts_per_token=8, num_attention_heads=16,
          num_key_value_heads=2, vocab_size=248320, head_dim=256, up_epi="swiglu", down_epi="rweight"),

    # ---- dense, for the LoRA anchor and the lm_head row -------------------------------------- #

    # Qwen/Qwen3.8-27B -- `text_config`.  The dense sibling of the flagship.
    Model("Qwen3.8-27B", "Qwen/Qwen3.8-27B", hidden_size=5120, intermediate_size=17408, moe_intermediate_size=None,
          num_experts=None, experts_per_token=None, num_attention_heads=24, num_key_value_heads=4, vocab_size=248320,
          head_dim=256),

    # ibm-granite/granite-4.1-8b -- 3.38M downloads in 30 days, dense.
    Model("granite-4.1-8b", "ibm-granite/granite-4.1-8b", hidden_size=4096, intermediate_size=12800,
          moe_intermediate_size=None, num_experts=None, experts_per_token=None, num_attention_heads=32,
          num_key_value_heads=8, vocab_size=100352, head_dim=128),
]

BY_NAME = {m.name: m for m in MODELS}
MOE = [m for m in MODELS if m.num_experts]

# ------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Case:
    """One (model, layer, epilogue) triple at one shape.  `M x N x K` is the GEMM the epilogue is
    fused into: M tokens, N out_features, K in_features."""
    model: str
    layer: str
    pairing: str
    M: int
    N: int
    K: int
    epi: str
    note: str = ""
    lim: float = 0.0  # the clamp in DeepSeek-V4's swiglu, 0 = no clamp

    @property
    def key(self):
        return (self.model, self.layer, self.pairing, self.M, self.N, self.K, self.epi)

    def label(self):
        return f"{self.model} {self.layer} {self.M}x{self.N}x{self.K} {self.epi}"


# Routing is never even.  1.0 is the mean; the other two are a hot expert and a cold one.  The
# factors are chosen, not measured, and are written into the layer name for that reason.
BALANCE = ((1.0, "even"), (3.0, "hot3x"), (0.25, "cold.25x"))
TOTAL_TOKENS = (16384, 4096)
RANKS = (8, 16, 32, 64)
LORA_TOKENS = (16384, 4096)
DECODE_TOKENS = (256, 64, 8, 1)
ATTN_TOKENS = (16384, 4096, 256)


def moe_cases():
    """1. The expert FFN.  Per-expert GEMMs at a realistic token count per expert."""
    out = []
    for m in MOE:
        for T in TOTAL_TOKENS:
            for bal, bname in BALANCE:
                t = m.tokens_per_expert(T, bal)
                tag = f"T{T // 1024}k/{bname}"
                if m.up_epi:
                    epi = m.up_epi
                    note = {
                        "swiglu": "silu(gate_proj) * up_proj",
                        "swiglu_cw": "routing weight * (clamped silu(gate) * clamped up)",
                        "relu2": "relu(up_proj)^2",
                    }[epi]
                    out.append(
                        Case(m.name, f"expert.up_proj {tag}", "moe_up", t, m.moe_intermediate_size, m.ein, epi,
                             note=note, lim=m.swiglu_limit))
                if m.down_epi:
                    out.append(
                        Case(m.name, f"expert.down_proj {tag}", "moe_down", t, m.ein, m.moe_intermediate_size,
                             m.down_epi, note="routing weight, one scalar per token"))
    return out


LORA_MODELS = ("Qwen3.8-2.4T-A95B", "DeepSeek-V4-Flash", "GLM-5.2", "Kimi-K2.6", "gpt-oss-120b", "MiniMax-M2",
               "Qwen3-30B-A3B", "Qwen3.8-27B")


def lora_cases():
    """2. LoRA merge, y = base + (x @ A^T) @ B^T * (alpha/r), fused into lora_B so K = rank.
    The epilogue is peft's `Linear.forward`: the frozen layer's output plus the scaled delta."""
    out = []
    for name in LORA_MODELS:
        m = BY_NAME[name]
        layers = [("o_proj.lora_B", m.hidden_size)]
        if m.head_dim:
            layers.append(("q_proj.lora_B", m.q_out))
        for layer, n in layers:
            for r in RANKS:
                for t in LORA_TOKENS:
                    out.append(Case(m.name, f"{layer} r={r}", "lora", t, n, r, "lora", note="base + delta*(alpha/r)"))
        # LoRA on a routed expert: same pairing, but M is the tokens that reached that expert.
        if m.moe_intermediate_size and m.num_experts:
            for r in RANKS:
                for T in TOTAL_TOKENS:
                    t = m.tokens_per_expert(T, 1.0)
                    out.append(
                        Case(m.name, f"expert.up_proj.lora_B r={r} T{T // 1024}k", "lora", t, m.moe_intermediate_size,
                             r, "lora", note="base + delta*(alpha/r), one expert"))
    return out


def lmhead_cases():
    """3. lm_head then argmax -- greedy decode.  K = hidden is large but the epilogue collapses
    `vocab` values to one, the largest output-shrinking effect there is."""
    seen, out = set(), []
    for m in MODELS:
        k = (m.hidden_size, m.vocab_size)
        if k in seen:
            continue
        seen.add(k)
        for t in DECODE_TOKENS:
            out.append(Case(m.name, "lm_head", "lmhead", t, m.vocab_size, m.hidden_size, "argmax",
                            note="greedy decode"))
    return out


def attn_cases():
    """4. The attention projections, expected to lose.  `o_proj` carries the post-attention
    residual add, which is what really follows it.  A fused `qkv_proj` is only well defined for
    plain grouped-query attention, and what really follows it in an fp8-served model is the
    scale-and-cast to fp8, so that is the epilogue it carries.

    The MLA models are absent from this arm.  Their q/k/v widths differ from each other, so their
    config has no single `head_dim`, and DeepSeek-V4 does not even have a single `o_proj`.  A row
    for a shape a model does not contain is worth less than no row."""
    out = []
    for m in MODELS:
        if not m.head_dim or not m.one_o_proj:
            continue
        for t in ATTN_TOKENS:
            out.append(
                Case(m.name, "o_proj", "attn", t, m.hidden_size, m.q_out, "resid", note="attn out + residual stream"))
            if m.gqa:
                out.append(
                    Case(m.name, "qkv_proj", "attn", t, m.qkv_out, m.hidden_size, "fp8q",
                         note="scale then cast to fp8"))
    return out


GROUPS = {
    "moe": moe_cases,
    "lora": lora_cases,
    "lmhead": lmhead_cases,
    "attn": attn_cases,
}


def build(names):
    out = []
    for n in names:
        out.extend(GROUPS[n]())
    return out


if __name__ == "__main__":
    for g, fn in GROUPS.items():
        cs = fn()
        print(f"{g:8s} {len(cs):4d} cases")
    print(f"{'total':8s} {sum(len(fn()) for fn in GROUPS.values()):4d}")
