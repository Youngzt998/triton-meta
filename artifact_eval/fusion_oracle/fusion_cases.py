"""Which of the 405 real-model cases `gemm.fusion` measures, and the rule that chose them.

`fusion_moe/models_2026.py` carries 405 reachable (model, layer, epilogue) cases.  Measuring all
of them is not the point: on 361 of them cuBLAS picks plan mode `plain`, and on `plain` the bit
constraint reduces to the epilogue's rounding, so a 361-row table would be one answer repeated.
The 17 cases where cuBLAS picks something else are the interesting ones and none may be dropped.

THE SELECTION RULE
------------------
1. **Every `split` case, all 17.**  They are the only cases in the whole model table where cuBLAS
   does not pick `plain`, so they are the only ones where the bit constraint reaches past the
   epilogue and into the mainloop.  15 are an MoE expert `up_proj` and 2 a dense `down_proj`.

2. **One `plain` case per (model, layer group)**, for the models `models_2026.py` records a
   ranking fact for, walked in this order and stopped at `MAX_PLAIN`:

   *Model rank.*  That file picked its models by Hugging Face trending score and 30-day download
   count on 2026-08-16, and its per-model comment records which of the two put that model in the
   table.  `MODEL_RANK` below is that evidence, nothing else: the two models with a trending
   position come first, in position order, then the models with a download count, descending.
   The six models the file records no number for are not drawn from at all -- they are still in
   the table through step 1 whenever they own a `split` case.

   *Layer importance within the model.*  `PAIRING_RANK`: the routed expert FFN first, then the
   dense FFN, then the attention output projection, then the LoRA merge.  That is the order these
   layers cost in a decode-only transformer -- an expert or dense FFN is roughly four times an
   attention projection per layer, and a LoRA merge at rank 8..64 is a rounding error of FLOPs
   next to either.

   *Which case stands for that layer.*  `_within` below: the larger token count first, and for
   the MoE layers the even routing before the hot or cold expert, because even is the mean of the
   distribution the other two bracket.  The first case in that order whose mode is `plain` wins;
   if the layer's cases are all `split` they are already in through step 1.

3. **Both dtypes.**  Every selected case is measured at fp16 and at bf16.  The models in the table
   ship in bf16 and the artifact's GEMM supports both, and a plan is a function of the dtype, so
   the mode is resolved per (case, dtype) rather than assumed from one of them.

WHAT THE RULE LEAVES OUT, ON PURPOSE
------------------------------------
* the 56 `argmax` / `lm_head` cases -- `model_cases.ARGMAX_VERDICT` says why, at length.
* the 27 `fp8q` cases on `qkv_proj`.  Their output dtype is fp8, and `cublas_equivalent_gemm`
  refuses an output dtype cuBLAS is not being told about, so there is no bit reference for them
  through this path.  Measured, not assumed: `plan_modes.py` declines all 27 at both dtypes.
* the hot and cold routing factors, and the smaller token count, of a layer already represented.
* every `plain` case of a model the table records no ranking number for.

A reviewer pruning rows afterwards should know that this list is already a pruning, and that the
axis it pruned hardest is the one it says least about: routing balance and token count.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE, HERE, os.path.join(AE, "fusion_moe")):
    if p not in sys.path:
        sys.path.insert(0, p)

# The ranking fact `models_2026.py` records for each model, in its own words.  A model absent
# from this dict is one that file gives no number for; it is not ranked and not drawn from.
#   ("trend", n)     "the highest trending score" / "4th by trending score"
#   ("dl", millions) "N.NNM downloads in 30 days"
MODEL_RANK = {
    "Qwen3.8-2.4T-A95B": ("trend", 1),
    "Nemotron-3.5-L-30B-A3B": ("trend", 4),
    "gpt-oss-20b": ("dl", 7.97),
    "gpt-oss-120b": ("dl", 4.48),
    "granite-4.1-8b": ("dl", 3.38),
    "GLM-5.2": ("dl", 2.71),
    "Qwen3-30B-A3B": ("dl", 2.63),
    "DeepSeek-V4-Flash": ("dl", 2.05),
    "GLM-4.7-Flash": ("dl", 2.02),
    "DeepSeek-V4-Pro": ("dl", 1.28),
}

# A trending position and a download count are not the same number, so they are not mixed: the
# trending models rank by position, then the download models by count descending.
MODEL_ORDER = ([m for m, (k, v) in sorted(MODEL_RANK.items(), key=lambda kv: kv[1][1]) if k == "trend"] +
               [m for m, (k, v) in sorted(MODEL_RANK.items(), key=lambda kv: -kv[1][1]) if k == "dl"])

PAIRING_RANK = {"moe_up": 0, "moe_down": 1, "mlp_up": 2, "mlp_down": 3, "attn": 4, "lora": 5}

MAX_PLAIN = 40


def _within(case):
    """Order the cases of one (model, layer group), most representative first.

    Bigger token count first; within a token count, even routing before the hot and cold experts
    it is the mean of; for LoRA, the projection every model has before the routed-expert one, and
    the smallest rank first, because the rank is what makes a LoRA merge a fusion candidate.
    """
    layer = case.layer
    tokens = -case.M
    balance = 0 if "even" in layer else (1 if "hot" in layer else (2 if "cold" in layer else 0))
    lora_layer = 0 if layer.startswith("o_proj") else (1 if layer.startswith("q_proj") else 2)
    rank = int(layer.split("r=")[1].split()[0]) if "r=" in layer else 0
    return (balance, tokens, lora_layer, rank, layer)


def select(modes, max_plain=MAX_PLAIN):
    """(chosen, why) from the mode table `plan_modes.py` writes.

    `modes` is {(M, N, K, epi, dtype): mode}.  A case is `split` if it resolves to `split` at ANY
    of the dtypes being measured -- a case that splits at one dtype is the interesting kind and
    must not be dropped because the other dtype happened to land on `plain`.
    """
    from model_cases import reachable_cases
    from models_2026 import GROUPS
    # Every group, `mlp` included.  `reachable_cases()`'s own default leaves `mlp` out, and the
    # dense FFN is where 2 of the 17 `split` cases live.
    cases, _skipped = reachable_cases(tuple(GROUPS))

    def mode_set(c):
        return {m for (M, N, K, epi, _dt), m in modes.items() if (M, N, K, epi) == (c.M, c.N, c.K, c.epi)}

    chosen, why = [], {}
    seen = set()
    for c in cases:  # step 1: every split case
        if "split" in mode_set(c) and c.key not in seen:
            seen.add(c.key)
            chosen.append(c)
            why[c.key] = "split: cuBLAS does not pick plain here"

    by_group = {}
    for c in cases:
        if c.model in MODEL_RANK and c.pairing in PAIRING_RANK:
            by_group.setdefault((c.model, c.pairing), []).append(c)

    groups = sorted(by_group, key=lambda g: (MODEL_ORDER.index(g[0]), PAIRING_RANK[g[1]]))
    n_plain = 0
    for model, pairing in groups:
        if n_plain >= max_plain:
            break
        for c in sorted(by_group[(model, pairing)], key=_within):
            if c.key in seen:
                continue
            if mode_set(c) == {"plain"}:
                seen.add(c.key)
                chosen.append(c)
                why[c.key] = (f"plain: {model} is #{MODEL_ORDER.index(model) + 1} by the rank fact "
                              f"models_2026.py records, {pairing} is rank {PAIRING_RANK[pairing]} by layer cost")
                n_plain += 1
                break
    return chosen, why


def main():
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--modes", default=os.path.join(HERE, "plan_modes.jsonl"))
    p.add_argument("--max-plain", type=int, default=MAX_PLAIN)
    args = p.parse_args()
    modes = {}
    for line in open(args.modes):
        r = json.loads(line)
        modes[(r["M"], r["N"], r["K"], r["epi"], r["dtype"])] = r["mode"]
    chosen, why = select(modes, args.max_plain)
    n_split = sum(1 for c in chosen if why[c.key].startswith("split"))
    print(f"{len(chosen)} cases: {n_split} split, {len(chosen) - n_split} plain\n")
    for c in chosen:
        print(f"  {why[c.key][:5]:6s} {c.model:24s} {c.layer:36s} {c.M:>6}x{c.N:>6}x{c.K:<6} {c.epi}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
