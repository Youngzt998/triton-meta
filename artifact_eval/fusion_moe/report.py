"""Turn `attempts.jsonl` into the table the paper needs: one row per (model, layer, shape,
epilogue), the three arm times, the ratio and the byte-identical count.

    PYTHONPATH=<repo> .venv/bin/python artifact_eval/fusion_real/report.py [--pairing lora]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ORDER = ["moe_up", "moe_down", "lora", "lmhead", "attn"]
TITLE = {
    "moe_up": "1a. MoE expert gate/up -- (tokens routed to one expert) x moe_intermediate x expert_in, "
    "epilogue = that model's activation",
    "moe_down": "1b. MoE expert down -- (tokens routed to one expert) x expert_in x moe_intermediate, "
    "epilogue = multiply by the routing weight before the experts are summed",
    "lora": "2. LoRA merge -- y = base + (x @ A^T) @ B^T * (alpha/r), fused into lora_B, K = rank",
    "lmhead": "3. lm_head then argmax (greedy decode) -- the epilogue collapses vocab to one value",
    "attn": "4. Attention projections -- o_proj + residual add, and qkv_proj at the fp8 boundary",
}


def current_keys():
    """The log is append-only, so it can hold a case that has since been removed from the case
    list.  Two `o_proj` rows for DeepSeek-V4 were logged before it turned out that model has no
    single output projection; they are dropped here rather than edited out of the log."""
    from models_2026 import GROUPS, build
    return {tuple(c.key) for c in build(list(GROUPS))}


def load(path):
    """One row per case.  A case measured more than once keeps the run whose speedup is the
    MEDIAN of its repeats, and records how many repeats there were -- a ratio near 1.0 is only
    worth reporting if it survives being measured again on a shared machine."""
    live = current_keys()
    runs, dropped = {}, 0
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "key" not in r:
            continue
        k = tuple(r["key"])
        if k not in live:
            dropped += 1
            continue
        runs.setdefault(k, []).append(r)
    if dropped:
        print(f"({dropped} log records are for cases no longer in the case list -- dropped)")
    out = []
    for key, rs in runs.items():
        good = [r for r in rs if r.get("verdict") == "byte-identical"]
        if good:
            good.sort(key=lambda r: r["speedup"])
            pick = dict(good[len(good) // 2])
            pick["repeats"] = len(good)
            pick["speedup_spread"] = round(good[-1]["speedup"] - good[0]["speedup"], 4)
            pick["bit_ok"] = sum(r["bit_ok"] for r in good)
            pick["bit_total"] = sum(r["bit_total"] for r in good)
        else:
            pick = dict(rs[-1])
            pick["repeats"] = len(rs)
        out.append(pick)
    return out


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(HERE, "attempts.jsonl"))
    ap.add_argument("--pairing", default="")
    ap.add_argument("--wins-only", action="store_true")
    args = ap.parse_args()

    rows = load(args.log)
    if args.pairing:
        rows = [r for r in rows if r.get("pairing") in args.pairing.split(",")]

    all_ok = []
    for pairing in ORDER:
        rs = [r for r in rows if r.get("pairing") == pairing]
        if not rs:
            continue
        rs.sort(key=lambda r: (r["model"], r["layer"], r["K"], -r["M"]))
        print(f"\n{TITLE[pairing]}")
        print(f"{'model':22s} {'layer':32s} {'M x N x K':22s} {'epi':9s} {'base':>8s} {'ours':>8s} "
              f"{'free':>8s} {'nornd':>8s} {'ratio':>7s} {'bits':>6s}  note")
        print("-" * 140)
        ok = []
        for r in rs:
            shape = f"{r['M']}x{r['N']}x{r['K']}"
            if r.get("verdict") != "byte-identical":
                if args.wins_only:
                    continue
                why = r.get("verdict") or r.get("skip") or r.get("error") or "?"
                bits = f"{r.get('bit_ok', '-')}/{r.get('bit_total', '-')}"
                print(f"{r['model']:22s} {r['layer']:32s} {shape:22s} {r['epi']:9s} {'':>8s} {'':>8s} {'':>8s} "
                      f"{'':>8s} {'--':>7s} {bits:>6s}  {why}")
                continue
            ok.append(r)
            fr = f"{r['free_fused_ms']:8.4f}" if r.get("free_fused_ms") else f"{'--':>8s}"
            nr = f"{r['noround_ms']:8.4f}" if r.get("noround_ms") else f"{'--':>8s}"
            print(f"{r['model']:22s} {r['layer']:32s} {shape:22s} {r['epi']:9s} {r['baseline_ms']:8.4f} "
                  f"{r['fused_ms']:8.4f} {fr} {nr} {r['speedup']:7.3f} {r['bit_ok']:3d}/{r['bit_total']:<3d} "
                  f"{'WIN' if r['speedup'] > 1.0 else '   '} "
                  f"{('+-%.3f over %d runs' % (r['speedup_spread'], r['repeats'])) if r.get('repeats', 1) > 1 else ''}")
        sp = [r["speedup"] for r in ok]
        wins = [x for x in sp if x > 1.0]
        if sp:
            print(f"  -> {len(wins)} wins of {len(sp)} byte-identical rows ({len(rs)} attempted); "
                  f"geomean {gmean(sp):.3f}x, best {max(sp):.3f}x, worst {min(sp):.3f}x")
            if wins:
                print(f"     over the wins only: geomean {gmean(wins):.3f}x")
        all_ok += ok

    print("\n" + "=" * 140)
    sp = [r["speedup"] for r in all_ok]
    if sp:
        wins = [x for x in sp if x > 1.0]
        print(f"all pairings: {len(wins)} wins of {len(sp)} byte-identical rows, geomean {gmean(sp):.3f}x, "
              f"best {max(sp):.3f}x")
    # The fourth arm.  `ours / noround` is OUR kernel divided by the same kernel with one line
    # changed -- the epilogue reading the unrounded fp32 accumulator.  That rounding IS the bit
    # constraint, so this ratio is its price with the kernel structure held fixed.
    print("\nthe price of the bit constraint (arm 2 / the same kernel with the rounding removed)")
    nr_all = []
    for pairing in ORDER:
        rt = sorted(r["fused_ms"] / r["noround_ms"]
                    for r in all_ok
                    if r.get("pairing") == pairing and r.get("noround_ms"))
        if rt:
            nr_all += rt
            print(f"  {pairing:10s} n={len(rt):4d}  geomean {gmean(rt):.4f}  median {rt[len(rt) // 2]:.4f}  "
                  f"min {rt[0]:.4f}  max {rt[-1]:.4f}")
    if nr_all:
        nr_all.sort()
        print(f"  {'ALL':10s} n={len(nr_all):4d}  geomean {gmean(nr_all):.4f}  median {nr_all[len(nr_all) // 2]:.4f}  "
              f"min {nr_all[0]:.4f}  max {nr_all[-1]:.4f}")
    bad = [r for r in rows if r.get("verdict") == "NOT BYTE-IDENTICAL"]
    nonplain = [r for r in rows if str(r.get("verdict", "")).startswith("NOT REPRODUCIBLE")]
    errs = [r for r in rows if "error" in r]
    print(f"byte gate failures: {len(bad)}   shapes cuBLAS routes off `plain`: {len(nonplain)}   errors: {len(errs)}")
    for r in bad:
        print(f"   FAILED BITS {r['model']} {r['layer']} {r['M']}x{r['N']}x{r['K']} {r['epi']} "
              f"{r.get('bit_ok')}/{r.get('bit_total')}")
    modes = {}
    for r in nonplain:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    if modes:
        print(f"   off-`plain` plan modes: {modes}")
    # Was TMA ever REFUSED?  A descriptor needs every stride but the innermost to be a whole
    # number of 16-byte lines: for a row-major A that is K * 2 bytes, for B and the output it is
    # N * elem bytes.  The previous run lost rows at N = 8184 because the fp8 output's row stride
    # was 8184 bytes.  Every out_features and every vocab in a real Llama is a multiple of 16, so
    # nothing here is refused -- the rows below chose the non-TMA kernel because the tuner found
    # it faster at small M, which is a different thing.
    def tma_legal(r):
        w = 16 if r["epi"] == "fp8q" else 8
        return r["K"] % 8 == 0 and r["N"] % 8 == 0 and r["N"] % w == 0

    refused = [r for r in all_ok if not tma_legal(r)]
    chose_plain = [r for r in all_ok if r.get("tma") is False and tma_legal(r)]
    print(f"rows where TMA was REFUSED by the stride: {len(refused)}")
    for r in refused:
        print(f"   TMA REFUSED {r['model']} {r['layer']} {r['M']}x{r['N']}x{r['K']} {r['epi']}")
    print(f"rows where TMA was legal but the tuner preferred the non-TMA kernel: {len(chose_plain)}"
          f" (median M {sorted(x['M'] for x in chose_plain)[len(chose_plain) // 2] if chose_plain else '-'})")


if __name__ == "__main__":
    main()
