"""Turn `attempts.jsonl` into the table the paper needs: one row per (model, layer, shape,
epilogue), the three arm times, the ratio and the byte-identical count.

    PYTHONPATH=<repo> .venv/bin/python artifact_eval/fusion_real/report.py [--pairing lora]
"""
from __future__ import annotations

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = ["lora", "fp8q", "lmhead", "swiglu", "resid"]
TITLE = {
    "lora": "1. LoRA merge -- y = base + (x @ A^T) @ B^T * (alpha/r), fused into lora_B, K = rank",
    "fp8q": "2. Quantization boundary -- y = (x * s) as fp8, K = hidden or ffn",
    "lmhead": "3. lm_head then argmax (greedy decode) -- the epilogue collapses vocab to one value",
    "swiglu": "4. SwiGLU -- silu(gate_proj) * up_proj, fused into up_proj, K = hidden",
    "resid": "5. Post-attention residual add -- o_proj + residual, K = hidden",
}


def load(path):
    """One row per case.  A case measured more than once keeps the run whose speedup is the
    MEDIAN of its repeats, and records how many repeats there were -- a ratio near 1.0 is only
    worth reporting if it survives being measured again on a shared machine."""
    runs = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "key" in r:
            runs.setdefault(tuple(r["key"]), []).append(r)
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
        print(f"{'model':12s} {'layer':22s} {'M x N x K':22s} {'epi':7s} {'base':>8s} {'ours':>8s} "
              f"{'free':>8s} {'ratio':>7s} {'bits':>6s}  note")
        print("-" * 118)
        ok = []
        for r in rs:
            shape = f"{r['M']}x{r['N']}x{r['K']}"
            if r.get("verdict") != "byte-identical":
                if args.wins_only:
                    continue
                why = r.get("verdict") or r.get("skip") or r.get("error") or "?"
                bits = f"{r.get('bit_ok', '-')}/{r.get('bit_total', '-')}"
                print(f"{r['model']:12s} {r['layer']:22s} {shape:22s} {r['epi']:7s} {'':>8s} {'':>8s} {'':>8s} "
                      f"{'--':>7s} {bits:>6s}  {why}")
                continue
            ok.append(r)
            fr = f"{r['free_fused_ms']:8.4f}" if r.get("free_fused_ms") else f"{'--':>8s}"
            print(f"{r['model']:12s} {r['layer']:22s} {shape:22s} {r['epi']:7s} {r['baseline_ms']:8.4f} "
                  f"{r['fused_ms']:8.4f} {fr} {r['speedup']:7.3f} {r['bit_ok']:3d}/{r['bit_total']:<3d} "
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

    print("\n" + "=" * 118)
    sp = [r["speedup"] for r in all_ok]
    if sp:
        wins = [x for x in sp if x > 1.0]
        print(f"all pairings: {len(wins)} wins of {len(sp)} byte-identical rows, geomean {gmean(sp):.3f}x, "
              f"best {max(sp):.3f}x")
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
