"""Print the two tables from the saved records. No GPU needed."""
from __future__ import annotations

import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def gm(v):
    v = [x for x in v if x]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


def step1():
    rows = [r for r in load("results.jsonl") if r.get("margin")]
    if not rows:
        print("no step-1 records")
        return
    best = {}
    for r in rows:  # last write wins
        best[r["id"]] = r
    rows = list(best.values())
    epis = ["silu", "fp8cast", "resscale", "softcap", "gate", "chain", "mul3", "rowsum", "bias_relu", "gelu"]
    epis = [e for e in epis if any(r["epi"] == e for r in rows)]
    byshape = collections.defaultdict(dict)
    for r in rows:
        byshape[(r["group"], r["M"], r["N"], r["K"])][r["epi"]] = r

    print("=" * 110)
    print("STEP 1 -- margin = (cuBLAS mm + separate epilogue kernel) / (Inductor Triton template, epilogue fused)")
    print("          > 1.000 means the fused Triton template is faster.  'F' = Inductor actually fused into")
    print("          the template; without F the template ran and the epilogue stayed a separate kernel.")
    print("=" * 110)
    print(f"{'group':9s}{'shape':22s}" + "".join(f"{e:>12s}" for e in epis))
    for k in sorted(byshape, key=lambda k: (k[0], k[1], k[2], k[3])):
        g, M, N, K = k
        line = f"{g[:8]:9s}{f'{M}x{N}x{K}':22s}"
        for e in epis:
            r = byshape[k].get(e)
            if not r:
                line += f"{'-':>12s}"
            else:
                line += f"{r['margin']:>10.3f}{'F' if r.get('fused') else ' ':>2s}"
        print(line)

    print("\nwins by epilogue (margin > 1.0), and the best margin seen")
    for e in epis:
        v = [r for r in rows if r["epi"] == e]
        w = [r for r in v if r["margin"] > 1.0]
        wf = [r for r in w if r.get("fused")]
        b = max(v, key=lambda r: r["margin"])
        print(f"  {e:10s} {len(w):3d}/{len(v):3d} win   {len(wf):3d} of those actually fused   "
              f"best {b['margin']:.3f} at {b['M']}x{b['N']}x{b['K']}   geomean {gm([r['margin'] for r in v]):.3f}")
    allw = [r for r in rows if r["margin"] > 1.0]
    print(f"  {'ALL':10s} {len(allw):3d}/{len(rows):3d} win   geomean {gm([r['margin'] for r in rows]):.3f}")


ARMS = (("_hot_cublas_gemm", "cuBLAS"), ("_mainloop_no_epilogue", "mainloop"), ("exact2k", "exact2k"),
        ("aten_e", "aten_e"), ("triton", "inductor"), ("triton_e", "ind+emul"), ("triton_tuned", "ind_tuned"),
        ("ours", "ours"), ("ours_tuned", "ours_tuned"), ("ours_divrn", "ours_divrn"), ("ours_helpers_tuned",
                                                                                       "ours_helpr"))


def step3():
    """The three-way table. Only records from the round-robin harness (three_way.py) are read;
    the earlier ones timed each arm in its own process, which the clock ramp makes unusable."""
    rows = [r for r in load("three_way.jsonl") if "exact2k" in r]
    if not rows:
        print("\nno step-3 records")
        return
    merged = {}
    for r in rows:  # last write wins
        merged[(r["M"], r["N"], r["K"], r["epi"])] = r

    print("\n" + "=" * 124)
    print("STEP 3 -- device time in ms. Every arm is timed in ONE process, round-robin, after the clock is")
    print("          warmed; CUDA-graph replay with the L2 flushed between replays; best median of 3 rounds.")
    print("            cuBLAS     the GEMM alone through a hot cuBLASLt closure -- the floor")
    print("            mainloop   Inductor's own template with the epilogue deleted -- the fused floor")
    print("            exact2k    ARM 1: cuBLAS, then the epilogue as one separate byte-exact kernel")
    print("            aten_e     what torch.compile gives with the ATEN backend: extern mm + one pointwise")
    print("            inductor   ARM 2: the fused template torch.compile emits, its own tile")
    print("            ind+emul   arm 2 with emulate_precision_casts=1")
    print("            ind_tuned  arm 2's kernel swept over the SAME tile space arm 3 gets")
    print("            ours       ARM 3: arm 2's source, epilogue swapped, arm 2's tile")
    print("            ours_tuned ARM 3 re-tuned over the tile space")
    print("            ours_divrn arm 3 with torch's C++ transcribed literally (div.rn on the accurate exp)")
    print("            ours_helpr arm 3 with Inductor's NaN-propagating clamp, to separate that from rounding")
    print("=" * 124)
    hdr = f"{'shape':16s}{'epi':9s}"
    print(hdr + "".join(f"{lbl:>11s}" for _k, lbl in ARMS))
    for k in sorted(merged):
        m = merged[k]
        line = f"{m['M']}x{m['N']}x{m['K']:<6d}"[:16].ljust(16) + f"{m['epi']:9s}"
        for key, _lbl in ARMS:
            t = (m.get(key) or {}).get("ms")
            line += f"{t:>11.4f}" if t else f"{'-':>11s}"
        print(line)
    print(f"\n{'shape':16s}{'epi':9s}" + "".join(f"{lbl:>11s}" for _k, lbl in ARMS) +
          "   bytes identical to eager, out of N draws")
    for k in sorted(merged):
        m = merged[k]
        line = f"{m['M']}x{m['N']}x{m['K']:<6d}"[:16].ljust(16) + f"{m['epi']:9s}"
        for key, _lbl in ARMS:
            a = m.get(key) or {}
            cell = "-" if "bit_ok" not in a else f"{a['bit_ok']}/{a['draws']}"
            line += f"{cell:>11s}"
        print(line)

    print(f"\n{'shape':16s}{'epi':9s}{'2/3':>9s}{'1/3':>9s}{'1i/3':>9s}{'2e/3':>9s}{'2t/3t':>9s}   ours tile")
    print("   2/3 = Inductor's shipped fused kernel / ours re-tuned. Above 1.000 means byte-exactness is free.")
    for k in sorted(merged):
        m = merged[k]
        line = f"{m['M']}x{m['N']}x{m['K']:<6d}"[:16].ljust(16) + f"{m['epi']:9s}"
        for key in ("r_triton_over_ours_tuned", "r_exact2k_over_ours_tuned", "r_aten_e_over_ours_tuned",
                    "r_triton_e_over_ours_tuned", "r_triton_tuned_over_ours_tuned"):
            v = m.get(key)
            line += f"{v:>9.3f}" if v else f"{'-':>9s}"
        line += "   " + str((m.get("ours_tuned") or {}).get("tile", ""))
        print(line)
    for label, key in (("2/3   Inductor fused / ours re-tuned      ", "r_triton_over_ours_tuned"),
                       ("1/3   byte-exact two-kernel / ours        ", "r_exact2k_over_ours_tuned"),
                       ("2t/3t both given the same tile sweep      ", "r_triton_tuned_over_ours_tuned"),
                       ("2/3   at Inductor's own tile (not re-tuned)", "r_triton_over_ours")):
        v = [
            merged[k].get(key)
            for k in merged
            if merged[k].get(key) and (merged[k].get("ours_tuned") or merged[k].get("ours") or {}).get("bit_exact")
        ]
        if v:
            print(f"  {label} geomean {gm(v):.3f}   per case {', '.join(f'{x:.3f}' for x in v)}")


if __name__ == "__main__":
    step1()
    step3()
    sys.exit(0)
