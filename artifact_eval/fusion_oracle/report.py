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
    for r in rows:  # last write wins. `params_tag` is part of the key because two LoRA cases at
        # the same shape with different alpha/rank are different kernels, not a rerun.
        merged[(r["M"], r["N"], r["K"], r["epi"], r.get("params_tag", ""))] = r

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


def _ms(rec, arm):
    return (rec.get(arm) or {}).get("ms")


def _bits(rec, arm):
    a = rec.get(arm) or {}
    return None if "bit_ok" not in a else (a["bit_ok"], a["draws"])


def constraint(tag="oracle2"):
    """What byte-exactness costs on the shapes where fusing already paid.

    Arms are numbered the way the question is asked, which is NOT the order `three_way.py` prints:

      arm 1  `triton`   the kernel torch.compile emits under max-autotune with the Triton backend.
                        Fused, and under no bit constraint.
      arm 2  `exact2k`  cuBLAS, then the epilogue as one separate kernel. The baseline a user has
                        today, and the bit reference.
      arm 3  `ours`     arm 1's own generated source with only the epilogue arithmetic replaced.
                        Same mainloop, same tile, same grid, same launch.

    A case counts only if arm 1 beat arm 2 -- fusion has to be a win before asking what the
    constraint costs -- and only if arm 3 was byte-identical to arm 2 on every draw. A time from a
    kernel that is not byte-identical is a failure, not a speedup.
    """
    rows = [r for r in load("three_way.jsonl") if r.get("tag") == tag]
    if not rows:
        print(f"\nno records tagged {tag!r}")
        return
    merged = {}
    for r in rows:  # last write wins. `params_tag` is part of the key because two LoRA cases at
        # the same shape with different alpha/rank are different kernels, not a rerun.
        merged[(r["M"], r["N"], r["K"], r["epi"], r.get("params_tag", ""))] = r
    recs = [merged[k] for k in sorted(merged)]

    tried, cases = [], []
    for r in recs:
        a1, a2, a3 = _ms(r, "triton"), _ms(r, "exact2k"), _ms(r, "ours")
        b3, b1 = _bits(r, "ours"), _bits(r, "triton")
        row = {
            "r": r,
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "b3": b3,
            "b1": b1,
            "fused": bool((r.get("triton") or {}).get("fused")),
            "win": bool(a1 and a2 and a2 > a1),
            "exact": bool((r.get("ours") or {}).get("bit_exact")),
        }
        tried.append(row)
        if row["win"] and row["exact"] and a3:
            cases.append(row)

    def shp(r):
        return f"{r['M']}x{r['N']}x{r['K']}"

    print("\n" + "=" * 132)
    print("THE COST OF BYTE-EXACTNESS, on the shapes where fusion was already a win")
    print("  arm 1  torch.compile's own fused Triton kernel, no bit constraint")
    print("  arm 2  cuBLAS + the epilogue as one separate kernel -- the baseline, and the bit reference")
    print("  arm 3  arm 1's kernel with only the epilogue arithmetic swapped -- same tile, same grid, same launch")
    print("  2/3 above 1.000 = the byte-exact fused kernel still beats the cuBLAS baseline")
    print("  1/3 below 1.000 = the bit constraint cost something")
    print("  device time in ms: CUDA-graph replay, L2 flushed between replays, clock warmed,")
    print("  arms interleaved within a round, best median of N rounds")
    print("=" * 132)
    hdr = (f"{'shape':17s}{'epi':10s}{'arm1':>9s}{'arm2':>9s}{'arm3':>9s}{'2/3':>8s}{'1/3':>8s}"
           f"{'arm3 bytes':>12s}{'arm1 bytes':>12s}")
    print(hdr)
    for c in cases:
        r = c["r"]
        b3 = f"{c['b3'][0]}/{c['b3'][1]}" if c["b3"] else "-"
        b1 = f"{c['b1'][0]}/{c['b1'][1]}" if c["b1"] else "-"
        print(f"{shp(r):17s}{r['epi']:10s}{c['a1']:>9.4f}{c['a2']:>9.4f}{c['a3']:>9.4f}"
              f"{c['a2'] / c['a3']:>8.3f}{c['a1'] / c['a3']:>8.3f}{b3:>12s}{b1:>12s}")

    if not cases:
        return
    r23 = [c["a2"] / c["a3"] for c in cases]
    r13 = [c["a1"] / c["a3"] for c in cases]
    print(f"\n  {len(cases)} cases")
    for label, v in (("2/3  bit-exact fused vs the cuBLAS baseline", r23), ("1/3  bit-exact fused vs unconstrained "
                                                                            "fused", r13)):
        s = sorted(v)
        q = lambda f: s[min(len(s) - 1, int(f * len(s)))]  # noqa: E731
        print(f"  {label:44s} geomean {gm(v):.3f}   min {s[0]:.3f}  p25 {q(.25):.3f}  "
              f"median {q(.5):.3f}  p75 {q(.75):.3f}  max {s[-1]:.3f}   "
              f"{sum(1 for x in v if x > 1.0)}/{len(v)} above 1.0")

    # per epilogue, because the two halves of the answer split cleanly along it
    print(f"\n  {'epilogue':12s}{'n':>4s}{'2/3 geomean':>14s}{'1/3 geomean':>14s}")
    for e in sorted({c["r"]["epi"] for c in cases}):
        v = [c for c in cases if c["r"]["epi"] == e]
        print(f"  {e:12s}{len(v):>4d}{gm([c['a2'] / c['a3'] for c in v]):>14.3f}"
              f"{gm([c['a1'] / c['a3'] for c in v]):>14.3f}")

    # hit rate: how many candidates had to be tried to get the cases above
    n_fused = sum(1 for t in tried if t["fused"])
    n_win = sum(1 for t in tried if t["win"])
    n_exact = sum(1 for t in tried if t["exact"])
    print(f"\n  candidates measured        {len(tried)}")
    print(f"  arm 1 was one fused kernel {n_fused}")
    print(f"  arm 1 beat arm 2 (warmed)  {n_win}   -- hit rate {n_win / len(tried):.0%} of candidates")
    print(f"  arm 3 byte-identical       {n_exact}")
    print(f"  both, i.e. usable cases    {len(cases)}")

    # arm 1 against eager: a finding on its own
    tot = sum(c["b1"][1] for c in tried if c["b1"])
    ok = sum(c["b1"][0] for c in tried if c["b1"])
    n_any = sum(1 for c in tried if c["b1"] and c["b1"][0] > 0)
    print(f"\n  arm 1 vs eager: {ok} of {tot} draws matched, over {sum(1 for c in tried if c['b1'])} cases; "
          f"{n_any} case(s) matched on even one draw")

    # How far above the SAME template with the epilogue deleted each arm sits. This is the frame
    # that makes the spread readable: the epilogue is nearly free on a small-K kernel that is
    # already store-bound, so what looks like a big ratio between two arms is often a small
    # absolute difference on top of a floor they share.
    fl = [c for c in cases if _ms(c["r"], "_mainloop_no_epilogue")]
    if fl:
        print("\n  above the same template with the epilogue DELETED (1.000 = the epilogue is free)")
        print(f"  {'epilogue':12s}{'n':>4s}{'arm3':>9s}{'arm1':>9s}{'arm2':>9s}")
        for e in sorted({c["r"]["epi"] for c in fl}) + ["ALL"]:
            w = fl if e == "ALL" else [c for c in fl if c["r"]["epi"] == e]
            f = [_ms(c["r"], "_mainloop_no_epilogue") for c in w]
            print(f"  {e:12s}{len(w):>4d}{gm([c['a3'] / x for c, x in zip(w, f)]):>9.3f}"
                  f"{gm([c['a1'] / x for c, x in zip(w, f)]):>9.3f}{gm([c['a2'] / x for c, x in zip(w, f)]):>9.3f}")

    # `ours_approx` is arm 3's own kernel with the approximate spelling of the same epilogue -- so
    # it differs from arm 3 in the arithmetic and in NOTHING else: same tile, same grid, same
    # harness. Careful reading it: for silu the variant KEEPS the accumulator rounding and only
    # swaps the correctly rounded sigmoid for `tl.sigmoid`, so there it prices the sigmoid. For
    # every other epilogue the variant DROPS the accumulator rounding, so there it prices exactly
    # that one rounding -- which is the only thing Inductor's `emulate_precision_casts` leaves out.
    ca = [c for c in cases if _ms(c["r"], "ours_approx")]
    if ca:
        print("\n  arm 3 against its own kernel with the approximate spelling (below 1.000 = the exact")
        print("  spelling costs). 'what it changes' says which single thing the variant drops.")
        print(f"  {'epilogue':12s}{'n':>4s}{'approx/arm3':>14s}{'arm1/arm3':>12s}   what the variant changes")
        for e in sorted({c["r"]["epi"] for c in ca}):
            w = [c for c in ca if c["r"]["epi"] == e]
            what = ("the sigmoid only (rounding kept)" if e == "silu" else "the accumulator rounding only")
            print(f"  {e:12s}{len(w):>4d}{gm([_ms(c['r'], 'ours_approx') / c['a3'] for c in w]):>14.3f}"
                  f"{gm([c['a1'] / c['a3'] for c in w]):>12.3f}   {what}")

    # is the gap a spelling Inductor could adopt? `triton_e` is Inductor's own emulate_precision_casts
    ce = [c for c in cases if _ms(c["r"], "triton_e") and (c["r"].get("triton_e") or {}).get("bit_exact")]
    if ce:
        v = [_ms(c["r"], "triton_e") / c["a3"] for c in ce]
        print(f"\n  Inductor's own emulate_precision_casts kernel was byte-exact on {len(ce)}/{len(cases)} cases, "
              f"at {gm(v):.3f}x arm 3 -- i.e. that much of the gap is a spelling Inductor already has")
    cc = [c for c in cases if _ms(c["r"], "ours_cheap")]
    if cc:
        v = [_ms(c["r"], "ours_cheap") / c["a3"] for c in cc]
        nx = sum(1 for c in cc if (c["r"].get("ours_cheap") or {}).get("bit_exact"))
        print(f"  the NaN-dropping clamp ran {gm(v):.3f}x arm 3 on {len(cc)} fp8cast cases "
              f"({nx} of them byte-exact on the draws, though not over the whole domain)")


if __name__ == "__main__":
    step1()
    step3()
    constraint(sys.argv[1] if len(sys.argv) > 1 else "oracle2")
    sys.exit(0)
