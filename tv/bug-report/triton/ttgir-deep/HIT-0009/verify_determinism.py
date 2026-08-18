"""Re-verify HIT-0009's two builds at a high rep count, under both allocation
patterns.

A 20-rep check only has ~54% power against a 4%-per-rep flaky build.  75 reps
gives ~95%, 115 gives ~99%.  This runs 200 per (side, pattern, input), which is
~99.98% power at p=0.04.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "/home/youngzt/fuzz/ttgir-deep")
from harness import runner as R
from harness.kernels import build_case

REPS = 200


def run_pattern(ctx, drop, dist, seed, n, pattern):
    """Return (n_reps_differing_from_first, n_distinct_outputs)."""
    case = ctx.case
    v = ctx.build(ctx.ttgir(drop), f"v_{pattern}")
    keep, base, bad, seen = [], None, 0, set()
    for _ in range(n):
        g = torch.Generator(device="cuda").manual_seed(seed)
        inp = case.make_inputs(g, dist)
        v.run(inp, case.grid)
        torch.cuda.synchronize()
        cur = {a: inp[a].clone() for a in case.out_args}
        if pattern == "alive":
            keep.append(inp); keep.append(cur)
        h = tuple(hash(R.int_view(cur[a]).cpu().numpy().tobytes()) for a in case.out_args)
        seen.add(h)
        if base is None:
            base = cur
        elif any(not R.bit_equal(base[a], cur[a]) for a in case.out_args):
            bad += 1
    del keep
    return bad, len(seen)


def main():
    R.ensure_allocator()
    m = json.load(open("artifacts/HIT-0009/meta.json"))
    case = build_case(m["family"], m["cfg"])
    ctx = R.CaseCtx(case, "cuda:90", Path("/tmp/v9"))
    culprit = m["culprit"]
    print(f"HIT-0009 case: {case.key}")
    print(f"reference = full pipeline minus {culprit};  candidate = full pipeline")
    print(f"{REPS} reps per (side, pattern, input)\n")
    inputs = [(m["dist"], 1262280619), ("randn", 1262280619), ("intvalued", 4242),
              ("tiny", 99), ("wide", 7)]
    rows, total_bad = [], 0
    for dist, seed in inputs:
        for side, drop in (("reference", culprit), ("candidate", [])):
            for pattern in ("alive", "free"):
                t0 = time.time()
                bad, ndistinct = run_pattern(ctx, drop, dist, seed, REPS, pattern)
                total_bad += bad
                rows.append((dist, seed, side, pattern, bad, ndistinct))
                print(f"  {dist:10s} seed {seed:<11d} {side:10s} {pattern:6s} "
                      f"-> {bad:3d}/{REPS} reps differ, {ndistinct} distinct outputs "
                      f"({time.time()-t0:.0f}s)")
    print()
    print(f"TOTAL self-inconsistent reps across all {len(rows)} runs "
          f"({len(rows)*REPS} launches): {total_bad}")
    verdict = "HOLDS" if total_bad == 0 else "FAILS"
    print(f"VERDICT: HIT-0009 {verdict}")
    Path("artifacts/HIT-0009/determinism_200reps.json").write_text(json.dumps(
        {"reps_per_run": REPS, "total_self_inconsistent_reps": total_bad,
         "verdict": verdict,
         "runs": [{"dist": d, "seed": s, "side": si, "pattern": p,
                   "reps_differing": b, "distinct_outputs": nd}
                  for d, s, si, p, b, nd in rows]}, indent=2))
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
