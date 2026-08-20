"""Which cuBLAS plan mode does each real-model case resolve to?

The heuristic query reads only the SHAPE -- `_cublas_direct(..., execute=False)` never touches a
data pointer and launches no kernel -- so this runs in a second on a shared GPU and allocates
nothing but the CUDA context.

    PYTHONPATH=<repo> .venv/bin/python plan_modes.py                 # the tally
    PYTHONPATH=<repo> .venv/bin/python plan_modes.py --json out.json # one record per case
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE, HERE, os.path.join(AE, "fusion_moe")):
    if p not in sys.path:
        sys.path.insert(0, p)


class _Shape:
    """Everything `_cublas_direct(execute=False)` reads off an operand: its shape."""

    def __init__(self, r, c):
        self.shape = (r, c)


def plan_of(M, N, K, kind, out_dtype):
    """(plan, None) for one shape, or (None, why) when cuBLAS or the planner declines.

    Nothing runs on the device: `_cublas_direct(execute=False)` asks cuBLAS's own heuristic what
    it WOULD run, and the heuristic reads the layouts, not the data.
    """
    from bitequiv.cublas_match.arch import platform
    from bitequiv.cublas_match.ltapi import _cublas_direct
    from bitequiv.cublas_match.plan import static_plan
    try:
        _d, config = _cublas_direct(_Shape(M, K), _Shape(K, N), kind, out_dtype, execute=False)
    except Exception as e:
        return None, f"cublas declined: {e}"
    plan, reason = static_plan(platform(), M, N, K, kind, config)
    return (plan, None) if plan is not None else (None, reason)


def mode_of(M, N, K, kind, out_dtype):
    """(mode, detail): the plan's mode name and the plan itself, or (None, why)."""
    plan, why = plan_of(M, N, K, kind, out_dtype)
    return (plan.mode, plan) if plan is not None else (None, why)


def main():
    import torch

    from cases import out_dtype_for
    from model_cases import reachable_cases
    from models_2026 import GROUPS

    p = argparse.ArgumentParser()
    p.add_argument("--groups", default=",".join(GROUPS))
    p.add_argument("--dtypes", default="fp16,bf16")
    p.add_argument("--json", default="")
    args = p.parse_args()

    ok, skipped = reachable_cases(tuple(g.strip() for g in args.groups.split(",")))
    torch_dt = {"fp16": torch.float16, "bf16": torch.bfloat16}
    recs, tally = [], Counter()
    for dt in args.dtypes.split(","):
        dt = dt.strip()
        for c in ok:
            out_dtype = out_dtype_for(c.epi, torch_dt[dt])
            mode, detail = mode_of(c.M, c.N, c.K, dt, out_dtype)
            tally[(dt, mode or "DECLINED")] += 1
            recs.append({
                "model": c.model, "layer": c.layer, "pairing": c.pairing, "M": c.M, "N": c.N, "K": c.K, "epi": c.epi,
                "lim": c.lim, "note": c.note, "dtype": dt, "out_dtype": str(out_dtype).replace("torch.", ""), "mode":
                mode, "why": None if mode else detail, "k_chunk": getattr(detail, "k_chunk", None) if mode else None,
                "algo_id": getattr(detail, "algo_id", None) if mode else None
            })
    print(f"{len(ok)} reachable cases, {len(skipped)} skipped\n")
    for (dt, mode), n in sorted(tally.items()):
        print(f"  {dt:5s} {mode:14s} {n:4d}")
    print("\nsplit cases (fp16):")
    for r in recs:
        if r["dtype"] == "fp16" and r["mode"] == "split":
            print(f"  {r['model']:24s} {r['layer']:34s} {r['M']:>6}x{r['N']:>6}x{r['K']:<6} {r['epi']}")
    if args.json:
        with open(args.json, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(recs)} records to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
