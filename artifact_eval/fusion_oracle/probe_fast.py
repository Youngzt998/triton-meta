"""Which config of our fused bit-exact kernel is fastest, and where does its time go?

The first step-3 run searched only the TMA + persistent + warp-specialized space and landed on a
config that was SLOWER than the unfused path. At K=16 the kernel is not a GEMM at all: A and B
together are 256 KB and C is 33 MB, so it is a store bound by memory with a little math attached.
Inductor's own autotuner picks a plain non-persistent 128x128x16 grid for it. This script searches
the plain space too, and prints the whole ranking rather than only the winner, so the next reader
can see how flat or sharp the choice is.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<repo> .venv/bin/python probe_fast.py --case 4096,4096,16,silu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from artifact import digest, hot_cublas, make_inputs  # noqa: E402
from cases import EPILOGUES, make_epi_args  # noqa: E402
import measure  # noqa: E402
import ours as O  # noqa: E402

ATTEMPTS = os.path.join(HERE, "attempts.jsonl")


def log(rec):
    with open(ATTEMPTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def space_of(name):
    if name == "plain":
        return O.PLAIN_SPACE
    if name == "tma":
        return O.TMA_SPACE
    if name == "both":
        return O.PLAIN_SPACE + O.TMA_SPACE
    if name == "wide":
        return O.WIDE_SPACE
    if "," in name:  # an explicit list: "128,128,16,8,8,4,None;64,128,16,8,4,3,None"
        out = []
        for tok in name.split(";"):
            f = tok.split(",")
            out.append(
                tuple(
                    int(v) if v not in ("None", "True", "False") else {"None": None, "True": True, "False": False}[v]
                    for v in f))
        return out
    raise SystemExit("space?")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--space", default="plain")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()

    M, N, K, epi = args.case.split(",")
    M, N, K = int(M), int(N), int(K)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    odt = O.FP8 if epi == "fp8cast" else dtype
    epi_id = O.EPI_ID[epi]
    needs_r = epi_id in O.EPI_NEEDS_R
    seed = M * 1000003 + N * 10007 + K

    a, w = make_inputs(torch, M, N, K, "fp16", 0, seed)
    if dtype is torch.bfloat16:
        a, w = a.to(dtype), w.to(dtype)
    extra = make_epi_args(epi, M, N, dtype, seed=seed)
    r = extra[0] if needs_r else None
    ref = digest(torch, EPILOGUES[epi](torch.mm(a, w), *extra))
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")

    def call(cfg, e=None):
        e = epi_id if e is None else e
        odt2 = dtype if e == O.EPI_NONE else odt
        if cfg[6] is not None:
            c = O.launch_plain_tma(a, w, odt2, e, r, cfg, mm_dtype=dtype)
            if c is not None:
                return c
            return None
        return O.launch_plain(a, w, odt2, e, r, cfg, mm_dtype=dtype)

    print(f"{M}x{N}x{K} {epi} {args.dtype}   space={args.space}")
    print(f"  warming the clock: {measure.warm(torch)}", flush=True)

    # compile and byte-check every candidate first, so the timing phase has no host work in it
    space = space_of(args.space)
    rows = []
    t0 = time.time()
    for i, cfg in enumerate(space):
        try:
            got = call(cfg)
            if got is None:
                continue
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            rows.append({"cfg": list(cfg), "err": type(e).__name__ + ": " + str(e)[:80]})
            continue
        rows.append({"cfg": list(cfg), "bit": digest(torch, got) == ref})
        del got
        if (i + 1) % 20 == 0:
            print(f"    compiled {i + 1}/{len(space)}  {time.time() - t0:.0f}s", flush=True)
    cand = [r for r in rows if r.get("bit")]
    bad = [r for r in rows if r.get("bit") is False]
    err = [r for r in rows if "err" in r]
    torch.cuda.empty_cache()

    # the floor: cuBLAS alone, and the unfused two-kernel path with our own exact epilogue
    from bitequiv.cublas_match import ltapi as L
    h = hot_cublas(L, torch, a, w, "fp16", dtype)
    mm = torch.mm(a, w)
    fns = {"gemm": h.run}
    for blk in (1024, 2048, 4096, 8192):
        for nw in (4, 8):
            fns[f"epi{blk}_{nw}"] = lambda blk=blk, nw=nw: O.launch_epi(mm, odt, epi_id, r, BLK=blk, num_warps=nw)
    t = measure.best_ms(torch, fns, flush, rounds=args.rounds, reps=args.reps)
    t_gemm = t.pop("gemm")
    best_epi_cfg, best_epi = min(t.items(), key=lambda kv: kv[1])
    print(f"  hot cuBLAS gemm alone      {t_gemm:.4f} ms")
    print(f"  our exact epilogue alone   {best_epi:.4f} ms  cfg={best_epi_cfg}")
    print(f"  two-kernel floor (sum)     {t_gemm + best_epi:.4f} ms\n")
    mm, h = None, None  # keep the names bound: the epilogue lambdas above close over `mm`
    torch.cuda.empty_cache()

    times = measure.best_ms(torch, {i: (lambda c=rr["cfg"]: call(tuple(c)))
                                    for i, rr in enumerate(cand)}, flush, rounds=args.rounds, reps=args.reps)
    for i, rr in enumerate(cand):
        rr["ms"] = times[i]
    ok_rows = sorted([r for r in cand if r.get("ms")], key=lambda r: r["ms"])
    print(f"\n  {len(rows)} configs tried, {len(ok_rows)} byte-identical + timed, "
          f"{len(bad)} byte-different, {len(err)} failed, {time.time() - t0:.0f}s")
    if bad:
        print(f"  byte-different configs: {[r['cfg'] for r in bad][:8]}")
    print(f"\n  {'BM':>4s}{'BN':>5s}{'BK':>5s}{'GM':>4s}{'nw':>4s}{'ns':>4s}{'ws':>7s}{'ms':>10s}{'vs 2k':>8s}")
    for rr in ok_rows[:args.top]:
        c = rr["cfg"]
        print(f"  {c[0]:>4d}{c[1]:>5d}{c[2]:>5d}{c[3]:>4d}{c[4]:>4d}{c[5]:>4d}{str(c[6]):>7s}"
              f"{rr['ms']:>10.4f}{(t_gemm + best_epi) / rr['ms']:>8.3f}")

    # where the best config's time goes
    if ok_rows:
        b = tuple(ok_rows[0]["cfg"])
        variants = {"full": lambda: call(b)}
        for label, e in (("mainloop_only", O.EPI_NONE), ("approx_sigmoid", O.EPI_SILU_FAST), ("exact_div_rn",
                                                                                              O.EPI_SILU_RN)):
            if e in (O.EPI_SILU_FAST, O.EPI_SILU_RN) and epi != "silu":
                continue
            try:
                call(b, e)
                torch.cuda.synchronize()
                variants[label] = lambda e=e: call(b, e)
            except Exception as ex:  # noqa: BLE001
                print("   ", label, "failed", str(ex)[:100])
        parts = measure.best_ms(torch, variants, flush, rounds=args.rounds, reps=args.reps)
        print(f"\n  best cfg {list(b)}: " + "   ".join(f"{k} {v:.4f}" for k, v in parts.items() if v))
        log({
            "step": "probe_fast", "M": M, "N": N, "K": K, "epi": epi, "dtype": args.dtype, "space": args.space, "when":
            time.strftime("%Y-%m-%d %H:%M:%S"), "gemm_ms": t_gemm, "epi_only_ms": best_epi, "epi_only_cfg":
            list(best_epi_cfg), "best": ok_rows[0], "parts": parts, "rounds": args.rounds, "reps": args.reps, "top":
            ok_rows[:args.top], "n_bit_bad": len(bad), "n_err": len(err), "n_ok": len(ok_rows)
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
