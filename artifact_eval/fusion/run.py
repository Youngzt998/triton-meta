"""Is there a workload where a fused BIT-EXACT Triton kernel beats the two-kernel path cuBLAS
gives a user today?

Three arms, same shape, same inputs, device time by CUDA-graph replay with L2 flushed:

  arm 1  baseline       cuBLAS through a hot closure, then the epilogue as ONE Triton kernel.
                        One, not several, even for the four-op chain -- Inductor fuses a
                        pointwise chain into a single kernel even when it cannot fuse into the
                        GEMM, so a multi-kernel baseline would be a straw man. The epilogue
                        kernel's block size and warp count are tuned too, so the baseline is not
                        losing to a launch-shape accident.
  arm 2  fused          the bit-exact GEMM with the epilogue folded in. The launch shape is
                        tuned over tile sizes, which are bit-neutral -- a tile only says which
                        output elements one program owns, the k loop inside is the same either
                        way -- and then the WINNING config is put through the byte gate. If it
                        is not byte-identical on every draw the pair is reported as a failure,
                        not a speedup.
  arm 3  unconstrained  the same fusion on an ordinary autotuned Triton matmul, accumulator
                        straight into the epilogue with no rounding at the GEMM boundary. The
                        ceiling, and what torch.compile with max-autotune would pick.

Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<repo> .venv/bin/python artifact_eval/fusion/run.py \
        --shapes lowk --reps 10 [--free]
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
import triton  # noqa: E402

from artifact import digest, graph_ms, hot_cublas, make_inputs  # noqa: E402
from fused import (EPI_CHAIN, EPI_FP8_OUT, EPI_GATE, EPI_GATE_Q, EPI_NAMES, EPI_SILU, EPI_SILU_Q,  # noqa: E402
                   free_fused, fused_plain, fused_plain_notma, run_epi)

LOG = os.path.join(HERE, "attempts.jsonl")


def log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------------------------
# Shape sets. Each tests a lever from
#     2*bytes/BW + T_launch/(M*N)  >  (1/r - 1) * 2K/F
# left = what fusion saves (one HBM round trip of the M x N intermediate, one launch),
# right = what it pays (our GEMM at 1/r of cuBLAS), which is linear in K.
# --------------------------------------------------------------------------------------------
SHAPES = {
    "lowk": [(8192, 8192, 16), (8192, 8192, 64), (8192, 8192, 256), (8192, 8192, 1024), (8192, 8192, 4096)],
    "lora": [(8192, 4096, 16), (8192, 4096, 64), (16384, 4096, 16), (16384, 4096, 64), (4096, 16384, 16),
             (4096, 16384, 64), (32768, 2048, 16), (32768, 2048, 64)],
    "kscan": [(8192, 8192, k) for k in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)],
    "unaligned": [(8192, 8189, 64), (8191, 8191, 64), (8193, 4095, 64), (8192, 4093, 16), (8189, 8189, 256)],
    "deep": [(8192, 8192, 8192), (4096, 4096, 4096), (1024, 6144, 4096)],
    "head": [(8192, 8192, 16), (16384, 4096, 64)],
    # lever 3: N a multiple of 8 but NOT of 16. cuBLAS declines to optimise there, and the plan
    # is still `plain` with every tensor TMA-able, so the same fused kernel covers it.
    "mis8": [(8192, 8184, 16), (8192, 8184, 64), (8190, 8184, 64), (16384, 4088, 64), (8184, 8184, 256)],
}
EPILOGUES = [EPI_SILU, EPI_GATE, EPI_CHAIN, EPI_SILU_Q, EPI_GATE_Q]

# Bit-neutral launch shapes for the fused arm: (BM, BN, num_warps, num_stages, SUBTILE).
FUSED_SPACE = [(bm, bn, nw, ns, sub)
               for (bm, bn) in ((128, 256), (256, 128), (128, 128), (256, 256), (64, 256), (256, 64), (64, 128))
               for (nw, ns) in ((8, 4), (4, 8), (8, 3)) for sub in (False, True)]
# The non-TMA fallback's launch shapes: (BM, BN, num_warps, num_stages).
NOTMA_SPACE = [(bm, bn, nw, ns) for (bm, bn) in ((128, 256), (256, 128), (128, 128), (64, 128), (128, 64))
               for (nw, ns) in ((8, 3), (8, 4), (4, 4))]
# The baseline epilogue kernel's own launch shape.
EPI_SPACE = [(blk, nw) for blk in (1024, 2048, 4096, 8192) for nw in (4, 8)]
# arm 3.
FREE_SPACE = [(bm, bn, bk, w, s) for (bm, bn) in ((64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128))
              for bk in (16, 32, 64) for w in (4, 8) for s in (3, 4)]


def _tune(cands, build, flush, reps=7):
    """Smallest median device time over `cands`. Returns (time, cand). Skips what will not run.

    A candidate whose launcher returns None ran NOTHING -- `fused_plain` does that when TMA
    cannot carry one of the tensors. Timing it gives a beautiful number for an empty graph, the
    same trap as a `cublasLtMatmul` that silently no-ops, so it has to be dropped here and not
    left for the byte gate to catch."""
    best, bcand = None, None
    for c in cands:
        try:
            fn = build(c)
            if fn is None or fn() is None:
                continue
            torch.cuda.synchronize()
        except Exception:
            continue
        t = graph_ms(torch, fn, flush, reps=reps)
        if t is not None and (best is None or t < best):
            best, bcand = t, c
    return best, bcand


def one(M, N, K, epi, args, flush):
    from bitequiv.cublas_match import cublas_equivalent_gemm, cublas_matmul
    from bitequiv.cublas_match import ltapi as L
    from bitequiv.cublas_match.errors import CublasUnsupportedShape
    from bitequiv.cublas_match.gemm import _resolve

    dt = torch.float16
    seed = M * 1000003 + N * 10007 + K
    rec = {"M": M, "N": N, "K": K, "epi": EPI_NAMES[epi], "dtype": "fp16", "when": time.strftime("%H:%M:%S")}
    a, b = make_inputs(torch, M, N, K, "fp16", 0, seed)
    try:
        rec["mode"] = _resolve(a, b, "fp16", dt).mode
    except CublasUnsupportedShape as e:
        rec["skip"] = f"unsupported: {e}"
        log(rec)
        print(f"  {M}x{N}x{K:<6} {EPI_NAMES[epi]:9s} SKIP {e}")
        return rec

    gen = torch.Generator("cuda")
    g = (torch.randn(M, N, device="cuda", generator=gen.manual_seed(seed + 1)) / 2).to(dt)
    r = (torch.randn(M, N, device="cuda", generator=gen.manual_seed(seed + 2)) / 2).to(dt)
    es = 0.75
    odt = torch.float8_e4m3fn if epi in EPI_FP8_OUT else dt
    obuf = torch.empty(M, N, device="cuda", dtype=odt)

    # ---- arm 1 ------------------------------------------------------------------------------
    h = hot_cublas(L, torch, a, b, "fp16", dt)
    t_gemm_only = graph_ms(torch, h.run, flush)
    t_epi_only, epi_cfg = _tune(EPI_SPACE, lambda c: (lambda: run_epi(h.out, g, r, es, epi, out=obuf, cfg=c)), flush)

    def arm1():
        h.run()
        run_epi(h.out, g, r, es, epi, out=obuf, cfg=epi_cfg)

    t_base = graph_ms(torch, arm1, flush)

    # ---- arm 2: tune the bit-neutral launch shape, then gate the winner on bytes -------------
    # `fused_plain` is the TMA + persistent + warp-specialized kernel, which is what
    # `cublas_equivalent_gemm` runs when every tensor can carry a descriptor. When one cannot --
    # every N that is not a whole number of 16-byte lines, i.e. exactly the non-aligned shapes --
    # bitequiv itself falls back to the plain launcher, so the fused arm falls back with it.
    def build_fused(c):
        if c[0] == "tma":
            return lambda: fused_plain(a, b, dt, epi, g=g, r=r, escale=es, out=obuf, cfg=c[1][:4], SUBTILE=c[1][4])
        return lambda: fused_plain_notma(a, b, dt, epi, g=g, r=r, escale=es, out=obuf, cfg=c[1])

    def run_fused(c, ai, bi, o=None):
        if c[0] == "tma":
            return fused_plain(ai, bi, dt, epi, g=g, r=r, escale=es, out=o, cfg=c[1][:4], SUBTILE=c[1][4])
        return fused_plain_notma(ai, bi, dt, epi, g=g, r=r, escale=es, out=o, cfg=c[1])

    cands = [("tma", c) for c in FUSED_SPACE] + [("plain", c) for c in NOTMA_SPACE]
    t_fused, fcfg = _tune(cands, build_fused, flush)
    if t_fused is None:
        rec["skip"] = "no fused config ran"
        log(rec)
        print(f"  {M}x{N}x{K:<6} {EPI_NAMES[epi]:9s} SKIP no fused config ran")
        return rec

    # The byte gate. `cublas_matmul` and the `hot_cublas` closure both run heuristic result 0
    # under the same workspace allowance, so they are the same cuBLAS call; `hot_agrees` checks
    # that on this shape rather than assuming it, because the timed baseline and the gated
    # baseline have to be the same computation for the gate to mean anything.
    ok, hot_agrees = 0, None
    for i in range(args.reps):
        ai, bi = make_inputs(torch, M, N, K, "fp16", i, seed)
        c_i = cublas_matmul(ai, bi, dt)
        base_i = run_epi(c_i, g, r, es, epi, cfg=epi_cfg)
        fus_i = run_fused(fcfg, ai, bi)
        torch.cuda.synchronize()
        ok += int(digest(torch, base_i) == digest(torch, fus_i))
        if i == 0:
            hot_cublas(L, torch, ai, bi, "fp16", dt)  # warm; the closure runs once on construction
            hh = hot_cublas(L, torch, ai, bi, "fp16", dt)
            hh.run()
            torch.cuda.synchronize()
            hot_agrees = digest(torch, hh.out) == digest(torch, c_i)
            del hh
        del ai, bi, base_i, fus_i, c_i
    rec["hot_cublas_agrees"] = hot_agrees
    rec["bit_ok"], rec["bit_total"] = ok, args.reps
    rec["fused_cfg"], rec["epi_cfg"] = [fcfg[0], list(fcfg[1])], list(epi_cfg)
    if ok != args.reps:
        rec["verdict"] = "NOT BYTE-IDENTICAL"
        rec["baseline_ms"], rec["fused_ms"] = t_base, t_fused
        log(rec)
        print(f"  {M}x{N}x{K:<6} {EPI_NAMES[epi]:9s} FAIL bits {ok}/{args.reps}  (would have been "
              f"{t_base / t_fused:.3f}x -- discarded)")
        torch.cuda.empty_cache()
        return rec

    # ---- how much of the win is the fusion, and how much is our GEMM? ------------------------
    def ours_unfused():
        run_epi(cublas_equivalent_gemm(a, b, dt), g, r, es, epi, out=obuf, cfg=epi_cfg)

    t_ours_unfused = graph_ms(torch, ours_unfused, flush)
    t_ours_gemm = graph_ms(torch, lambda: cublas_equivalent_gemm(a, b, dt), flush)

    # ---- arm 3 -------------------------------------------------------------------------------
    t_free, free_cfg = (None, None)
    if args.free:
        t_free, free_cfg = _tune(FREE_SPACE,
                                 lambda c: (lambda: free_fused(a, b, c, epi, g=g, r=r, escale=es, out=obuf)), flush)

    rec.update({
        "baseline_ms": t_base, "cublas_gemm_ms": t_gemm_only, "epi_kernel_ms": t_epi_only, "fused_ms": t_fused,
        "ours_unfused_ms": t_ours_unfused, "ours_gemm_ms": t_ours_gemm, "free_fused_ms": t_free,
        "free_cfg": list(free_cfg) if free_cfg else None, "speedup": t_base / t_fused, "verdict": "byte-identical"
    })
    log(rec)
    sp = rec["speedup"]
    print(f"  {M}x{N}x{K:<6} {EPI_NAMES[epi]:9s} base {t_base:7.3f} (cublas {t_gemm_only:6.3f} + epi {t_epi_only:6.3f})"
          f"  fused {t_fused:7.3f}  free {('%7.3f' % t_free) if t_free else '     --'}"
          f"  -> {sp:.3f}x {'WIN ' if sp > 1.0 else 'loss'}  bits {ok}/{args.reps}  cfg {fcfg}", flush=True)
    del h, g, r, obuf, a, b
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="lowk")
    ap.add_argument("--epi", default="")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--log", default="attempts.jsonl")
    args = ap.parse_args()
    global LOG
    LOG = os.path.join(HERE, args.log)

    names = [n.strip() for n in args.shapes.split(",")]
    shapes = [s for n in names for s in SHAPES[n]]
    epis = EPILOGUES if not args.epi else [k for k, v in EPI_NAMES.items() if v in args.epi.split(",")]

    print(f"device {torch.cuda.get_device_name(0)}  cap {torch.cuda.get_device_capability()}")
    print(f"shapes {names}  epilogues {[EPI_NAMES[e] for e in epis]}  reps {args.reps}  free-arm {args.free}\n")
    flush = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device="cuda")
    rows = []
    for (M, N, K) in shapes:
        for epi in epis:
            try:
                rows.append(one(M, N, K, epi, args, flush))
            except Exception as e:
                import traceback
                traceback.print_exc()
                log({"M": M, "N": N, "K": K, "epi": EPI_NAMES[epi], "error": f"{type(e).__name__}: {e}"})
            torch.cuda.empty_cache()
    wins = [r for r in rows if r.get("speedup", 0) and r["speedup"] > 1.0 and r.get("verdict") == "byte-identical"]
    print(f"\n  {len(wins)} byte-identical wins out of {len(rows)} pairs")
    for r in sorted(wins, key=lambda r: -r["speedup"])[:15]:
        print(f"    {r['M']}x{r['N']}x{r['K']} {r['epi']}  {r['speedup']:.3f}x")


if __name__ == "__main__":
    sys.exit(main())
