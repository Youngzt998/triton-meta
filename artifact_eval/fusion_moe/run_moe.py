"""Does a fused, byte-identical Triton kernel beat the two-kernel cuBLAS path on shapes and
operator pairings that come from real models?

Three arms per case, same shape, same inputs, device time from CUDA-graph replay with L2 flushed
between replays:

  arm 1  eager unfused   cuBLAS through a hot closure, then the epilogue as ONE Triton kernel.
                         One, not several, because Inductor fuses a pointwise chain into a
                         single kernel even when it cannot fuse into the GEMM.  The epilogue
                         kernel's block size and warp count are tuned, so the baseline is not
                         losing to a launch-shape accident.
  arm 2  ours            the bit-exact GEMM with the epilogue folded in.  The launch shape is
                         tuned over bit-neutral knobs and the winner is then put through the
                         byte gate.  Not byte-identical means the row is a failure, not a
                         speedup.
  arm 3  unconstrained   measured two ways, because "unconstrained" has two readings.
                         `free_fused_ms` is the same fusion on an ordinary autotuned Triton
                         matmul -- no TMA, no warp specialization, not persistent -- which is the
                         shape of the template torch.compile emits, so it is what max-autotune
                         gives a user today.  It is NOT an upper bound: on this hardware that
                         structure is slower than the bit-exact kernel.  `noround_ms` is OUR
                         kernel with one line changed, the epilogue reading the fp32 accumulator
                         instead of the accumulator rounded to fp16.  That single rounding is the
                         entire bit constraint, so `fused_ms / noround_ms` is what the constraint
                         costs with the kernel structure held fixed.

Every case names a model and a layer (see `models_2026.py`) and carries the epilogue that
actually follows that GEMM in that model.  This copy runs the 2026 open-weight models, whose MoE
expert GEMMs are narrower than a dense FFN and see only the tokens routed to one expert -- both
the shape and the token count move, so the boundary the previous run found has to be measured
again rather than carried over.

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/youngzt/bitwise-equiv/triton \
        .venv/bin/python artifact_eval/fusion_moe/run_moe.py --group moe --reps 10
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
from fused_moe import (EPI_ARGMAX, EPI_CODES, EPI_FP8_OUT, EPI_NEEDS_G, EPI_NEEDS_R,  # noqa: E402
                       EPI_NEEDS_W, free_argmax, free_fused, fused_argmax, fused_plain, fused_plain_notma,
                       out_dtype_for, run_argmax, run_epi)
from models_2026 import Case, build  # noqa: E402

LOG = os.path.join(HERE, "attempts.jsonl")

# Bit-neutral launch shapes for the fused arm: (BM, BN, num_warps, num_stages, SUBTILE).
_TILES = ((128, 256), (256, 128), (128, 128), (256, 256), (64, 256), (64, 128), (256, 64), (128, 64))
_TILES_SMALL = ((64, 256), (64, 128), (128, 256), (128, 128))
_WS = ((8, 4), (4, 8), (8, 3))


def fused_space(small):
    return [(bm, bn, nw, ns, sub)
            for (bm, bn) in (_TILES_SMALL if small else _TILES)
            for (nw, ns) in _WS
            for sub in (False, True)]


NOTMA_SPACE = [(bm, bn, nw, ns)
               for (bm, bn) in ((128, 256), (256, 128), (128, 128), (64, 128), (128, 64))
               for (nw, ns) in ((8, 3), (8, 4), (4, 4))]
# Two block sizes smaller than the previous run's, because an MoE expert sees only a few hundred
# tokens and the standalone epilogue kernel would otherwise be launch-starved.  Widening arm 1's
# tuning space can only make the BASELINE faster, so it is the conservative direction.
EPI_SPACE = [(blk, nw) for blk in (256, 512, 1024, 2048, 4096, 8192) for nw in (4, 8)]
ARGMAX_SPACE = [(blk, nw) for blk in (1024, 2048, 4096, 8192, 16384) for nw in (4, 8, 16)]
FREE_SPACE = [(bm, bn, bk, w, s)
              for (bm, bn) in ((64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128))
              for bk in (32, 64)
              for w in (4, 8)
              for s in (3, 4)]

# The scalar in the epilogue.  Only its existence matters for the timing; the values are the ones
# the workload really uses.
LORA_ALPHA = 16.0
FP8_STATIC_SCALE = 2.0


def log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def done_keys():
    if not os.path.exists(LOG):
        return set()
    out = set()
    for line in open(LOG):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "key" in r:
            out.add(tuple(r["key"]))
    return out


def stable_ms(fn, flush, rounds=3, reps=25):
    """The two arms in the ratio are measured against each other, so a 2% wobble from another
    process on the card is the difference between a win and a loss.  `graph_ms` already takes a
    median; this takes the smallest of several of those, which is the statistic that is robust
    to interference (interference can only add time)."""
    best = None
    for _ in range(rounds):
        t = graph_ms(torch, fn, flush, reps=reps)
        if t is not None and (best is None or t < best):
            best = t
    return best


def _tune(cands, build_fn, flush, reps=7):
    """Smallest median device time over `cands`.  A candidate whose launcher returns None ran
    NOTHING -- `fused_plain` does that when TMA cannot carry a tensor.  Timing it gives a
    beautiful number for an empty graph, the same trap as a `cublasLtMatmul` that silently
    no-ops, so it is dropped here rather than left for the byte gate."""
    best, bcand = None, None
    for c in cands:
        try:
            fn = build_fn(c)
            if fn is None or fn() is None:
                continue
            torch.cuda.synchronize()
        except Exception:
            continue
        t = graph_ms(torch, fn, flush, reps=reps)
        if t is not None and (best is None or t < best):
            best, bcand = t, c
    return best, bcand


def side_tensor(M, N, rep, seed, dt=torch.float16):
    """The second M x N tensor an epilogue reads -- the frozen layer's output, the gate
    projection, or the residual stream.  Same two distributions as `make_inputs`: even draws
    ordinary gaussian, odd draws with the exponents spread across the dtype's range, because
    narrow exponents hide a change in the order of the additions."""
    g = torch.Generator(device="cuda").manual_seed(seed + 7919 * rep + 13)
    if rep % 2 == 1:
        sign = torch.where(torch.rand(M, N, generator=g, device="cuda") < 0.5, -1.0, 1.0)
        return (sign * torch.exp2(torch.rand(M, N, generator=g, device="cuda") * 20.0 - 12.0)).to(dt)
    return (torch.randn(M, N, generator=g, device="cuda") / 2).to(dt)


def weight_vector(M, rep, seed):
    """The MoE routing weight: one fp32 scalar per token, the value the expert's output is
    multiplied by before the experts are summed.  Even draws use the range a normalised routing
    weight actually lives in, (0, 1); odd draws spread the exponents across the range, because a
    narrow multiplier hides a change in where the rounding happens.  Both arms read the same
    vector, so the values do not move the byte comparison -- only the rounding they provoke."""
    g = torch.Generator(device="cuda").manual_seed(seed + 104729 * rep + 31)
    if rep % 2 == 1:
        sign = torch.where(torch.rand(M, generator=g, device="cuda") < 0.5, -1.0, 1.0)
        return sign * torch.exp2(torch.rand(M, generator=g, device="cuda") * 20.0 - 12.0)
    return torch.rand(M, generator=g, device="cuda", dtype=torch.float32)


def epi_scale(case):
    if case.epi == "lora":
        return LORA_ALPHA / case.K
    if case.epi == "fp8q":
        return FP8_STATIC_SCALE
    return 1.0


# ------------------------------------------------------------------------------------------- #


def one_pointwise(case, epi, args, flush, rec):
    """The four pointwise epilogues: lora, fp8q, swiglu, resid."""
    from bitequiv.cublas_match import cublas_equivalent_gemm, cublas_matmul
    from bitequiv.cublas_match import ltapi as L

    M, N, K = case.M, case.N, case.K
    dt = torch.float16
    seed = M * 1000003 + N * 10007 + K * 101 + epi
    es = epi_scale(case)
    lim = case.lim
    a, b = make_inputs(torch, M, N, K, "fp16", 0, seed)
    g = side_tensor(M, N, 0, seed) if epi in EPI_NEEDS_G else None
    r = side_tensor(M, N, 0, seed) if epi in EPI_NEEDS_R else None
    w = weight_vector(M, 0, seed) if epi in EPI_NEEDS_W else None
    obuf = torch.empty(M, N, device="cuda", dtype=out_dtype_for(epi, dt))

    # ---- arm 1 -------------------------------------------------------------------------------
    h = hot_cublas(L, torch, a, b, "fp16", dt)
    t_gemm_only = graph_ms(torch, h.run, flush)
    t_epi_only, epi_cfg = _tune(EPI_SPACE, lambda c:
                                (lambda: run_epi(h.out, g, r, es, epi, out=obuf, cfg=c, w=w, lim=lim)), flush)

    def arm1():
        h.run()
        run_epi(h.out, g, r, es, epi, out=obuf, cfg=epi_cfg, w=w, lim=lim)

    t_base = stable_ms(arm1, flush)

    # ---- arm 2 -------------------------------------------------------------------------------
    def build_fused(c):
        if c[0] == "tma":
            return lambda: fused_plain(a, b, dt, epi, g=g, r=r, escale=es, out=obuf, cfg=c[1][:4], SUBTILE=c[1][4], w=w,
                                       lim=lim)
        return lambda: fused_plain_notma(a, b, dt, epi, g=g, r=r, escale=es, out=obuf, cfg=c[1], w=w, lim=lim)

    def run_fused(c, ai, bi, gi, ri, wi, o=None):
        if c[0] == "tma":
            return fused_plain(ai, bi, dt, epi, g=gi, r=ri, escale=es, out=o, cfg=c[1][:4], SUBTILE=c[1][4], w=wi,
                               lim=lim)
        return fused_plain_notma(ai, bi, dt, epi, g=gi, r=ri, escale=es, out=o, cfg=c[1], w=wi, lim=lim)

    small = M * N <= 2**23
    cands = [("tma", c) for c in fused_space(small)] + [("plain", c) for c in NOTMA_SPACE]
    _, fcfg = _tune(cands, build_fused, flush)
    if fcfg is None:
        rec["skip"] = "no fused config ran"
        return rec
    t_fused = stable_ms(build_fused(fcfg), flush)
    rec["fused_cfg"], rec["epi_cfg"], rec["tma"] = [fcfg[0], list(fcfg[1])], list(epi_cfg), fcfg[0] == "tma"

    # ---- the byte gate -----------------------------------------------------------------------
    ok, hot_agrees = 0, None
    for i in range(args.reps):
        ai, bi = make_inputs(torch, M, N, K, "fp16", i, seed)
        gi = side_tensor(M, N, i, seed) if epi in EPI_NEEDS_G else None
        ri = side_tensor(M, N, i, seed) if epi in EPI_NEEDS_R else None
        wi = weight_vector(M, i, seed) if epi in EPI_NEEDS_W else None
        c_i = cublas_matmul(ai, bi, dt)
        base_i = run_epi(c_i, gi, ri, es, epi, cfg=epi_cfg, w=wi, lim=lim)
        fus_i = run_fused(fcfg, ai, bi, gi, ri, wi)
        torch.cuda.synchronize()
        ok += int(digest(torch, base_i) == digest(torch, fus_i))
        if i == 0:
            hh = hot_cublas(L, torch, ai, bi, "fp16", dt)
            hh.run()
            torch.cuda.synchronize()
            hot_agrees = digest(torch, hh.out) == digest(torch, c_i)
            del hh
        del ai, bi, gi, ri, wi, base_i, fus_i, c_i
    rec["hot_cublas_agrees"], rec["bit_ok"], rec["bit_total"] = hot_agrees, ok, args.reps
    if ok != args.reps:
        rec.update({"verdict": "NOT BYTE-IDENTICAL", "baseline_ms": t_base, "fused_ms": t_fused})
        return rec

    # ---- how much of the win is the fusion and how much is our GEMM? -------------------------
    t_ours_unfused = graph_ms(
        torch, lambda: run_epi(cublas_equivalent_gemm(a, b, dt), g, r, es, epi, out=obuf, cfg=epi_cfg, w=w, lim=lim),
        flush)
    t_ours_gemm = graph_ms(torch, lambda: cublas_equivalent_gemm(a, b, dt), flush)

    # ---- arm 3 -------------------------------------------------------------------------------
    # Two readings of "unconstrained", because they answer different questions.
    #   free     an ordinary autotuned Triton matmul with the fusion: no TMA, no warp
    #            specialization, not persistent.  That is the shape of the template
    #            torch.compile emits, so it is what a user gets from max-autotune today.
    #   noround  OUR kernel with one line changed: the epilogue reads the fp32 accumulator
    #            instead of the accumulator rounded to fp16.  That single rounding is the whole
    #            bit constraint, so ours/noround is what the constraint costs, with the kernel
    #            structure held fixed.
    t_free, free_cfg = _tune(FREE_SPACE, lambda c:
                             (lambda: free_fused(a, b, c, epi, g=g, r=r, escale=es, out=obuf, w=w, lim=lim)),
                             flush) if args.free else (None, None)

    def build_noround(c):
        if c[0] == "tma":
            return lambda: fused_plain(a, b, dt, epi, g=g, r=r, escale=es, out=obuf, cfg=c[1][:4], SUBTILE=c[1][4],
                                       ROUND=False, w=w, lim=lim)
        return lambda: fused_plain_notma(a, b, dt, epi, g=g, r=r, escale=es, out=obuf, cfg=c[1], ROUND=False, w=w, lim=
                                         lim)

    _, nr_cfg = _tune(cands, build_noround, flush)
    t_noround = stable_ms(build_noround(nr_cfg), flush) if nr_cfg else None
    rec["noround_ms"] = t_noround
    rec["noround_cfg"] = [nr_cfg[0], list(nr_cfg[1])] if nr_cfg else None

    rec.update({
        "baseline_ms": t_base, "cublas_gemm_ms": t_gemm_only, "epi_kernel_ms": t_epi_only, "fused_ms": t_fused,
        "ours_unfused_ms": t_ours_unfused, "ours_gemm_ms": t_ours_gemm, "free_fused_ms": t_free, "free_cfg":
        list(free_cfg) if free_cfg else None, "speedup": t_base / t_fused, "verdict": "byte-identical"
    })
    del h, g, r, w, obuf, a, b
    return rec


def one_argmax(case, args, flush, rec):
    """lm_head then greedy decode.  arm 1 is cuBLAS plus one row-wise argmax kernel; the faster of
    a tuned Triton kernel and `torch.argmax` is used, so the baseline is not a straw man."""
    from bitequiv.cublas_match import cublas_equivalent_gemm, cublas_matmul
    from bitequiv.cublas_match import ltapi as L

    M, N, K = case.M, case.N, case.K
    dt = torch.float16
    seed = M * 1000003 + N * 10007 + K * 101 + 991
    a, b = make_inputs(torch, M, N, K, "fp16", 0, seed)
    ov = torch.empty(M, device="cuda", dtype=torch.float32)
    oi = torch.empty(M, device="cuda", dtype=torch.int32)
    # Per-column-tile winners.  Sized for the smallest BN in either tuning space so the timed
    # call allocates nothing inside the graph capture.
    tmax = triton.cdiv(N, 64)
    bufs = (torch.empty(tmax * M, device="cuda",
                        dtype=torch.float32), torch.empty(tmax * M, device="cuda", dtype=torch.int32), ov, oi)

    h = hot_cublas(L, torch, a, b, "fp16", dt)
    t_gemm_only = graph_ms(torch, h.run, flush)
    t_tri, epi_cfg = _tune(ARGMAX_SPACE, lambda c: (lambda: run_argmax(h.out, ov, oi, cfg=c)), flush)
    t_torch = graph_ms(torch, lambda: torch.argmax(h.out, dim=1), flush)
    t_epi_only = min(x for x in (t_tri, t_torch) if x is not None)
    rec["argmax_epi_triton_ms"], rec["argmax_epi_torch_ms"] = t_tri, t_torch
    if epi_cfg is None:
        rec["skip"] = "the Triton argmax kernel would not run"
        return rec

    def arm1():
        h.run()
        run_argmax(h.out, ov, oi, cfg=epi_cfg)

    def arm1_torch():
        h.run()
        torch.argmax(h.out, dim=1)

    t_base = min(x for x in (stable_ms(arm1, flush), stable_ms(arm1_torch, flush)) if x is not None)

    def build_fused(c):
        return lambda: fused_argmax(a, b, dt, cfg=c[:4], bufs=bufs)

    _, fcfg = _tune([c for c in fused_space(True) if not c[4]], build_fused, flush)
    if fcfg is None:
        rec["skip"] = "no fused config ran (TMA refused)"
        return rec
    t_fused = stable_ms(build_fused(fcfg), flush)
    rec["fused_cfg"], rec["epi_cfg"], rec["tma"] = ["tma", list(fcfg)], list(epi_cfg), True

    ok = 0
    for i in range(args.reps):
        ai, bi = make_inputs(torch, M, N, K, "fp16", i, seed)
        c_i = cublas_matmul(ai, bi, dt)
        bv, bidx = run_argmax(c_i, cfg=epi_cfg)
        fv, fi = fused_argmax(ai, bi, dt, cfg=fcfg[:4], bufs=None)
        torch.cuda.synchronize()
        ok += int(digest(torch, bv) == digest(torch, fv) and digest(torch, bidx) == digest(torch, fi))
        if i == 0:
            rec["torch_argmax_agrees"] = bool((bidx.long() == torch.argmax(c_i, dim=1)).all().item())
        del ai, bi, c_i, bv, bidx, fv, fi
    rec["bit_ok"], rec["bit_total"] = ok, args.reps
    if ok != args.reps:
        rec.update({"verdict": "NOT BYTE-IDENTICAL", "baseline_ms": t_base, "fused_ms": t_fused})
        return rec

    t_ours_unfused = graph_ms(torch, lambda: run_argmax(cublas_equivalent_gemm(a, b, dt), ov, oi, cfg=epi_cfg), flush)
    t_ours_gemm = graph_ms(torch, lambda: cublas_equivalent_gemm(a, b, dt), flush)
    t_free, free_cfg = _tune(FREE_SPACE, lambda c:
                             (lambda: free_argmax(a, b, c, bufs=bufs)), flush) if args.free else (None, None)
    rec.update({
        "baseline_ms": t_base, "cublas_gemm_ms": t_gemm_only, "epi_kernel_ms": t_epi_only, "fused_ms": t_fused,
        "ours_unfused_ms": t_ours_unfused, "ours_gemm_ms": t_ours_gemm, "free_fused_ms": t_free, "free_cfg":
        list(free_cfg) if free_cfg else None, "speedup": t_base / t_fused, "verdict": "byte-identical"
    })
    del h, a, b, bufs
    return rec


def one(case: Case, args, flush):
    from bitequiv.cublas_match.errors import CublasUnsupportedShape
    from bitequiv.cublas_match.gemm import _resolve

    epi = EPI_CODES[case.epi]
    rec = {
        "key": list(case.key), "model": case.model, "layer": case.layer, "pairing": case.pairing, "M": case.M, "N":
        case.N, "K": case.K, "epi": case.epi, "note": case.note, "lim": case.lim, "dtype": "fp16", "when":
        time.strftime("%H:%M:%S")
    }
    a, b = make_inputs(torch, case.M, case.N, case.K, "fp16", 0, 1)
    try:
        rec["mode"] = _resolve(a, b, "fp16", torch.float16).mode
    except CublasUnsupportedShape as e:
        rec["skip"] = f"unsupported: {e}"
        del a, b
        return rec
    del a, b
    if rec["mode"] != "plain":
        # The fused kernel reproduces `plain` only.  A shape cuBLAS routes elsewhere is a failure
        # of coverage, not a slow result -- report it as such.
        rec["verdict"] = f"NOT REPRODUCIBLE: cuBLAS plan mode is {rec['mode']}, not plain"
        return rec
    if epi == EPI_ARGMAX:
        return one_argmax(case, args, flush, rec)
    return one_pointwise(case, epi, args, flush, rec)


def show(rec):
    head = (f"  {rec['model']:22s} {rec['layer']:30s} {rec['M']:>5}x{rec['N']:<6}x{rec['K']:<6} "
            f"{rec['epi']:9s}")
    if "skip" in rec:
        print(head + f" SKIP {rec['skip']}", flush=True)
    elif rec.get("verdict", "").startswith("NOT REPRODUCIBLE"):
        print(head + f" {rec['verdict']}", flush=True)
    elif rec.get("verdict") == "NOT BYTE-IDENTICAL":
        print(
            head + f" FAIL bits {rec['bit_ok']}/{rec['bit_total']} (would have been "
            f"{rec['baseline_ms'] / rec['fused_ms']:.3f}x -- discarded)", flush=True)
    elif "speedup" in rec:
        fr = f"{rec['free_fused_ms']:7.3f}" if rec.get("free_fused_ms") else "     --"
        print(
            head + f" base {rec['baseline_ms']:8.3f} (gemm {rec['cublas_gemm_ms']:7.3f} + epi "
            f"{rec['epi_kernel_ms']:7.3f})  ours {rec['fused_ms']:8.3f}  free {fr}  -> "
            f"{rec['speedup']:.3f}x {'WIN ' if rec['speedup'] > 1.0 else 'loss'}  bits "
            f"{rec['bit_ok']}/{rec['bit_total']}", flush=True)
    else:
        print(head + f" {rec}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="moe", help="comma-separated: moe,lora,lmhead,attn")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-free", dest="free", action="store_false")
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--only", default="", help="substring filter on the case label")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true", help="re-run cases already in the log")
    args = ap.parse_args()
    global LOG
    LOG = os.path.join(HERE, args.log)

    cases = build([n.strip() for n in args.group.split(",")])
    if args.only:
        cases = [c for c in cases if args.only in c.label()]
    if not args.redo:
        seen = done_keys()
        cases = [c for c in cases if tuple(c.key) not in seen]
    if args.limit:
        cases = cases[:args.limit]

    print(f"device {torch.cuda.get_device_name(0)}  cap {torch.cuda.get_device_capability()}")
    print(f"{len(cases)} cases to run  reps {args.reps}  free-arm {args.free}  log {LOG}\n", flush=True)
    flush = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device="cuda")
    t0 = time.time()
    for n, c in enumerate(cases):
        try:
            rec = one(c, args, flush)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec = {
                "key": list(c.key), "model": c.model, "layer": c.layer, "pairing": c.pairing, "M": c.M, "N": c.N, "K":
                c.K, "epi": c.epi, "error": f"{type(e).__name__}: {e}"
            }
        rec["elapsed_s"] = round(time.time() - t0, 1)
        log(rec)
        show(rec)
        torch.cuda.empty_cache()
        if n % 10 == 9:
            print(f"    [{n + 1}/{len(cases)}  {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
