"""gemm.bitmatch -- does the Triton GEMM return cuBLAS's bytes, and when it does not, whose
fault is it?

Moved out of `artifact.py` unchanged. The committed results (67,306 shapes) were produced by
this code with this shape draw and this seed, so a change to either invalidates them.
"""
from __future__ import annotations

import json
import random
import time

from ._common import digest, draw_shape, make_inputs, writer

NAME = "gemm.bitmatch"
ORDER = 10
DESCRIPTION = ("random shapes, is the Triton GEMM byte-identical to cuBLAS, "
               "and when not, is cuBLAS itself wrong")
IMPLEMENTED = True

TABLES = {
    "gemm.bitmatch": {
        "doc":
        "One row per random shape. Is the Triton GEMM byte-identical to cuBLAS on that "
        "shape, over `reps` independent input draws (even draws ordinary gaussian, odd "
        "draws with the exponents spread across the dtype range).",
        "cols": [
            ("M", "int", "rows of A"),
            ("N", "int", "columns of B"),
            ("K", "int", "contraction length"),
            ("dtype", "str", "operand dtype: fp16 or fp8 (e4m3)"),
            ("mode", "str", "which cuBLAS kernel family the plan resolved to"),
            ("reps", "int", "independent input draws compared on this shape"),
            ("n_differ", "int", "draws whose output differed from cuBLAS; 0 means byte-identical"),
            ("draws_that_differ", "str", "space-separated draw indices that differed, empty if none"),
            ("declined", "str", "non-empty if the shape is out of scope and no comparison was made"),
            ("error", "str", "non-empty if the shape failed to run"),
        ],
    },
}


def run(args, env):
    import torch

    from bitequiv.cublas_match import cublas_equivalent_gemm, cublas_matmul
    from bitequiv.cublas_match.errors import CublasUnsupportedShape

    out = writer("gemm.bitmatch")
    rng = random.Random(args.seed)
    deadline = time.time() + args.minutes * 60
    n = ok = declined = mism = 0
    print(f"\n[gemm.bitmatch] {args.minutes} min, {args.reps} input draws per shape, seed {args.seed}")
    while time.time() < deadline:
        M, N, K, kind = draw_shape(rng)
        esz = 1 if kind == "fp8" else 2
        if (M * K + K * N + M * N) * esz > args.max_bytes:
            continue
        seed = M * 1000003 + N * 10007 + K
        rec = {"M": M, "N": N, "K": K, "dtype": kind, "reps": args.reps}
        try:
            diffs = []
            for r in range(args.reps):
                a, b = make_inputs(torch, M, N, K, kind, r, seed)
                if r == 0:
                    from bitequiv.cublas_match.gemm import _resolve
                    from bitequiv.cublas_match.ltapi import _kind_of
                    rec["mode"] = _resolve(a, b, _kind_of(a), torch.float16).mode
                same = digest(torch,
                              cublas_equivalent_gemm(a,
                                                     b, torch.float16)) == digest(torch,
                                                                                  cublas_matmul(a, b, torch.float16))
                if not same:
                    diffs.append(r)
                del a, b
                torch.cuda.empty_cache()
            rec["n_differ"] = len(diffs)
            rec["draws_that_differ"] = diffs
            if diffs:
                mism += 1
                print(
                    f"  MISMATCH  {kind} {M}x{N}x{K} mode={rec.get('mode')}  "
                    f"{len(diffs)}/{args.reps} draws differ", flush=True)
            else:
                ok += 1
            n += 1
        except CublasUnsupportedShape as e:
            declined += 1
            rec["declined"] = str(e)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        out.write(json.dumps(rec) + "\n")
        out.flush()
        torch.cuda.empty_cache()
    out.close()
    print(f"\n  {n} shapes x {args.reps} draws = {n * args.reps} comparisons")
    print(f"  byte-identical to cuBLAS      {ok}/{n}")
    print(f"  shapes with any differing draw {mism}   (logged in data/gemm.bitmatch.jsonl)")
    print(f"  declined (out of scope)       {declined}")
    if mism:
        print("  A mismatch is not automatically ours: cuBLAS itself drops the k tail on some\n"
              "  shapes -- run gemm.cublas-bug, which reproduces that defect on this machine.")
