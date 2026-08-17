"""Step 1 -- let PyTorch tell us where fusing a GEMM epilogue already pays.

For every (shape, epilogue) we compile the SAME module twice and time both the same way:

  aten    `max_autotune_gemm_backends="ATEN"`   -- extern cuBLAS mm, then Inductor's own
                                                   pointwise kernel for the whole epilogue chain.
                                                   This is the two-kernel path a user gets today.
  triton  `max_autotune_gemm_backends="TRITON"` -- Inductor's mm template with the epilogue fused
                                                   into it, autotuned on this machine.

`margin = aten_ms / triton_ms`. Above 1.0 the fused Triton template wins, and that is exactly the
comparison `max_autotune_gemm_backends="ATEN,TRITON"` makes internally (`benchmark_epilogue_fusion`
is on under max-autotune), so running both arms ourselves gives the same verdict plus the losing
margins, which the log of the merged run would not show.

Resumable: every finished case is appended to results.jsonl and skipped on the next run.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=<repo> .venv/bin/python step1.py --groups ksweep,lora
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = os.path.join(HERE, "results.jsonl")
ATTEMPTS = os.path.join(HERE, "attempts.jsonl")

EPI_SETS = {
    "core": ["silu", "gate", "chain", "fp8cast", "rowsum"],
    "wide": ["silu", "gate", "resscale", "chain", "fp8cast", "rowsum", "softcap", "mul3", "bias_relu", "gelu"],
}


def case_id(group, M, N, K, epi, dtype):
    return f"{group}|{M}x{N}x{K}|{epi}|{dtype}"


def log(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def done_ids():
    if not os.path.exists(RESULTS):
        return set()
    out = set()
    with open(RESULTS) as f:
        for line in f:
            try:
                out.add(json.loads(line)["id"])
            except Exception:
                pass
    return out


# ------------------------------------------------------------------------------------------
# worker: one shape, every epilogue asked for
# ------------------------------------------------------------------------------------------


def worker(group, M, N, K, epis, dtype_name):
    import torch
    import torch._dynamo
    import torch._inductor.config as ic
    from torch._inductor.utils import run_and_get_code

    from artifact import graph_ms
    from cases import CUBLAS_CAN_FUSE, EPILOGUES, make_epi_args

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    torch.manual_seed(M * 7 + N * 13 + K)
    a = (torch.randn(M, K, device="cuda") / 8).to(dtype)
    w = (torch.randn(K, N, device="cuda") / 8).to(dtype)
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")

    for epi in epis:
        rec = {
            "id": case_id(group, M, N, K, epi, dtype_name), "group": group, "M": M, "N": N, "K": K, "epi": epi,
            "dtype": dtype_name, "cublas_can_fuse": epi in CUBLAS_CAN_FUSE, "when": time.strftime("%H:%M:%S")
        }
        t0 = time.time()
        try:
            fn = EPILOGUES[epi]
            extra = make_epi_args(epi, M, N, dtype)

            def mod(x, y, *rest):
                return fn(x @ y, *rest)

            arms = {}
            for arm, backends in (("aten", "ATEN"), ("triton", "TRITON")):
                torch._dynamo.reset()
                with ic.patch(max_autotune_gemm_backends=backends):
                    g = torch.compile(mod, mode="max-autotune-no-cudagraphs")
                    out, codes = run_and_get_code(g, a, w, *extra)
                    torch.cuda.synchronize()
                    src = "\n".join(codes)
                    arms[arm] = {
                        "ms": graph_ms(torch, lambda g=g: g(a, w, *extra), flush),
                        "tem": src.count("triton_tem_fused"),
                        "poi": src.count("triton_poi_fused") + src.count("triton_red_fused") +
                        src.count("triton_per_fused"),
                        "extern": src.count("extern_kernels."),
                    }
                    del out, codes, src, g
                torch.cuda.empty_cache()
            for arm in ("aten", "triton"):
                for k, v in arms[arm].items():
                    rec[f"{arm}_{k}"] = v
            ta, tt = rec["aten_ms"], rec["triton_ms"]
            rec["margin"] = (ta / tt) if (ta and tt) else None
            rec["fused"] = rec["triton_tem"] > 0 and rec["triton_poi"] == 0
            rec["ok"] = True
        except Exception as e:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["compile_s"] = round(time.time() - t0, 1)
        log(RESULTS, rec)
        log(ATTEMPTS, {"step": 1, **rec})
        m = rec.get("margin")
        print(f"  {epi:10s} aten {rec.get('aten_ms')} triton {rec.get('triton_ms')} "
              f"margin {m if m is None else round(m, 3)} fused={rec.get('fused')} "
              f"{rec.get('error','')}", flush=True)
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", default="ksweep,lora,ksweep_big,thinm,mlp,lmhead,decode")
    p.add_argument("--epis", default="core")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--max-bytes", type=int, default=12 * 2**30)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--worker", default="")  # internal: "group,M,N,K,dtype,epi1+epi2"
    args = p.parse_args()

    if args.worker:
        group, M, N, K, dtype_name, epis = args.worker.split(",")
        worker(group, int(M), int(N), int(K), epis.split("+"), dtype_name)
        return 0

    from cases import SHAPES
    epis = EPI_SETS.get(args.epis, args.epis.split(","))
    have = done_ids()
    todo = []
    for group in args.groups.split(","):
        for (M, N, K) in SHAPES.get(group, []):
            want = [e for e in epis if case_id(group, M, N, K, e, args.dtype) not in have]
            if not want:
                continue
            esz = 2
            if (M * K + K * N + 3 * M * N) * esz > args.max_bytes:
                print(f"skip {M}x{N}x{K}: too big")
                continue
            todo.append((group, M, N, K, want))
    print(f"{len(todo)} shapes to do, {sum(len(t[4]) for t in todo)} cases", flush=True)
    for i, (group, M, N, K, want) in enumerate(todo):
        print(f"[{i + 1}/{len(todo)}] {group} {M}x{N}x{K} {' '.join(want)}", flush=True)
        spec = f"{group},{M},{N},{K},{args.dtype},{'+'.join(want)}"
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", spec]
        env = dict(os.environ, PYTHONPATH=ROOT)
        try:
            subprocess.run(cmd, env=env, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("  worker timed out", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
