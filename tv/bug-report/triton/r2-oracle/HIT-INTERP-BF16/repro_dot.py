#!/usr/bin/env python3
"""Same defect, shown on `tl.dot` with bfloat16 inputs.

`tl.dot` is the op most bf16 kernels are written for, so it gets its own
reproduction.  Run it the same way as `repro.py`:

    source /home/youngzt/fuzz/env.sh
    CUDA_VISIBLE_DEVICES=0 python \
      /home/youngzt/tv/triton/tv/bug-report/triton/r2-oracle/HIT-INTERP-BF16/repro_dot.py

16x16 identity-like inputs with small whole numbers, fp32 accumulator, so the
right answer is exact and the two arms must agree bit for bit.
"""
import json
import os
import subprocess
import sys

import torch

import triton
import triton.language as tl

M = N = K = 16


@triton.jit
def dot_kernel(a_ptr, b_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    ra = tl.arange(0, M)[:, None]
    rk = tl.arange(0, K)[None, :]
    rk2 = tl.arange(0, K)[:, None]
    rb = tl.arange(0, N)[None, :]
    a = tl.load(a_ptr + ra * K + rk)
    b = tl.load(b_ptr + rk2 * N + rb)
    acc = tl.dot(a, b, input_precision="ieee")
    tl.store(o_ptr + ra * N + rb, acc)


def make_inputs(dtype, dev):
    g = torch.Generator().manual_seed(0)
    a = torch.randint(-3, 4, (M, K), generator=g).to(torch.float32)
    b = torch.randint(-3, 4, (K, N), generator=g).to(torch.float32)
    return a.to(dev).to(dtype), b.to(dev).to(dtype)


def run_child():
    dev = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
    out = {}
    for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        try:
            a, b = make_inputs(dtype, dev)
            o = torch.zeros((M, N), dtype=torch.float32, device=dev)
            dot_kernel[(1, )](a, b, o, M=M, N=N, K=K)
            out[name] = {"ok": True,
                         "bits": o.view(torch.uint8).cpu().numpy().tobytes().hex(),
                         "corner": o[:2, :2].cpu().numpy().tolist()}
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "err": f"{type(e).__name__}: {e}".split(chr(10))[0][:160]}
    print("RESULT " + json.dumps(out))


def main():
    def child(interp):
        e = dict(os.environ)
        e["TRITON_ALWAYS_COMPILE"] = "1"
        e.setdefault("TRITON_CACHE_DIR", "/home/youngzt/fuzz/triton-cache")
        e["_REPRO_CHILD"] = "1"
        if interp:
            e["TRITON_INTERPRET"] = "1"
        else:
            e.pop("TRITON_INTERPRET", None)
        p = subprocess.run([sys.executable, os.path.abspath(__file__)],
                           env=e, capture_output=True, text=True, timeout=900)
        for line in p.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT "):])
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit("child produced no RESULT line")

    interp, gpu = child(True), child(False)
    # exact answer: integer matmul, every entry well below 2**24
    a, b = make_inputs(torch.float32, "cpu")
    ref = (a @ b)

    for name in interp:
        i, g = interp[name], gpu[name]
        same = i.get("bits") == g.get("bits")
        print(f"tl.dot / {name}")
        print(f"   interpreter top-left 2x2 : {i.get('corner', i.get('err'))}")
        print(f"   compiled    top-left 2x2 : {g.get('corner', g.get('err'))}")
        print(f"   correct     top-left 2x2 : {ref[:2, :2].numpy().tolist()}")
        print(f"   interpreter bits == gpu bits : {same}")
        if g["ok"]:
            print("   gpu bits == correct bits     : "
                  f"{g['bits'] == ref.view(torch.uint8).numpy().tobytes().hex()}")
        print()


if __name__ == "__main__":
    if os.environ.get("_REPRO_CHILD") == "1":
        run_child()
    else:
        main()
