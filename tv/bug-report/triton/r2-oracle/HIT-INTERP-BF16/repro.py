#!/usr/bin/env python3
"""Triton interpreter mode does bf16 / fp8 arithmetic on the raw bit patterns.

Run it exactly like this (no arguments):

    source /home/youngzt/fuzz/env.sh
    CUDA_VISIBLE_DEVICES=0 python \
      /home/youngzt/tv/triton/tv/bug-report/triton/r2-oracle/HIT-INTERP-BF16/repro.py

The script re-launches itself twice as a child process: once with
`TRITON_INTERPRET=1` (Triton's own Python/numpy interpreter, no GPU) and once
without (the ordinary compiled kernel on the GPU).  It then prints the two
answers side by side.  Two child runs are needed because `TRITON_INTERPRET` is
read when `@triton.jit` runs, so one process cannot do both.

A third column, "correct", is the same computation done with torch on the CPU
in float32 and rounded back to the kernel's dtype.  With these inputs (whole
numbers, -4..4) no operation rounds, so that column is the exact answer both
arms must produce.  It is there because for float8 the compiled arm refuses to
compile at all, so there is no GPU number to compare against.

Expected output: for every float32 and float16 case all three columns agree;
for the bfloat16 `add`, `mul`, `neg`, `sum` and `cmp` cases the interpreter
disagrees with both the GPU and the correct answer, and raises nothing; for the
float8 cases the compiled arm raises `RuntimeError: PassManager::run failed`
while the interpreter returns a wrong answer with no error.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# the kernels
# --------------------------------------------------------------------------- #
@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(out_ptr + i, tl.load(a_ptr + i) + tl.load(b_ptr + i))


@triton.jit
def mul_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(out_ptr + i, tl.load(a_ptr + i) * tl.load(b_ptr + i))


@triton.jit
def max_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(out_ptr + i, tl.maximum(tl.load(a_ptr + i), tl.load(b_ptr + i)))


@triton.jit
def neg_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(out_ptr + i, -tl.load(a_ptr + i))


@triton.jit
def sum_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(out_ptr + i, tl.sum(tl.load(a_ptr + i)) + tl.load(b_ptr + i) * 0)


@triton.jit
def cmp_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    # store 1.0 / 0.0 so the result type does not depend on the input dtype
    i = tl.arange(0, N)
    lt = tl.load(a_ptr + i) < tl.load(b_ptr + i)
    tl.store(out_ptr + i, tl.where(lt, 1, 0))


@triton.jit
def upcast_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    # the *correct* way to write it: cast to fp32, add, cast back
    i = tl.arange(0, N)
    s = tl.load(a_ptr + i).to(tl.float32) + tl.load(b_ptr + i).to(tl.float32)
    tl.store(out_ptr + i, s.to(out_ptr.dtype.element_ty))


KERNELS = {
    "add": add_kernel,
    "mul": mul_kernel,
    "max": max_kernel,
    "neg": neg_kernel,
    "sum": sum_kernel,
    "cmp": cmp_kernel,
    "upcast": upcast_kernel,
}

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8e5": torch.float8_e5m2,
    "float8e4nv": torch.float8_e4m3fn,
}

# (kernel, dtype) pairs to try
CASES = [
    ("add", "float32"), ("add", "float16"), ("add", "bfloat16"),
    ("add", "float8e5"), ("add", "float8e4nv"),
    ("mul", "bfloat16"),
    ("max", "bfloat16"), ("max", "float8e5"),
    ("neg", "bfloat16"),
    ("sum", "bfloat16"),
    ("cmp", "bfloat16"), ("cmp", "float8e5"),
    ("upcast", "bfloat16"),
]

N = 8
# small whole numbers: -4 -3 -2 -1 0 1 2 3   and   -3 -2 -1 0 1 2 3 4
A = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
B = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]


def run_child():
    """Run every case in this process and print one JSON line."""
    out = {}
    dev = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
    for kname, dname in CASES:
        tdt = DTYPES[dname]
        try:
            a = torch.tensor(A, dtype=torch.float32, device=dev).to(tdt)
            b = torch.tensor(B, dtype=torch.float32, device=dev).to(tdt)
            odt = torch.float32 if kname == "cmp" else tdt
            o = torch.zeros(N, dtype=odt, device=dev)
            KERNELS[kname][(1, )](a, b, o, N=N)
            # report the exact bits, and the value read back as float32
            bits = o.view(torch.uint8).cpu().numpy().tobytes().hex()
            vals = o.to(torch.float32).cpu().numpy().tolist()
            out[f"{kname}/{dname}"] = {"ok": True, "bits": bits, "vals": vals}
        except Exception as e:  # noqa: BLE001 - we want the message, whatever it is
            msg = f"{type(e).__name__}: {e}"
            out[f"{kname}/{dname}"] = {"ok": False, "err": msg.split(chr(10))[0][:160]}
    print("RESULT " + json.dumps(out))


def fmt(v):
    if v is None:
        return "-"
    return "%.6g" % v


def reference():
    """The exact answer: compute in float32 on the CPU, round to the dtype."""
    out = {}
    for kname, dname in CASES:
        tdt = DTYPES[dname]
        a32 = torch.tensor(A, dtype=torch.float32)
        b32 = torch.tensor(B, dtype=torch.float32)
        # go through the storage format, exactly as the kernel sees the inputs
        a32 = a32.to(tdt).to(torch.float32)
        b32 = b32.to(tdt).to(torch.float32)
        if kname in ("add", "upcast"):
            r = a32 + b32
        elif kname == "mul":
            r = a32 * b32
        elif kname == "max":
            r = torch.maximum(a32, b32)
        elif kname == "neg":
            r = -a32
        elif kname == "sum":
            r = torch.full_like(a32, a32.sum())
        elif kname == "cmp":
            r = (a32 < b32).to(torch.float32)
        else:
            raise AssertionError(kname)
        odt = torch.float32 if kname == "cmp" else tdt
        r = r.to(odt)
        out[f"{kname}/{dname}"] = {
            "bits": r.view(torch.uint8).numpy().tobytes().hex(),
            "vals": r.to(torch.float32).numpy().tolist(),
        }
    return out


def main():
    env = dict(os.environ)
    env["TRITON_ALWAYS_COMPILE"] = "1"
    env.setdefault("TRITON_CACHE_DIR", "/home/youngzt/fuzz/triton-cache")

    def child(interp):
        e = dict(env)
        if interp:
            e["TRITON_INTERPRET"] = "1"
        else:
            e.pop("TRITON_INTERPRET", None)
        e["_REPRO_CHILD"] = "1"
        p = subprocess.run([sys.executable, os.path.abspath(__file__)],
                           env=e, capture_output=True, text=True, timeout=900)
        for line in p.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT "):])
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit("child produced no RESULT line")

    interp = child(True)
    gpu = child(False)
    ref = reference()

    print(f"inputs  a = {A}")
    print(f"        b = {B}")
    print()
    hdr = (f"{'case':20s} {'interpreter (TRITON_INTERPRET=1)':40s} "
           f"{'compiled (GPU)':40s} {'correct':30s} interp==correct")
    print(hdr)
    print("-" * len(hdr))
    nbad = 0
    for k in interp:
        i, g, r = interp[k], gpu[k], ref[k]
        si = ", ".join(fmt(v) for v in i["vals"]) if i["ok"] else i["err"]
        sg = ", ".join(fmt(v) for v in g["vals"]) if g["ok"] else g["err"]
        sr = ", ".join(fmt(v) for v in r["vals"])
        ok = i["ok"] and i["bits"] == r["bits"]
        if not ok:
            nbad += 1
        print(f"{k:20s} {si[:40]:40s} {sg[:40]:40s} {sr[:30]:30s} {ok}")
        if i["ok"] and not ok:
            print(f"{'':20s}   bits interp={i['bits']}  correct={r['bits']}")
            if g["ok"]:
                print(f"{'':20s}   bits gpu   ={g['bits']}  "
                      f"gpu==correct: {g['bits'] == r['bits']}")
    print()
    print(f"{nbad} of {len(interp)} cases: the interpreter's answer is wrong")


if __name__ == "__main__":
    if os.environ.get("_REPRO_CHILD") == "1":
        run_child()
    else:
        main()
