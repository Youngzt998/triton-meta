"""Which Triton spelling of sigmoid returns torch eager's fp16 bytes, and what does each cost?

torch's CUDA sigmoid for half is `1.f / (1.f + ::expf(-x))` evaluated in fp32 and rounded back to
half. Triton's `tl.exp` is the approximate `ex2.approx.f32`, which is off by a couple of ulp --
enough to move the fp16 rounding on a small fraction of inputs.

Two questions, both answered here:

  bits  sweep the WHOLE fp16 domain, not a sample. 63488 finite values is all of it, so for a
        unary function this is a complete check, not evidence.
  cost  count the PTX the spelling generates. The measured price of the exact epilogue is 0.0122
        ms against 0.0040 ms for the approximate one on 4096x4096, and that gap is what decides
        whether a bit-exact fused kernel can beat Inductor's. Correctly rounded division is the
        expensive half: `div.rn.f32` is a Newton iteration, while `rcp.rn.f32` computes the same
        value when the numerator is exactly 1.0, in fewer instructions.
"""
import os
import re
import sys

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from triton.language.extra.cuda import libdevice  # noqa: E402

LOG2E = 1.4426950408889634


@triton.jit
def _sig(x, WHICH: tl.constexpr):
    if WHICH == 0:
        return tl.sigmoid(x)
    elif WHICH == 1:
        return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))
    elif WHICH == 2:
        return libdevice.rcp_rn(1.0 + libdevice.exp(-x))
    elif WHICH == 3:
        return 1.0 / (1.0 + libdevice.exp(-x))
    elif WHICH == 4:
        return 1.0 / (1.0 + tl.exp(-x))
    elif WHICH == 5:
        return libdevice.rcp_rn(1.0 + libdevice.exp2(-x * LOG2E))
    elif WHICH == 6:
        return libdevice.rcp_rn(1.0 + libdevice.fast_expf(-x))
    elif WHICH == 7:
        return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x)) * 1.0
    elif WHICH == 8:
        # one Newton step on the approximate reciprocal. `div.approx.f32` is MUFU.RCP plus a
        # multiply, about 22 good bits; `q + (1 - d*q)*q` doubles that to ~44, far past the 24 fp32
        # carries and unreachably far past the 11 the fp16 store keeps. Three instructions where
        # `div.rn.f32` is a full Newton iteration with a rounding fixup.
        d = 1.0 + libdevice.fast_expf(-x)
        q = libdevice.fast_dividef(1.0, d)
        return libdevice.fma_rn(libdevice.fma_rn(-d, q, 1.0), q, q)
    elif WHICH == 9:
        d = 1.0 + libdevice.exp(-x)
        q = libdevice.fast_dividef(1.0, d)
        return libdevice.fma_rn(libdevice.fma_rn(-d, q, 1.0), q, q)
    elif WHICH == 10:
        return libdevice.fast_dividef(1.0, 1.0 + libdevice.fast_expf(-x))
    elif WHICH == 11:  # one Newton step, in fp32 via tl.fma
        d = 1.0 + libdevice.fast_expf(-x)
        q = libdevice.fast_dividef(1.0, d)
        return tl.fma(tl.fma(-d, q, 1.0), q, q)
    elif WHICH == 12:  # two Newton steps
        d = 1.0 + libdevice.fast_expf(-x)
        q = libdevice.fast_dividef(1.0, d)
        q = tl.fma(tl.fma(-d, q, 1.0), q, q)
        return tl.fma(tl.fma(-d, q, 1.0), q, q)
    elif WHICH == 13:  # one Newton step, accurate exp
        d = 1.0 + libdevice.exp(-x)
        q = libdevice.fast_dividef(1.0, d)
        return tl.fma(tl.fma(-d, q, 1.0), q, q)
    elif WHICH == 14:  # correctly rounded divide, cheap exp -- the one that is bit-exact today
        return libdevice.div_rn(1.0, 1.0 + libdevice.fast_expf(-x))
    elif WHICH == 15:
        # one Newton step, with the saturating end guarded. For x below about -88, expf(-x)
        # overflows to inf, so q is exactly 0 and the correction computes inf*0 = NaN. That, and
        # not accuracy, is the whole of the 9844 mismatches the unguarded version shows -- the
        # count is the number of fp16 values below -88. q is 0 only there, so testing q settles it.
        d = 1.0 + libdevice.fast_expf(-x)
        q = libdevice.fast_dividef(1.0, d)
        return tl.where(q > 0.0, tl.fma(tl.fma(-d, q, 1.0), q, q), q)
    elif WHICH == 16:
        d = 1.0 + libdevice.fast_expf(-x)
        q = libdevice.fast_dividef(1.0, d)
        r = tl.fma(tl.fma(-d, q, 1.0), q, q)
        return tl.where(q > 0.0, tl.fma(tl.fma(-d, r, 1.0), r, r), q)
    elif WHICH == 17:  # guarded Newton on the accurate exp
        d = 1.0 + libdevice.exp(-x)
        q = libdevice.fast_dividef(1.0, d)
        return tl.where(q > 0.0, tl.fma(tl.fma(-d, q, 1.0), q, q), q)
    else:
        return libdevice.div_rn(1.0, 1.0 + libdevice.exp2(-x * LOG2E))


NAMES = {
    0: "tl.sigmoid",
    1: "div_rn(1, 1+libdevice.exp(-x))",
    2: "rcp_rn(1+libdevice.exp(-x))",
    3: "1/(1+libdevice.exp(-x))",
    4: "1/(1+tl.exp(-x))",
    5: "rcp_rn(1+libdevice.exp2(-x*log2e))",
    6: "rcp_rn(1+fast_expf(-x))",
    8: "libdevice.fma newton, fast_expf",
    9: "libdevice.fma newton, exp",
    10: "fast_dividef(1, 1+fast_expf(-x))",
    11: "tl.fma newton x1, fast_expf",
    12: "tl.fma newton x2, fast_expf",
    13: "tl.fma newton x1, libdevice.exp",
    14: "div_rn(1, 1+fast_expf(-x))",
    15: "guarded newton x1, fast_expf",
    16: "guarded newton x2, fast_expf",
    17: "guarded newton x1, libdevice.exp",
}


@triton.jit
def _k(X, Y, n, WHICH: tl.constexpr, SILU: tl.constexpr, BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < n
    x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
    s = _sig(x, WHICH)
    if SILU:
        s = s.to(tl.float16).to(tl.float32)
        y = x * s
    else:
        y = s
    tl.store(Y + off, y.to(tl.float16), mask=m)


bits = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.uint16).view(torch.float16)
x = bits[torch.isfinite(bits.float())].contiguous()
n = x.numel()
print(f"domain: {n} finite fp16 values -- the whole of it, so a 0 below is a proof, not a sample\n")

for silu, ref in ((0, torch.sigmoid(x)), (1, x * torch.sigmoid(x))):
    print("silu = x * sigmoid(x)" if silu else "sigmoid(x)")
    for which in sorted(NAMES):
        y = torch.empty_like(x)
        try:
            _k[(triton.cdiv(n, 1024), )](x, y, n, WHICH=which, SILU=silu, BLK=1024)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {NAMES[which]:38s} ERROR {type(e).__name__}: {str(e)[:80]}")
            continue
        bad = int((y.view(torch.uint16) != ref.view(torch.uint16)).sum())
        print(f"  {NAMES[which]:38s} differs on {bad}/{n}")
    print()

# ------------------------------------------------------------------------------------------
# cost. PTX counts alone mislead: `rcp.rn.f32` is one PTX instruction and a whole Newton
# iteration in SASS, so it is no cheaper than `div.rn.f32` however the PTX reads. Time it.
# ------------------------------------------------------------------------------------------
import measure  # noqa: E402

print("cost of one silu over 16.8M fp16 values, plus the PTX it came from")
print(f"  clock warmed: {measure.warm(torch)}")
big = torch.randn(4096 * 4096, device="cuda", dtype=torch.float16).contiguous()
bigy = torch.empty_like(big)
nb = big.numel()
flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
WANT = ("rcp.approx", "rcp.rn", "div.rn", "div.approx", "div.full", "ex2.approx", "fma.rn.f32", "mul.f32", "add.f32",
        "sub.f32", "cvt.rn.f16", "cvt.f32.f16", "selp", "setp", "mov.f32", "min.f32", "max.f32")
fns, ptxs = {}, {}
for which in sorted(NAMES):
    try:
        k = _k.warmup(torch.float16, torch.float16, 1, WHICH=which, SILU=1, BLK=1024, grid=(1, ))
        ptxs[which] = k.asm["ptx"]
        _k[(triton.cdiv(nb, 1024), )](big, bigy, nb, WHICH=which, SILU=1, BLK=1024)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"  {NAMES[which]:38s} ERROR {str(e)[:60]}")
        continue
    fns[which] = lambda w=which: _k[(triton.cdiv(nb, 1024), )](big, bigy, nb, WHICH=w, SILU=1, BLK=1024)
ts = measure.best_ms(torch, fns, flush, rounds=4, reps=15)
base = torch.empty_like(big)


@triton.jit
def _copy(X, Y, n, BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < n
    tl.store(Y + off, tl.load(X + off, mask=m, other=0.0), mask=m)


t_copy = measure.best_ms(torch, {"copy": lambda: _copy[(triton.cdiv(nb, 1024), )](big, base, nb, BLK=1024)}, flush,
                         rounds=4, reps=15)["copy"]
print(f"  {'plain fp16 copy (the floor)':38s} {t_copy:.4f} ms")
for which in sorted(ts):
    body = [ln.strip() for ln in ptxs[which].splitlines()]
    total = sum(1 for ln in body if re.match(r"^[a-z@].*;$", ln) and not ln.startswith("//"))
    hist = {w: sum(1 for ln in body if ln.startswith(w)) for w in WANT}
    hist = {kk: v for kk, v in hist.items() if v}
    print(f"  {NAMES[which]:38s} {ts[which]:.4f} ms  (math {ts[which] - t_copy:+.4f})  {total:4d} PTX  " +
          " ".join(f"{kk}x{v}" for kk, v in hist.items()))
