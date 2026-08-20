"""Check every `inductor_kernel.EPI_SRC` spelling against torch eager over the WHOLE fp16 domain.

Arm 3 is arm 1's generated kernel with the `tmp<n> = ...` suffix lines swapped for an `EPI_SRC`
entry, so those lines are the only thing that decides whether arm 3 is byte-identical. They are
short unary chains over one fp32 value that the first `.to(fp16).to(fp32)` pins to a fp16 grid, so
the input domain is the 65536 fp16 bit patterns and nothing else -- small enough to enumerate
instead of sample. A 12-draw check inside `three_way.py` can only say "no difference on these
draws"; this says "no difference anywhere", which is a different claim.

The suffix is exercised in the same shape `patch_epilogue` produces it: `acc` in, `tmp_out` stored
through the output pointer's own cast.

An epilogue that reads a SECOND operand has a domain of 65536^2 and is not checked here --
`probe_pairs.py` enumerates all 4,294,967,296 of those pairs. This file keeps the x-only ones and
the second check `probe_pairs.py` does not do: the same complete sweep over `ours.py::_epilogue`,
the older transcription of the same arithmetic.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<repo> .venv/bin/python probe_epi_src.py
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import inductor_kernel as IK  # noqa: E402
from cases import (EPILOGUES, EPI_OPERANDS, base_key, must_be_exact, out_dtype_for,  # noqa: E402
                   params_for)

FP8 = torch.float8_e4m3fn

PROBE = '''
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers  # noqa: F401
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math  # noqa: F401


@triton.jit
def probe(in_ptr, out_ptr, n, BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    mask = off < n
    acc = tl.load(in_ptr + off, mask=mask, other=0.0)
{body}
    tl.store(out_ptr + off, tmp_out, mask)
'''


def build(key):
    params = params_for(key)
    lines = [ln.format(**params) if "{" in ln else ln for ln in IK.EPI_SRC[key]]
    body = "\n".join("    " + ln for ln in lines)
    src = PROBE.format(body=body)
    import hashlib
    import importlib.util
    cache = os.path.join(HERE, "cache")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"probe_{key}_{hashlib.sha1(src.encode()).hexdigest()[:10]}.py")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(src)
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.probe


def full_fp16_domain():
    """All 65536 bit patterns, widened to fp32 -- which is exactly what `acc` holds after the mm."""
    bits = torch.arange(0, 65536, dtype=torch.int32, device="cuda").to(torch.int16)
    x16 = bits.view(torch.float16)
    return x16, x16.to(torch.float32)


def run_one(key, epi, x16, x32):
    odt = out_dtype_for(epi)
    fn = build(key)
    n = x32.numel()
    got = torch.empty(n, device="cuda", dtype=odt)
    fn[(1, )](x32, got, n, BLK=65536, enable_fp_fusion=IK.NO_FP_FUSION)
    torch.cuda.synchronize()
    ref = EPILOGUES[epi](x16, **params_for(epi)).to(odt)
    gb = got.view(torch.uint8)
    rb = ref.contiguous().view(torch.uint8)
    differ = gb != rb
    if odt is torch.float16:
        differ = differ.view(-1, 2).any(dim=1)
    both_nan = torch.isnan(got.to(torch.float32)) & torch.isnan(ref.to(torch.float32))
    return int(differ.sum()), int((differ & ~both_nan).sum()), n


def run_ours(epi, x16):
    """The same complete check for `ours.py::_epilogue`, which is arm 2 -- the bit reference.

    Arm 2 is what arm 3 has to match, so a NaN hole here would silently redefine the target rather
    than fail anything. It gets the same 65536-value sweep, not the 12 draws `three_way.py` runs.
    """
    import ours as O
    odt = O.FP8 if epi == "fp8cast" else torch.float16
    got = O.launch_epi(x16, odt, O.EPI_ID[epi], None)
    torch.cuda.synchronize()
    ref = EPILOGUES[epi](x16).to(odt)
    differ = got.view(torch.uint8) != ref.contiguous().view(torch.uint8)
    if odt is torch.float16:
        differ = differ.view(-1, 2).any(dim=1)
    both_nan = torch.isnan(got.to(torch.float32)) & torch.isnan(ref.to(torch.float32))
    return int(differ.sum()), int((differ & ~both_nan).sum()), x16.numel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keys", default="")
    args = p.parse_args()
    x16, x32 = full_fp16_domain()
    keys = args.keys.split(",") if args.keys else sorted(IK.EPI_SRC)
    print(f"{'EPI_SRC key':<20} {'epilogue':<12} {'differ':>8} {'differ!nan':>11} {'of':>7}  verdict")
    bad = 0
    for key in keys:
        epi = base_key(key)
        if epi == "none" or epi not in EPILOGUES:
            print(f"{key:<20} {'-':<12} {'':>8} {'':>11} {'':>7}  skipped (no torch reference)")
            continue
        if EPI_OPERANDS.get(epi):
            print(f"{key:<20} {epi:<12} {'':>8} {'':>11} {'':>7}  skipped (reads an extra operand: "
                  f"probe_pairs.py)")
            continue
        try:
            d, dn, n = run_one(key, epi, x16, x32)
        except Exception as e:  # noqa: BLE001
            print(f"{key:<20} {epi:<12} {'':>8} {'':>11} {'':>7}  ERROR {type(e).__name__}: {str(e)[:90]}")
            bad += 1
            continue
        want_exact = must_be_exact(key)
        ok = (dn == 0) if want_exact else True
        print(f"{key:<20} {epi:<12} {d:>8} {dn:>11} {n:>7}  "
              f"{'EXACT' if dn == 0 else 'differs'}{'' if ok else '   <-- FAILS, must be exact'}")
        bad += int(not ok)

    import ours as O
    print(f"\narm 2 -- ours.py::_epilogue, the bit reference\n"
          f"{'epilogue':<20} {'':<12} {'differ':>8} {'differ!nan':>11} {'of':>7}  verdict")
    for epi in sorted(set(O.EPI_ID) & set(EPILOGUES)):
        if EPI_OPERANDS.get(epi):
            print(f"{epi:<20} {'':<12} {'':>8} {'':>11} {'':>7}  skipped (binary: the sweep is over "
                  f"one operand's domain)")
            continue
        try:
            d, dn, n = run_ours(epi, x16)
        except Exception as e:  # noqa: BLE001
            print(f"{epi:<20} {'':<12} {'':>8} {'':>11} {'':>7}  ERROR {type(e).__name__}: {str(e)[:90]}")
            bad += 1
            continue
        print(f"{epi:<20} {'':<12} {d:>8} {dn:>11} {n:>7}  "
              f"{'EXACT' if dn == 0 else 'differs   <-- FAILS, must be exact'}")
        bad += int(dn != 0)

    print(f"\n{bad} spelling(s) that must be exact are not")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
