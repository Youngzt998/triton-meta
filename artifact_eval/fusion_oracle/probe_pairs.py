"""Check the extra-operand `EPI_SRC` spellings against torch eager over their COMPLETE domain.

Why this file exists
--------------------
`probe_epi_src.py` checks the epilogues that read nothing but the accumulator. Those are unary
chains over one fp32 value that the leading `.to(fp16).to(fp32)` pins to the fp16 grid, so their
whole input domain is 65536 patterns and "0 of 65536 differ" is a complete proof.

An epilogue that reads a second operand does not have that luxury: the domain is 65536^2 =
4,294,967,296 pairs. That is too many to sample honestly and -- it turns out -- few enough to
enumerate. A pair sweep is one Triton kernel and three or four torch kernels over 2^24 elements,
repeated 256 times; it costs seconds, not hours. So the claim here is not "no difference on a
large sample", it is "no difference anywhere in the domain", with the count printed.

What each mode covers
---------------------
    --unary    every unary piece and every x-only spelling: all 65536 fp16 patterns.
    --pairs    every one-extra-operand epilogue: all 2^32 (accumulator, operand) pairs.
    --triples  `swiglu_cw`, which reads two extra operands. 2^48 is out of reach, so it is
               covered by pinning one operand over a set that includes NaN, +-inf, +-0, both
               subnormal ends, both normal ends and the clamp boundary, and enumerating the other
               two -- 2^32 pairs per pinned value, in both directions.

               The composition closes the rest. `swiglu_cw` is `w * t` where
               `t = round16(clamp(x) * silu(clamp(g)))`. `t` is materialised as an fp16 value in
               BOTH paths (eager writes the tensor; the fused form rounds it in a register), so
               the outer step is a multiply of two fp16 values -- exactly the function `rweight`
               is, and `--pairs` enumerates all 2^32 of its (value, weight) pairs. The pinned-w
               runs then check the two halves end to end on the same kernel.

Why the fp16 grid is the whole domain for the accumulator
---------------------------------------------------------
Every spelling claimed exact begins `tmp_x = acc.to(tl.float16).to(tl.float32)` and mentions
`acc` nowhere else, so the output depends on the fp32 accumulator only through its rounding to
fp16. Sweeping the 65536 fp16 patterns therefore covers every fp32 accumulator there is. That is
checked mechanically below (`_check_acc_use`) rather than trusted, because it is the one step
that turns a finite sweep into a statement about all inputs.

Two NaNs count as equal
-----------------------
An fp16 NaN's payload is not pinned by anything in this path -- `cvt.rn.f16.f32` of a NaN and
torch's own store need not agree on which quiet NaN comes out -- so a NaN against a NaN is
reported in the `differ` column and excluded from `differ!nan`, and it is `differ!nan` that must
be zero. A NaN against a number is a failure. This is the same convention `probe_epi_src.py` uses.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=<repo> .venv/bin/python probe_pairs.py --unary --pairs
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import inductor_kernel as IK  # noqa: E402
from cases import EPILOGUES, EPI_OPERANDS, base_key, must_be_exact, params_for  # noqa: E402

FP8 = torch.float8_e4m3fn


def build(key, operands, params):
    """The epilogue as one standalone kernel -- the SAME builder arm 1 uses.

    Nothing is transcribed here. `inductor_kernel.standalone` compiles the shipped `EPI_SRC` text
    with the extra operands loaded and named `tmp_e<i>`, exactly as `patch_epilogue` names them
    inside Inductor's own kernel, so the text this file sweeps is the text both arms run.
    """
    fn, _kind = IK.standalone(key, operands, params, os.path.join(HERE, "cache"), force_flat=True)
    return fn


def _check_acc_use(key, params):
    """The sweep is only complete if the chain reads the accumulator once, through a round to the
    output dtype. Anything else and the fp16 grid is not the whole domain."""
    body = IK.EPI_SRC[key]
    if params:
        body = [ln.format(**params) if "{" in ln else ln for ln in body]
    uses = [ln for ln in body if re.search(r"\bacc\b", ln)]
    if len(uses) != 1:
        return f"reads `acc` on {len(uses)} lines"
    if uses[0].strip() != "tmp_x = acc" + IK._ROUND:
        return f"reads `acc` as {uses[0].strip()!r}, not through a round to the output dtype"
    return None


# ------------------------------------------------------------------------------------------
# the domains
# ------------------------------------------------------------------------------------------

# The fp16 patterns a pinned operand is walked over: both zeros, both infinities, a quiet and a
# signalling NaN of each sign, both ends of the subnormal range, both ends of the normal range,
# the identities, and the clamp boundaries the model epilogues use (+-10 for DeepSeek-V4's
# swiglu_limit, +-448 for the fp8 range) with one ulp either side of each.
SPECIAL = (
    0x0000, 0x8000,  # +-0
    0x0001, 0x8001,  # +-smallest subnormal
    0x03FF, 0x83FF,  # +-largest subnormal
    0x0400, 0x8400,  # +-smallest normal
    0x7BFF, 0xFBFF,  # +-largest normal, 65504
    0x7C00, 0xFC00,  # +-inf
    0x7E00, 0xFE00,  # +-quiet NaN
    0x7C01, 0xFDFF,  # +-NaN with another payload
    0x3C00, 0xBC00,  # +-1
    0x4900, 0xC900,  # +-10, DeepSeek-V4's swiglu_limit
    0x48FF, 0x4901,  # one ulp either side of +10
    0x5F00, 0xDF00,  # +-448, the fp8 e4m3 range
    0x5EFF, 0x5F01,  # one ulp either side of +448
    0x3800, 0x4000,  # 0.5, 2
)


def bits_to_f16(b):
    """int32 bit patterns -> the fp16 values they name. The narrowing wraps, which is what puts
    0x8000..0xFFFF on the negative half."""
    return b.to(torch.int16).view(torch.float16)


def const_f16(bit, n):
    """A whole array of one fp16 bit pattern, for a pinned operand."""
    signed = bit - 65536 if bit >= 32768 else bit
    return torch.full((n, ), signed, device="cuda", dtype=torch.int16).view(torch.float16)


def _mismatch_mask(got, ref):
    """Where the bytes differ, and where they differ for a reason other than two NaNs."""
    d = (got.view(torch.uint8) != ref.contiguous().view(torch.uint8))
    if got.element_size() == 2:
        d = d.view(-1, 2).any(dim=1)
    both_nan = torch.isnan(got.to(torch.float32)) & torch.isnan(ref.to(torch.float32))
    return d, d & ~both_nan


def run_sweep(key, epi, params, axes, pins, chunk_log2, out_dtype, progress=0):
    """Enumerate the product of the swept axes, exactly -- no sampling anywhere in here.

    `axes` lists the slots to sweep: 0 is the accumulator, 1.. are the extra operands in
    `EPI_OPERANDS` order. `pins` fixes every other slot at one fp16 bit pattern. Two axes is all
    2^32 pairs, walked in chunks of 2^chunk_log2.

    The operands are flat here even where the real kernel broadcasts an [M,1] column along N. A
    broadcast changes which element meets which, never what the arithmetic does to a pair, and
    the sweep covers every pair.
    """
    operands = EPI_OPERANDS.get(epi, ())
    n_extra = len(operands)
    fn = build(key, operands, params)
    total_log2 = 16 * len(axes)
    n_chunks = max(1, 1 << max(0, total_log2 - chunk_log2))
    per = 1 << min(total_log2, chunk_log2)
    d_all = dn_all = 0
    first = None
    for c in range(n_chunks):
        base = c * per
        idx = torch.arange(base, base + per, device="cuda", dtype=torch.int64)
        slots = {slot: const_f16(bit, per) for slot, bit in pins.items()}
        for j, slot in enumerate(axes):  # the last axis varies fastest
            shift = 16 * (len(axes) - 1 - j)
            slots[slot] = bits_to_f16(((idx >> shift) & 0xFFFF).to(torch.int32))
        del idx
        x16 = slots[0]
        extras16 = [slots[1 + i] for i in range(n_extra)]
        got = torch.empty(per, device="cuda", dtype=out_dtype)
        # `enable_fp_fusion=False` is not a probe convenience: it is the launch option arm 3 uses,
        # and without it the compiler contracts the rounding away. See `inductor_kernel.NO_FP_FUSION`.
        fn[(max(1, per // 1024), )](x16.to(torch.float32), *extras16, got, per, BLK=1024,
                                    enable_fp_fusion=IK.NO_FP_FUSION)
        ref = EPILOGUES[epi](x16, *extras16, **params).to(out_dtype)
        torch.cuda.synchronize()
        d, dn = _mismatch_mask(got, ref)
        nd, ndn = int(d.sum()), int(dn.sum())
        if ndn and first is None:
            i = int(dn.nonzero()[0])
            first = {
                "acc": float(x16[i]), "extras": [float(e[i]) for e in extras16], "got": float(got[i].to(torch.float32)),
                "want": float(ref[i].to(torch.float32))
            }
        d_all += nd
        dn_all += ndn
        del slots, x16, got, ref, extras16, d, dn
        if progress and c % progress == 0:
            print(f"      chunk {c + 1}/{n_chunks}  differ!nan so far {dn_all}", flush=True)
    return d_all, dn_all, n_chunks * per, first


# ------------------------------------------------------------------------------------------


def out_dtype_of(epi):
    return FP8 if base_key(epi) in ("fp8cast", "fp8q") else torch.float16


def report(key, epi, d, dn, n, first, want_exact, t):
    verdict = "EXACT" if dn == 0 else "differs"
    if want_exact and dn:
        verdict += "   <-- FAILS, must be exact"
    print(f"  {key:20s} {d:>12} {dn:>12} {n:>14}  {t:6.1f}s  {verdict}", flush=True)
    if first:
        print(
            f"      first miss: acc={first['acc']!r} extras={first['extras']!r} "
            f"got {first['got']!r} want {first['want']!r}", flush=True)
    return int(bool(want_exact and dn))


def header():
    print(f"  {'EPI_SRC key':20s} {'differ':>12} {'differ!nan':>12} {'of':>14}  {'time':>7}  verdict", flush=True)


def keys_for(epi):
    """Every spelling of one epilogue, and whether it has to be exact."""
    out = []
    for key in sorted(IK.EPI_SRC):
        if base_key(key) != epi:
            continue
        out.append((key, must_be_exact(key)))
    return out


def param_variants(epi, args):
    """The scalar values to sweep an epilogue at.

    The values the case list actually uses come first, then values chosen to be awkward. This
    matters: `LORA_ALPHA / rank` is 2, 1, 0.5 or 0.25 for every rank in the case list, all powers
    of two, so `x * scale` is EXACT and a sweep over only those never exercises the rounding the
    scaled delta is supposed to have. The same trap sits under `fp8q`, whose static scale is 2.0.
    """
    if epi == "lora":
        return [{"scale": float(s)} for s in args.lora_scales.split(",")]
    if epi == "fp8q":
        return [{"scale": float(s)} for s in args.fp8q_scales.split(",")]
    if epi == "swiglu_cw":
        return [{"lim": float(s)} for s in args.lims.split(",")]
    return [{}]


def label_of(key, params):
    return key if not params else key + " " + " ".join(f"{k}={v:g}" for k, v in sorted(params.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unary", action="store_true", help="all 65536 fp16 values, x-only epilogues")
    p.add_argument("--pairs", action="store_true", help="all 2^32 pairs, one-extra-operand epilogues")
    p.add_argument("--triples", action="store_true", help="swiglu_cw, one operand pinned")
    p.add_argument("--epi", default="", help="comma-separated subset")
    p.add_argument("--chunk-log2", type=int, default=24)
    p.add_argument("--pins", type=int, default=len(SPECIAL), help="how many of SPECIAL to pin over")
    p.add_argument(
        "--lora-scales", default="2.0,1.0,0.5,0.25,0.1,0.3333333333333333",
        help="alpha/rank values to sweep lora at; the first four are the ones the "
        "case list uses, the rest are not representable in fp32 or fp16")
    p.add_argument("--fp8q-scales", default="2.0,0.375,0.1")
    p.add_argument("--lims", default="10.0", help="swiglu_limit values; 10.0 is DeepSeek-V4's")
    p.add_argument("--progress", type=int, default=0)
    args = p.parse_args()
    if not (args.unary or args.pairs or args.triples):
        args.unary = args.pairs = args.triples = True
    only = set(args.epi.split(",")) if args.epi else None
    bad = 0

    unary = [e for e in EPILOGUES if not EPI_OPERANDS.get(e) and e != "rowsum"]
    one = [e for e in EPILOGUES if len(EPI_OPERANDS.get(e, ())) == 1]
    two = [e for e in EPILOGUES if len(EPI_OPERANDS.get(e, ())) == 2]

    if args.unary:
        print("\n== every fp16 accumulator, no extra operand: 65536 values, exhaustive ==")
        header()
        for epi in sorted(unary):
            if only and epi not in only:
                continue
            for over in param_variants(epi, args):
                for key, want in keys_for(epi):
                    params = params_for(epi, over)
                    why = _check_acc_use(key, params) if want else None
                    if why:
                        print(f"  {key:20s} {'':>12} {'':>12} {'':>14} {'':>7}  NOT SWEEPABLE: {why}")
                        bad += 1
                        continue
                    t0 = time.time()
                    try:
                        d, dn, n, first = run_sweep(key, epi, params, [0], {}, args.chunk_log2, out_dtype_of(epi))
                    except Exception as e:  # noqa: BLE001
                        print(f"  {key:20s} ERROR {type(e).__name__}: {str(e)[:110]}")
                        bad += 1
                        continue
                    bad += report(label_of(key, over), epi, d, dn, n, first, want, time.time() - t0)

    if args.pairs:
        print("\n== every (accumulator, operand) pair: 4,294,967,296 pairs, exhaustive ==")
        header()
        for epi in sorted(one):
            if only and epi not in only:
                continue
            for over in param_variants(epi, args):
                for key, want in keys_for(epi):
                    params = params_for(epi, over)
                    why = _check_acc_use(key, params) if want else None
                    if why:
                        print(f"  {key:20s} {'':>12} {'':>12} {'':>14} {'':>7}  NOT SWEEPABLE: {why}")
                        bad += 1
                        continue
                    label = label_of(key, over)
                    t0 = time.time()
                    try:
                        d, dn, n, first = run_sweep(key, epi, params, [0, 1], {}, args.chunk_log2, out_dtype_of(epi),
                                                    args.progress)
                    except Exception as e:  # noqa: BLE001
                        print(f"  {label:20s} ERROR {type(e).__name__}: {str(e)[:110]}")
                        bad += 1
                        continue
                    bad += report(label, epi, d, dn, n, first, want, time.time() - t0)

    if args.triples:
        pins = SPECIAL[:args.pins]
        print(f"\n== two extra operands: all 2^32 pairs on two axes, the third pinned over "
              f"{len(pins)} values ==")
        header()
        for epi in sorted(two):
            if only and epi not in only:
                continue
            over = param_variants(epi, args)[0]
            params = params_for(epi, over)
            for key, want in keys_for(epi):
                why = _check_acc_use(key, params) if want else None
                if why:
                    print(f"  {key:20s} NOT SWEEPABLE: {why}")
                    bad += 1
                    continue
                for pinned_slot in (2, 1):
                    swept = [0, 1 if pinned_slot == 2 else 2]
                    tot_d = tot_dn = tot_n = 0
                    first = None
                    t0 = time.time()
                    try:
                        for b in pins:
                            d, dn, n, f0 = run_sweep(key, epi, params, swept, {pinned_slot: b}, args.chunk_log2,
                                                     out_dtype_of(epi))
                            tot_d += d
                            tot_dn += dn
                            tot_n += n
                            first = first or f0
                    except Exception as e:  # noqa: BLE001
                        print(f"  {key:20s} ERROR {type(e).__name__}: {str(e)[:110]}")
                        bad += 1
                        continue
                    kind = EPI_OPERANDS[epi][pinned_slot - 1]
                    bad += report(f"{key} pin e{pinned_slot - 1}:{kind}", epi, tot_d, tot_dn, tot_n, first, want,
                                  time.time() - t0)

    print(f"\n{bad} spelling(s) that must be exact are not")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
