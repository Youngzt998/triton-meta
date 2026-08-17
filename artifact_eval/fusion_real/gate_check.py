"""Does the byte gate in `run.py` actually catch a wrong kernel?

A check that cannot fail is not evidence.  This runs the same gate -- same draws, same digest --
against three kernels on real cases:

  correct   the fused kernel the sweep times.                       expect 10/10
  no-round  THE SAME kernel with `ROUND=False`: one line changed, so the epilogue reads the fp32
            accumulator instead of the accumulator rounded to fp16.  That single rounding is the
            whole bit constraint -- what Inductor calls "emulate unfused numerics".
                                                                    expect a failure
  split-acc the same GEMM with TWO fp32 accumulators over the two halves of k, added at the end
            -- a wrong split-K merge, the defect the whole artifact is about.  expect a failure.
            It needs at least two k tiles to be a different computation at all, so on a K = 16
            LoRA shape (one tile of BK = 32) it is reported `n/a` rather than as a pass -- there
            the only perturbation the shape admits is `no-round`, and the gate catches that.

Narrow exponents hide a regrouping, so the draws alternate between ordinary gaussian and
exponents spread across the dtype's range, exactly as the sweep does.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=<repo> .venv/bin/python artifact_eval/fusion_real/gate_check.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from artifact import digest, make_inputs  # noqa: E402
from fused_real import (EPI_CODES, EPI_NEEDS_G, EPI_NEEDS_R, apply_epi, fused_plain,  # noqa: E402
                        out_dtype_for, run_epi)
from run import epi_scale, side_tensor  # noqa: E402
from models import Case  # noqa: E402


@triton.jit
def _split_acc_gemm(A, B, C, G, R, M, N, K, am, ak, bk, bn, cm, cn, gm, gn, s, es, EPI: tl.constexpr,
                    NEED_G: tl.constexpr, NEED_R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    ODT: tl.constexpr):
    """DELIBERATELY WRONG: two fp32 accumulators over the halves of k, added at the end."""
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
    on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
    ok = tl.arange(0, BK).to(tl.int64)
    kt = tl.cdiv(K, BK)
    half = (kt + 1) // 2
    acc0 = tl.zeros((BM, BN), dtype=tl.float32)
    acc1 = tl.zeros((BM, BN), dtype=tl.float32)
    ap = A + om[:, None] * am + ok[None, :] * ak
    bp = B + ok[:, None] * bk + on[None, :] * bn
    for k in range(0, kt):
        m = ok < K - k * BK
        a = tl.load(ap, mask=m[None, :], other=0.0)
        b = tl.load(bp, mask=m[:, None], other=0.0)
        if k < half:
            acc0 = tl.dot(a, b, acc0)
        else:
            acc1 = tl.dot(a, b, acc1)
        ap += BK * ak
        bp += BK * bk
    acc = (acc0 + acc1) * s
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    msk = (ocm[:, None] < M) & (ocn[None, :] < N)
    off = gm * ocm[:, None] + gn * ocn[None, :]
    x = acc.to(ODT).to(tl.float32)
    g = tl.load(G + off, mask=msk, other=0.0).to(tl.float32) if NEED_G else x
    r = tl.load(R + off, mask=msk, other=0.0).to(tl.float32) if NEED_R else x
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], apply_epi(x, g, r, es, EPI).to(C.dtype.element_ty), mask=msk)


def split_acc(a, b, epi, g=None, r=None, escale=1.0, BM=128, BN=128, BK=32):
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty(M, N, device="cuda", dtype=out_dtype_for(epi, a.dtype))
    ref = g if g is not None else (r if r is not None else out)
    _split_acc_gemm[(triton.cdiv(M, BM) * triton.cdiv(N, BN), )](a, b, out, g if g is not None else out,
                                                                 r if r is not None else out, M, N, K, a.stride(0),
                                                                 a.stride(1), b.stride(0), b.stride(1), out.stride(0),
                                                                 out.stride(1), ref.stride(0), ref.stride(1), 1.0,
                                                                 escale, EPI=epi, NEED_G=epi in EPI_NEEDS_G, NEED_R=epi
                                                                 in EPI_NEEDS_R, BM=BM, BN=BN, BK=BK, ODT=tl.float16,
                                                                 num_warps=8)
    return out


CASES = [
    Case("Llama-3-8B", "q_proj.lora_B r=16", "lora", 4096, 4096, 16, "lora"),
    Case("Llama-3-8B", "up_proj.lora_B r=64", "lora", 8192, 14336, 64, "lora"),
    Case("Llama-3-8B", "o_proj", "fp8q", 4096, 4096, 4096, "fp8q"),
    Case("Llama-3-8B", "up_proj", "swiglu", 4096, 14336, 4096, "swiglu"),
    Case("Llama-3-8B", "o_proj", "resid", 4096, 4096, 4096, "resid"),
]
REPS = 10


def main():
    from bitequiv.cublas_match import cublas_matmul

    print(f"{'case':52s} {'correct':>9s} {'no-round':>9s} {'split-acc':>10s}")
    print("-" * 84)
    bad = 0
    for case in CASES:
        epi = EPI_CODES[case.epi]
        M, N, K = case.M, case.N, case.K
        seed = M * 1000003 + N * 10007 + K * 101 + epi
        es = epi_scale(case)
        counts = [0, 0, 0]
        splittable = -(-K // 32) >= 2  # two accumulators need two k tiles to differ at all
        for i in range(REPS):
            a, b = make_inputs(torch, M, N, K, "fp16", i, seed)
            g = side_tensor(M, N, i, seed) if epi in EPI_NEEDS_G else None
            r = side_tensor(M, N, i, seed) if epi in EPI_NEEDS_R else None
            ref = digest(torch, run_epi(cublas_matmul(a, b, torch.float16), g, r, es, epi))
            got = [
                fused_plain(a, b, torch.float16, epi, g=g, r=r, escale=es, cfg=(128, 256, 8, 4), SUBTILE=False),
                fused_plain(a, b, torch.float16, epi, g=g, r=r, escale=es, cfg=(128, 256, 8, 4), SUBTILE=False,
                            ROUND=False),
                split_acc(a, b, epi, g=g, r=r, escale=es) if splittable else None,
            ]
            torch.cuda.synchronize()
            for j, t in enumerate(got):
                counts[j] += int(t is not None and digest(torch, t) == ref)
            del a, b, g, r, got
            torch.cuda.empty_cache()
        sa = f"{counts[2]:6d}/{REPS:<3d}" if splittable else f"{'n/a':>10s}"
        print(f"{case.label():52s} {counts[0]:5d}/{REPS:<3d} {counts[1]:5d}/{REPS:<3d} {sa}")
        if counts[0] != REPS:
            bad += 1
            print("   ^^ the CORRECT kernel did not match: the gate is reporting a real problem")
        if counts[1] == REPS or (splittable and counts[2] == REPS):
            bad += 1
            print("   ^^ a DELIBERATELY WRONG kernel passed: the gate is not sensitive on this case")
    print(f"\n{'gate is sensitive on every case' if bad == 0 else f'{bad} problems -- see above'}")


if __name__ == "__main__":
    main()
