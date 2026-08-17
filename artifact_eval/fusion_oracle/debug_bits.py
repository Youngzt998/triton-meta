"""Where does the fused kernel stop agreeing with the unfused eager path?

Four questions, in order, so a mismatch is attributed rather than guessed at:
  1. does torch.mm return cuBLASLt's bytes?
  2. does bitequiv's bit-exact GEMM return torch.mm's bytes?
  3. does OUR mainloop (epilogue off) return torch.mm's bytes?
  4. does our epilogue, fed the eager mm result, return the eager epilogue's bytes?
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from artifact import digest, make_inputs  # noqa: E402
from cases import EPILOGUES, make_epi_args  # noqa: E402
import ours as O  # noqa: E402

M, N, K = (int(x) for x in sys.argv[1:4])
epi = sys.argv[4] if len(sys.argv) > 4 else "silu"
dtype = torch.float16
odt = O.FP8 if epi == "fp8cast" else dtype

from bitequiv.cublas_match import cublas_equivalent_gemm, cublas_matmul  # noqa: E402
from bitequiv.cublas_match.gemm import _resolve  # noqa: E402
from bitequiv.cublas_match.ltapi import _kind_of, cublaslt_version  # noqa: E402

print("cuBLASLt", cublaslt_version())
a, w = make_inputs(torch, M, N, K, "fp16", 0, M * 1000003 + N * 10007 + K)
extra = make_epi_args(epi, M, N, dtype)
plan = _resolve(a, w, _kind_of(a), dtype)
print("plan", plan.mode, "algo", plan.algo_id, "raw", plan.raw_config)

d_torch = digest(torch, torch.mm(a, w))
d_lt = digest(torch, cublas_matmul(a, w, dtype))
d_be = digest(torch, cublas_equivalent_gemm(a, w, dtype))
print("1. torch.mm == cublasLt        ", d_torch == d_lt)
print("2. bitequiv gemm == torch.mm   ", d_be == d_torch, " == cublasLt", d_be == d_lt)

cfgs = [(128, 256, 64, 8, 8, 3, True), (128, 128, 64, 8, 8, 3, True), (128, 128, 64, 8, 8, 3, False),
        (64, 64, 64, 8, 4, 4, False)]
for cfg in cfgs:
    for name, fn in (("tma", O.launch_plain_tma), ("plain", O.launch_plain)):
        try:
            c = fn(a, w, dtype, O.EPI_NONE, None, cfg, mm_dtype=dtype)
        except Exception as e:  # noqa: BLE001
            print(f"3. ours[{name}] {cfg} -> {type(e).__name__}: {str(e)[:100]}")
            continue
        if c is None:
            print(f"3. ours[{name}] {cfg} -> declined")
            continue
        d = digest(torch, c)
        print(f"3. ours[{name}] {cfg} == torch.mm {d == d_torch}  == bitequiv {d == d_be}")

# 4. epilogue only: feed the eager mm result through our epilogue via a 1-step "GEMM"
x = torch.mm(a, w)
want = digest(torch, EPILOGUES[epi](x, *extra))
eye = torch.eye(K, dtype=dtype, device="cuda")
# identity trick is not exact for fp16; instead check the epilogue on the CPU-equivalent path:
# run our kernel with K=0 is not possible, so compare through a tiny pointwise Triton kernel.
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _epi_only(X, Y, R, n, EPI: tl.constexpr, HAS_R: tl.constexpr, RDT: tl.constexpr, BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < n
    acc = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
    r = tl.load(R + off, mask=m, other=0.0).to(tl.float32) if HAS_R else acc
    y = O._epilogue(acc, r, EPI, HAS_R, RDT)
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=m)


xf = x.contiguous().view(-1)
y = torch.empty_like(xf, dtype=odt)
r = extra[0].contiguous().view(-1) if O.EPI_ID[epi] in O.EPI_NEEDS_R else xf
_epi_only[(triton.cdiv(xf.numel(), 1024), )](xf, y, r, xf.numel(), EPI=O.EPI_ID[epi],
                                             HAS_R=O.EPI_ID[epi] in O.EPI_NEEDS_R, RDT=tl.float16, BLK=1024)
got = digest(torch, y.view(M, N))
print(f"4. our epilogue == eager epilogue {got == want}")
if got != want:
    ref = EPILOGUES[epi](x, *extra)
    gv = y.view(M, N)
    if odt == O.FP8:
        bad = (ref.view(torch.uint8) != gv.view(torch.uint8))
    else:
        bad = (ref.view(torch.uint16) != gv.view(torch.uint16))
    idx = bad.nonzero()[:5]
    print("   mismatching elements", int(bad.sum()), "of", ref.numel())
    for i in idx:
        i0, i1 = int(i[0]), int(i[1])
        print(f"   x={float(x[i0, i1])!r} eager={ref[i0, i1]!r} ours={gv[i0, i1]!r}")
