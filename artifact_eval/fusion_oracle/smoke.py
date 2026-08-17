"""One case, end to end, to see the autotune log format and what compile costs."""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch._inductor.config as ic  # noqa: E402

ic.max_autotune_gemm_backends = "ATEN,TRITON"

M, N, K = 4096, 4096, 256
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
w = torch.randn(K, N, device="cuda", dtype=torch.float16)


def f(a, w):
    x = a @ w
    return x * torch.sigmoid(x)


t0 = time.time()
g = torch.compile(f, mode="max-autotune-no-cudagraphs")
out = g(a, w)
torch.cuda.synchronize()
print("compile+run seconds", time.time() - t0, file=sys.stderr)
print("out", out.shape, out.dtype, file=sys.stderr)
