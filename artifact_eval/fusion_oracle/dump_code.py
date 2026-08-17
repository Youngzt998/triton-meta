"""Print the Triton code Inductor generates for one fused GEMM+epilogue case.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=<repo> .venv/bin/python dump_code.py silu 4096 4096 256 TRITON
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

import torch._inductor.config as ic  # noqa: E402
from torch._inductor.utils import run_and_get_code  # noqa: E402

from cases import EPILOGUES, make_epi_args  # noqa: E402

name = sys.argv[1]
M, N, K = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
backends = sys.argv[5] if len(sys.argv) > 5 else "ATEN,TRITON"
ic.max_autotune_gemm_backends = backends

torch.manual_seed(0)
a = (torch.randn(M, K, device="cuda") / 8).to(torch.float16)
w = (torch.randn(K, N, device="cuda") / 8).to(torch.float16)
extra = make_epi_args(name, M, N, torch.float16)
fn = EPILOGUES[name]

g = torch.compile(lambda *xs: fn(xs[0] @ xs[1], *xs[2:]), mode="max-autotune-no-cudagraphs")
out, codes = run_and_get_code(g, a, w, *extra)
for c in codes:
    print(c)
