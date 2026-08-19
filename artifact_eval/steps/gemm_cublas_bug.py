"""gemm.cublas-bug -- cuBLAS itself returns a wrong answer on some shapes, so a disagreement is
not automatically ours.

Moved out of `artifact.py` unchanged.
"""
from __future__ import annotations

import ctypes
import os
import sys

from ._common import ROOT, print_design  # noqa: F401  (print_design: for whoever automates this)

NAME = "gemm.cublas-bug"
ORDER = 50
DESCRIPTION = "reproduce the cuBLAS split-K tail defect"
IMPLEMENTED = True

TABLES: dict = {}  # prints instructions only; the reproducer is run by hand and writes nothing


def cublas_drops_k_tail(torch, L, M, N, K, kind, out_dtype):
    """Is cuBLAS itself wrong on this shape?

    With A and B all ones every output must be exactly K. The known cuBLAS split-K defect returns
    the product over only the first `K - (K % block_k)` elements, so the output comes back as a
    smaller whole number. fp16 stops being able to represent consecutive integers above 2048, so
    the test is run in fp32 output where it is exact. Returns (is_wrong, observed, expected).

    Nothing calls this yet. It is the shape-level version of the standalone reproducer, and it is
    what an automated "is this mismatch cuBLAS's fault?" verdict on a `gemm.bitmatch` row would
    be built out of.
    """
    if kind == "fp8":
        return None, None, K  # the defect is an fp16/bf16 split-K path; fp8 never reaches it
    # The layouts are built here rather than through `_make_layouts` because the package only
    # describes fp16 and bf16 output to cuBLAS -- ask it for fp32 and it silently says fp16, and
    # reading the fp16 buffer as fp32 gives garbage. K reaches hundreds of thousands, which fp16
    # cannot even represent as an integer (it is exact only to 2048), so the output has to be
    # fp32 for the test to mean anything. This mirrors the standalone reproducer.
    lib = L._load_lt()
    a = torch.ones(M, K, device="cuda", dtype=torch.float16)
    b = torch.ones(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    handle, desc, pref = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    layouts = []
    try:
        L._ck("create", lib.cublasLtCreate(ctypes.byref(handle)))
        L._ck("desc", lib.cublasLtMatmulDescCreate(ctypes.byref(desc), L._CUBLAS_COMPUTE_32F, L._CUDA_R_32F))
        for rows, cols, dt in ((M, K, L._CUDA_R_16F), (K, N, L._CUDA_R_16F), (M, N, L._CUDA_R_32F)):
            lay = ctypes.c_void_p()
            L._ck(
                "layout",
                lib.cublasLtMatrixLayoutCreate(ctypes.byref(lay), ctypes.c_int(dt), ctypes.c_uint64(rows),
                                               ctypes.c_uint64(cols), ctypes.c_int64(cols)))
            L._set_row(lib, lay)
            layouts.append(lay)
        la, lb, lc = layouts
        L._ck("pref", lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)))
        ws = ctypes.c_size_t(L._workspace_bytes())
        L._ck("pref-set", lib.cublasLtMatmulPreferenceSetAttribute(pref, L._PREF_MAX_WS, ctypes.byref(ws), 8))
        wsbuf = L._ws_buffer(L._workspace_bytes())
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        rc = lib.cublasLtMatmul(handle, desc, ctypes.cast(ctypes.pointer(alpha), ctypes.c_void_p),
                                ctypes.c_void_p(a.data_ptr()), la, ctypes.c_void_p(b.data_ptr()), lb,
                                ctypes.cast(ctypes.pointer(beta), ctypes.c_void_p), ctypes.c_void_p(c.data_ptr()), lc,
                                ctypes.c_void_p(c.data_ptr()), lc, None, ctypes.c_void_p(wsbuf.data_ptr()),
                                ctypes.c_size_t(L._workspace_bytes()),
                                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize()
        if rc != 0:
            return None, None, K
        v = float(c.flatten()[0].item())
    except Exception:
        return None, None, K
    finally:
        del a, b, c
        torch.cuda.empty_cache()
    return (abs(v - K) > 0.5), v, K


def run(args, env):
    """Points at the reproducer that ships with the package; it is meant to be run by hand.

    It depends on nothing in bitequiv -- ctypes, torch and libcublasLt only -- so it can be
    handed to NVIDIA unchanged. It ends in an assert, so a successful reproduction exits
    non-zero; that is the intended behaviour and the reason this step does not run it for you.
    """
    script = os.path.join(ROOT, "bitequiv", "cublas_match", "cublas_gemm_bug_reproduce.py")
    print(f"""
[gemm.cublas-bug] run this by hand, once per libcublasLt you want to test:

    CUBLASLT=/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcublasLt.so.13.1.1.3 \\
        {sys.executable} {script}

A and B are all ones, so every element of C must be exactly K. The three parts carry different
K values because the loss happens only where cuBLASLt decides to split K across threadblocks,
and that decision moves with both the architecture and the library -- "large K" is not the
trigger. The script's header table says which part fires where. It ends in an assert, so a
successful reproduction exits non-zero.

Observed on this machine (sm_103):

    13.2.2   part one   K =  8648 -> 8640   short by 8
             part three K = 11528 -> 11520  short by 8
    13.1.1   part one   K =  8648 -> 8640   short by 8
             part three K = 11528 -> 11520  short by 8
    12.8.5   part two   K = 57608 -> 57600  short by 8
""")
