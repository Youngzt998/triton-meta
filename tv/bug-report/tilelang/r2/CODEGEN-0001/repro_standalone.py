"""CODEGEN-0001 -- self-contained reproduction (no fuzzing harness needed).

A `T.Pipelined` loop that stages a 1-D slice into shared memory.  TileLang
gives the shared buffer one slot per pipeline stage and lowers the copy to a
TMA bulk copy.  The slot stride is the slot size, so stage 1 starts at
`base + slot_bytes`.  `cp.async.bulk.tensor` needs a 128-byte aligned shared
destination, so every slot size that is not a multiple of 128 bytes dies with

    CUDA error: misaligned address

Usage
-----
    python repro_standalone.py            # sweep, one child process per point
    python repro_standalone.py 96         # one point: 96 halves = 192 bytes

One point per process on purpose: after one CUDA fault every later launch in
the same process reports the same error, which would make the sweep nonsense.
"""

import subprocess
import sys

SWEEP = [32, 64, 96, 128, 160, 192]  # halves -> 64,128,192,256,320,384 bytes

KERNEL = """import tilelang
import tilelang.language as T


@T.prim_func
def main(X: T.Tensor(({total},), "float16"), Y: T.Tensor(({total},), "float16")):
    with T.Kernel(1, threads=32) as bx:
        Xs = T.alloc_shared(({nelem},), "float16")
        for ko in T.Pipelined({iters}, num_stages=2):
            T.copy(X[ko * {nelem}], Xs)
            T.copy(Xs, Y[ko * {nelem}])
"""


def run_one(nelem, iters=4):
    import importlib.util
    import pathlib
    import tempfile

    import torch  # noqa: F401  loads the CUDA runtime first
    import tilelang  # noqa: F401

    total = nelem * iters
    d = pathlib.Path(tempfile.mkdtemp(prefix="codegen0001_"))
    path = d / "k.py"
    path.write_text(KERNEL.format(total=total, nelem=nelem, iters=iters))
    ms = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(ms)
    sys.modules["k"] = mod
    ms.loader.exec_module(mod)

    kern = tilelang.compile(mod.main, out_idx=None, target="cuda")
    for line in kern.get_kernel_source().splitlines():
        if "tma_load" in line:
            print("   ", line.strip())
    x = torch.arange(total, dtype=torch.float16, device="cuda")
    y = torch.zeros(total, dtype=torch.float16, device="cuda")
    kern(x, y)
    torch.cuda.synchronize()
    assert torch.equal(x, y)
    print("    RESULT: ran fine")


def main():
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        print(f"nelem={n} slot_bytes={n * 2} slot%128={n * 2 % 128}")
        run_one(n)
        return
    for n in SWEEP:
        print(f"nelem={n} slot_bytes={n * 2} slot%128={n * 2 % 128}")
        p = subprocess.run([sys.executable, __file__, str(n)],
                           capture_output=True, text=True)
        out = [l for l in (p.stdout + p.stderr).splitlines()
               if "tma_load" in l or "RESULT" in l or "CUDA error" in l]
        for l in out:
            print("   ", l.strip())


main()
