"""HIT-0001 reproduction: tl.cumsum is silently wrong when num_ctas > 1.

    source /home/youngzt/fuzz/r2-oracle/env.sh
    python HIT-0001/repro.py

Needs a Hopper (sm_90) or newer GPU: num_ctas > 1 is rejected on sm_89 and
below. Exit code 1 means the bug is present.

Three independent references agree with each other and disagree with the
compiled kernel: torch.cumsum, Triton's own interpreter (TRITON_INTERPRET=1),
and Triton itself at num_ctas=1.
"""
import os
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(x_ptr, y_ptr, N: tl.constexpr):
    off = tl.arange(0, N)
    tl.store(y_ptr + off, tl.cumsum(tl.load(x_ptr + off), axis=0))


@triton.jit
def sum_kernel(x_ptr, y_ptr, N: tl.constexpr):
    off = tl.arange(0, N)
    tl.store(y_ptr, tl.sum(tl.load(x_ptr + off), axis=0))


# The corpus kernel the fuzzer actually hit, kept verbatim.
# flaggems :: moe_align_block_size_stage2_vec
@triton.jit
def moe_align_block_size_stage2_vec(tokens_cnts_ptr, num_experts: tl.constexpr):
    pid = tl.program_id(0)
    offset = tl.arange(0, num_experts) + 1
    token_cnt = tl.load(tokens_cnts_ptr + offset * num_experts + pid)
    cnt = tl.cumsum(token_cnt, axis=0)
    tl.store(tokens_cnts_ptr + offset * num_experts + pid, cnt)


def main() -> int:
    bad = 0

    # ---- 1. the smallest case: cumsum of all-ones must be 1,2,3,... -------- #
    print("1. tl.cumsum of eight ones, one program, all-ones input")
    for ctas in (1, 2, 4, 8):
        x = torch.ones(8, device="cuda", dtype=torch.int32)
        y = torch.zeros(8, device="cuda", dtype=torch.int32)
        cumsum_kernel[(1, 1, 1)](x, y, N=8, num_warps=4, num_ctas=ctas)
        torch.cuda.synchronize()
        got = y.tolist()
        ok = got == [1, 2, 3, 4, 5, 6, 7, 8]
        bad += not ok
        print(f"   num_ctas={ctas}: {got}" + ("" if ok else "   <-- WRONG"))
    print("   the answer restarts from 1 every 8/num_ctas elements: each CTA")
    print("   scans only its own slice and never adds the earlier CTAs' total\n")

    # ---- 2. tl.sum on the same tensor is correct -------------------------- #
    print("2. tl.sum (tt.reduce) on the same tensor, same configs")
    for ctas in (1, 2, 4, 8):
        x = torch.ones(8, device="cuda", dtype=torch.int32)
        y = torch.zeros(1, device="cuda", dtype=torch.int32)
        sum_kernel[(1, 1, 1)](x, y, N=8, num_warps=4, num_ctas=ctas)
        torch.cuda.synchronize()
        got = int(y[0])
        ok = got == 8
        bad += not ok
        print(f"   num_ctas={ctas}: {got}" + ("" if ok else "   <-- WRONG"))
    print("   reduce has a cross-CTA path; scan does not\n")

    # ---- 3. larger tiles turn the wrong answer into a compiler assert ----- #
    print("3. the same scan at larger tile sizes")
    for N in (64, 256, 1024):
        x = torch.ones(N, device="cuda", dtype=torch.int32)
        y = torch.zeros(N, device="cuda", dtype=torch.int32)
        try:
            cumsum_kernel[(1, 1, 1)](x, y, N=N, num_warps=4, num_ctas=4)
            torch.cuda.synchronize()
            want = list(range(1, N + 1))
            ok = y.tolist() == want
            bad += not ok
            print(f"   N={N:5} num_ctas=4: "
                  + ("ok" if ok else f"WRONG, restarts every {N // 4}"))
        except Exception as e:
            print(f"   N={N:5} num_ctas=4: {type(e).__name__} "
                  "(ScanOpToLLVM.cpp:159 assertion)")

    # ---- 4. the corpus kernel from the fuzzing hit ------------------------ #
    print("\n4. the corpus kernel bch:moe_align_block_size_stage2_vec_88dc7283")
    NE = 8
    g = torch.Generator().manual_seed(7)
    base = torch.randint(1, 9, (9, NE), generator=g, dtype=torch.int32)
    ref = base.clone()
    ref[1:NE + 1, :] = torch.cumsum(base[1:NE + 1, :].to(torch.int64),
                                    dim=0).to(torch.int32)
    for ctas in (1, 2, 4):
        t = base.clone().cuda()
        moe_align_block_size_stage2_vec[(NE, 1, 1)](t, num_experts=NE,
                                                    num_warps=4, num_ctas=ctas)
        torch.cuda.synchronize()
        nd = int((t.cpu() != ref).sum())
        bad += nd > 0
        print(f"   num_ctas={ctas}: cells differing from torch = {nd}/72"
              + ("" if nd == 0 else "   <-- WRONG"))

    print(f"\nwrong results: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    if os.environ.get("TRITON_INTERPRET"):
        print("note: under TRITON_INTERPRET=1 num_ctas is ignored and every "
              "case is correct; that is the point of the comparison.")
    sys.exit(main())
