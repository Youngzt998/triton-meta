"""gemm.bitmatch -- does the Triton GEMM return cuBLAS's bytes, and when it does not, whose
fault is it?

The shape draw is `_common.draw_shape` plus two regimes added here, `gemv` and `simt`. The four
regimes in `_common` reach five of the nine cuBLAS plan modes and, over 67,306 shapes, sent 57
of them to a vector kernel and none at all to the CUDA-core chain GEMM -- so those two families
were the least tested part of the claim by a wide margin. See `_draw_shape` for what the two
new regimes are and what each is for.

That earlier run is kept as `cache/gemm.bitmatch.pre-regimes.jsonl` -- it is a real result, and
it is the record of what the narrower draw covered. It is not comparable shape for shape with
what this file now produces.
"""
from __future__ import annotations

import collections
import json
import random
import time

from ._common import digest, draw_shape, make_inputs, writer

NAME = "gemm.bitmatch"
ORDER = 10
DESCRIPTION = ("random shapes, is the Triton GEMM byte-identical to cuBLAS, "
               "and when not, is cuBLAS itself wrong")
IMPLEMENTED = True

TABLES = {
    "gemm.bitmatch": {
        "doc":
        "One row per random shape. Is the Triton GEMM byte-identical to cuBLAS on that "
        "shape, over `reps` independent input draws (even draws ordinary gaussian, odd "
        "draws with the exponents spread across the dtype range).",
        "cols": [
            ("M", "int", "rows of A"),
            ("N", "int", "columns of B"),
            ("K", "int", "contraction length"),
            ("dtype", "str", "operand dtype: fp16 or fp8 (e4m3)"),
            ("mode", "str", "which cuBLAS kernel family the plan resolved to"),
            ("reps", "int", "independent input draws compared on this shape"),
            ("n_differ", "int", "draws whose output differed from cuBLAS; 0 means byte-identical"),
            ("draws_that_differ", "str", "space-separated draw indices that differed, empty if none"),
            ("declined", "str", "non-empty if the shape is out of scope and no comparison was made"),
            ("error", "str", "non-empty if the shape failed to run"),
        ],
    },
}

# The nine modes a `plan.CublasGemmPlan` can carry, in the order that dataclass lists them. The
# end-of-run breakdown prints all nine whether or not a shape reached them.
#
# One of them, `split_blocks`, cannot be reached on this machine by any shape at all, so a draw
# is the wrong place to look for it: `plan._plan_tensor_core` returns it only for a
# (family, STAGES_ID) key listed in the arch profile's `block_level_keys`, and of the four
# profiles only sm_90 carries any. On sm_103 that tuple is empty, and the mode is dead code
# until someone measures a key for it.
PLAN_MODES = ("plain", "k_per_dot", "split", "split_blocks", "splitk_groups", "gemmsn", "gemv13", "gemv_cslice",
              "gemv14")


def _decline_reason(msg):
    """A decline message with the shape taken out of it, so the reasons group into a few lines
    instead of one line per shape."""
    reason = msg.split(": ", 1)[1] if ": " in msg else msg
    return reason.split(" on a ")[0]  # "... on a 65536-element gemv (measured up to 9728)"


# --------------------------------------------------------------------------------------------
# The shape draw
# --------------------------------------------------------------------------------------------
# Every rule below was measured on the machine in `data/env.csv` (GB300, sm_103, cuBLASLt
# 13.2.2) by asking the cuBLAS heuristic which kernel it would pick -- no GEMM was run to
# establish any of it. On another GPU or another cuBLAS the boundaries will sit elsewhere.


def _log_uniform(rng, lo, hi):
    """A draw uniform in the logarithm, so every decade of the range gets the same number of
    shapes. Drawing 1..120000 uniformly instead would put nine shapes in ten above 12,000 and
    almost never reach a short vector, which is where the three gemv kernels differ."""
    return max(lo, min(hi, int(round(lo * (hi / lo)**rng.random()))))


def _draw_gemv(rng, kind):
    """One of M, N is exactly 1: cuBLAS's vector kernels, ALGO 13 and ALGO 14.

    Those two only run on a shape with a dimension of exactly 1 -- `plan._plan_gemv` declines
    anything else, and cuBLAS does not offer them anyway. `_common.draw_shape` never sets a
    dimension to 1 on purpose; its narrowest regime, `decode`, draws M from 1..64, so M == 1
    turned up about once in 64 draws and only when the fp16 round-to-16 did not fire.

    Which of the three vector modes a shape gets is mostly N: measured over M == 1 with N off a
    multiple of 8, N above about 512 is `gemv13` at every K, and N below it is `gemv13` for K in
    the tens, `gemv_cslice` for K in the hundreds and mostly `gemv14` for K in the thousands. So
    the wide log draw on the other dimension is not decoration -- a narrow one would reach one
    mode of the three.
    """
    K = _log_uniform(rng, 16, 60000)
    other = _log_uniform(rng, 1, 120000)
    M, N = (1, other) if rng.random() < 0.5 else (other, 1)
    if kind == "fp8":
        # cuBLAS's fp8 rule is usually written "every dimension a multiple of 16", and on this
        # machine that is too strong: with N and K multiples of 16, M is unconstrained -- 1, 3
        # and 1023 are all accepted. So the 1 is left where it is rather than rounded up to 16,
        # which would turn the draw back into the thin GEMM it is meant not to be. What that
        # costs is in `_draw_shape`: the N == 1 half of these shapes mostly declines.
        K = max(16, K // 16 * 16)
        if N != 1:
            N = max(16, N // 16 * 16)
    return M, N, K


def _draw_simt(rng, kind):
    """A few rows against a wide, unaligned N: cuBLAS's CUDA-core chain GEMM, ALGO 11 and 16.

    Two conditions have to hold at once, which is why `_common.draw_shape` reaches this family
    zero times in 67,306 shapes rather than rarely. M must be between 2 and about 16 -- at M == 1
    cuBLAS picks a vector kernel and from about M == 24 a tensor-core one -- and N must NOT be a
    multiple of 8. Over an M in 2..16 x N in {128..8192, powers of two} x K grid the chain GEMM
    was chosen 0 times out of 637; over the same M and K with N off a multiple of 8 it was chosen
    for every K up to about 136 (up to about 45 once M reaches 8). Hence no rounding here, and
    hence a K range that reaches well below a hundred.

    The committed draw satisfies neither condition: M is 16 or more everywhere except `decode`,
    and `decode` pairs its small M with N and K in the thousands.
    """
    M = rng.randint(2, 16)
    N = _log_uniform(rng, 128, 4096)
    K = _log_uniform(rng, 8, 2048)
    if kind == "fp8":
        N, K = max(16, N // 16 * 16), max(16, K // 16 * 16)
    return M, N, K


def _draw_shape(rng):
    """`_common.draw_shape` 72 draws in 100, `gemv` 16, `simt` 12.

    A weight sets the ratio of the shape counts, not the ratio of the time, so what the split
    costs the four regimes that were already there had to be measured rather than assumed. Two
    four-minute trials said it would cost a lot: `simt` ran at 0.06 s a shape and the committed
    four at 0.07, but `gemv` at 0.22, and on those numbers a sixth of the draws going to `gemv`
    would have left the old four with three fifths of the shapes they used to get.

    The 90-minute run says otherwise, and the trials were the misleading measurement. It drew
    66,345 shapes from the committed four regimes against 67,306 in the whole previous budget --
    99% of them -- and got 15,086 gemv-shaped and 11,268 simt-shaped shapes on top. Almost all
    of `gemv`'s cost is compiling a Triton kernel for a vector recipe it has not seen yet, which
    is paid once per recipe and not once per shape, so a four-minute trial pays nearly all of it
    and a 90-minute run amortises it. Every mode the earlier run covered came out at least as
    large as before, so nothing was traded away for the new coverage.

    fp8 is one draw in six here against one in three in `_common.draw_shape`, because fp8 cannot
    reach either new family and a third would buy nothing. Both CUDA-core planners refuse fp8
    outright, and the question does not even reach them: for an fp8 operand cuBLAS's heuristic
    answers these shapes with a tensor-core kernel or with nothing. What an fp8 draw here does
    is worth having in the table rather than assuming, so a sixth of them are kept, and over the
    2,494 fp8 vector shapes of the 90-minute run this is what they did:

      M == 1, N a multiple of 16   accepted, all 1,246 of them, and routed to `plain`, or to
                                   `split` at deep K. Never to a vector kernel: cuBLAS has no
                                   fp8 one, so the fp8 rows here test the tensor-core modes on a
                                   single-row shape and nothing else.
      N == 1                       accepted only when M is a multiple of 8. The separation is
                                   exact over 1,248 shapes: all 133 accepted have M % 8 == 0 and
                                   all 1,115 declined do not. The draw does not round M, so 89%
                                   of them decline -- 658 because cuBLAS offers no algorithm at
                                   all and 433 because it offers ALGO_ID 74, a family the arch
                                   profile has never measured.
      M == 1 and N == 1            no algorithm, so skipped rather than declined. 76 shapes.

    A shape cuBLAS itself has no algorithm for is skipped, not counted: there is no answer to
    match, so declining is the correct result rather than a shortfall, and counting it would
    overstate the gap.  The rest are kept rather than drawn around.  A decline is a recorded
    outcome in this artifact and not a failure -- it is coverage lost, not wrong bits returned -- and an fp8
    column vector landing on an unmeasured cuBLAS family is a gap worth having a thousand rows
    of evidence for. Rounding M up to a multiple of 8 would turn them into `plain` rows, which
    would say less.
    """
    r = rng.random()
    if r >= 0.28:
        return draw_shape(rng)
    kind = rng.choice(["fp16", "fp16", "fp16", "fp16", "fp16", "fp8"])
    M, N, K = _draw_gemv(rng, kind) if r < 0.16 else _draw_simt(rng, kind)
    return M, N, K, kind


def run(args, env):
    import torch

    from bitequiv.cublas_match import cublas_equivalent_gemm, cublas_matmul
    from bitequiv.cublas_match.errors import CublasUnsupportedShape

    out = writer("gemm.bitmatch")
    rng = random.Random(args.seed)
    deadline = time.time() + args.minutes * 60
    n = ok = declined = mism = skipped = 0
    shapes_of, differ_of, declines = collections.Counter(), collections.Counter(), collections.Counter()
    print(f"\n[gemm.bitmatch] {args.minutes} min, {args.reps} input draws per shape, seed {args.seed}")
    while time.time() < deadline:
        M, N, K, kind = _draw_shape(rng)
        esz = 1 if kind == "fp8" else 2
        if (M * K + K * N + M * N) * esz > args.max_bytes:
            continue
        seed = M * 1000003 + N * 10007 + K
        rec = {"M": M, "N": N, "K": K, "dtype": kind, "reps": args.reps}
        try:
            diffs = []
            for r in range(args.reps):
                a, b = make_inputs(torch, M, N, K, kind, r, seed)
                if r == 0:
                    from bitequiv.cublas_match.gemm import _resolve
                    from bitequiv.cublas_match.ltapi import _kind_of
                    rec["mode"] = _resolve(a, b, _kind_of(a), torch.float16).mode
                same = digest(torch,
                              cublas_equivalent_gemm(a,
                                                     b, torch.float16)) == digest(torch,
                                                                                  cublas_matmul(a, b, torch.float16))
                if not same:
                    diffs.append(r)
                del a, b
                torch.cuda.empty_cache()
            rec["n_differ"] = len(diffs)
            rec["draws_that_differ"] = diffs
            shapes_of[rec["mode"]] += 1
            if diffs:
                mism += 1
                differ_of[rec["mode"]] += 1
                print(
                    f"  MISMATCH  {kind} {M}x{N}x{K} mode={rec.get('mode')}  "
                    f"{len(diffs)}/{args.reps} draws differ", flush=True)
            else:
                ok += 1
            n += 1
        except CublasUnsupportedShape as e:
            msg = str(e)
            if "no cuBLAS algo" in msg:
                # cuBLAS's own heuristic returned nothing, so there is no answer to be matched and
                # declining IS the right result. The shape is not a test case; it is not counted.
                skipped += 1
            else:
                declined += 1
                declines[_decline_reason(msg)] += 1
            rec["declined"] = msg
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        out.write(json.dumps(rec) + "\n")
        out.flush()
        torch.cuda.empty_cache()
    out.close()
    print(f"\n  {n} shapes x {args.reps} draws = {n * args.reps} comparisons")
    print(f"  byte-identical to cuBLAS      {ok}/{n}")
    print(f"  shapes with any differing draw {mism}   (logged in data/gemm.bitmatch.jsonl)")
    print(f"  declined (out of scope)       {declined}")
    print(f"  skipped, cuBLAS had no algorithm {skipped}   (no answer exists, so nothing to match)")
    if mism:
        print("  A mismatch is not automatically ours: cuBLAS itself drops the k tail on some\n"
              "  shapes -- run gemm.cublas-bug, which reproduces that defect on this machine.")
    print("\n  by cuBLAS kernel family. All nine plan modes are listed, zeros included: which\n"
          "  families a shape draw never reaches is the part of this table worth reading, and\n"
          "  it is invisible if only what came up is printed.\n")
    print(f"    {'mode':16} {'shapes':>8} {'differing':>10}")
    for mode in PLAN_MODES + tuple(m for m in sorted(shapes_of) if m not in PLAN_MODES):
        print(f"    {mode:16} {shapes_of[mode]:>8} {differ_of[mode]:>10}")
    if declines:
        print("\n  declined, by reason:")
        for reason, count in declines.most_common():
            print(f"    {count:>8}  {reason}")
