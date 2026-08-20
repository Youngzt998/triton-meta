"""gemm.perf.random -- five arms on random shapes: where does a GEMM that must return cuBLAS's
exact bits stand against cuBLAS itself, against torch, and against a search?

Five arms, one bit reference, the same operands and the same timing method for all of them.

    cublas               cuBLAS through `hot_cublas`.  NVIDIA's heuristic picks the kernel.
                         THE BIT REFERENCE, and the baseline every ratio is taken against.
    torch_auto           torch.compile(mode="max-autotune-no-cudagraphs") as its autotuner picks
                         it, extern included.  Unconstrained.
    torch_triton         the same with the extern excluded (max_autotune_gemm_backends="TRITON"),
                         so it is Inductor's own Triton template.  Unconstrained.
    bitequiv_autotuning  bit-equivalent: the kernel AS IT STOOD BEFORE the sm_103 rewrites, with
                         its configuration chosen by a full autotuner sweep.  Must equal cublas.
    gb300_accelerated    bit-equivalent: `cublas_equivalent_gemm` as it ships, the GB300-specific
                         rewrites and the fitted rule.  Must equal cublas.

The two torch arms are one arm compiled twice, not two implementations.  The two bit-equivalent
arms are byte-identical to cuBLAS and differ in HOW THEY GOT THERE: one searches a configuration
space, the other is rewrites written for this architecture plus a fitted rule.  They differ in
two ways at once -- the kernel and the way its configuration is chosen -- so a comparison between
them covers both, not just "search against hand-tuning".

WHAT IS REPORTED, AND WHAT IS NOT
---------------------------------
Every arm is reported against cuBLAS and against nothing else, as cuBLAS time over that arm's
time, so above 1 means faster than cuBLAS.  Ratios BETWEEN arms are deliberately not printed.
They hide the levels -- one arm over another says nothing about whether either is any good
against the reference -- and where two arms differ in more than one way they invite a causal
reading this measurement cannot support.  Four numbers on one baseline let a reader form
whatever comparison they want, and re-presenting the same data this way is what made it visible
that torch_auto and torch_triton are near-identical on every family, i.e. that excluding the
extern changes almost nothing.

In particular, gb300_accelerated against a torch arm is NOT the price of bit-exactness.  An
unconstrained autotuner's space CONTAINS the bit-exact configurations, so that ratio could not
exceed 1 if the two were the same kernel.  It does exceed 1 because they are not: ours has TMA
descriptors, a persistent grid and warp specialization, and Inductor's `mm_template` has none of
them.  It is a product comparison -- what a user gets by switching -- and the kernel difference
dominates whatever the bit constraint costs.

The per-shape measurement is `measure_shape`, and everything it is built from is importable:
`gemm.perf.static` runs the same five arms on fixed real-model shapes and imports them from here
rather than copying them.  The shape drawing and the report belong to this step.

bitequiv_autotuning, AND WHAT A SEARCH CAN AND CANNOT REACH
-----------------------------------------------------------
bitequiv_autotuning is the same kernels as gb300_accelerated before the sm_103 work, driven
by a search instead of a rule.
There is no separate pre-optimization package: the sm_103 commit added one branch at the top of
each launcher and left the original body underneath, character for character.  So bitequiv_autotuning launches
the original `@triton.jit` kernels from `bitequiv.cublas_match.kernels` directly, with the
launcher bodies replicated in `pre_gb300_launcher` so the configuration can be varied.
`_kcontig`, the int64 addressing and the row-major tile order are kept exactly as they are there.
The replication is checked, not assumed: at the default configuration every mode is byte-identical
to what the package's own pre-GB300 body returns.

The rule for the space: **anything that is part of the cuBLAS plan is pinned, because it defines
the answer; launch parameters are searched.**

    pinned      k_chunk / nsplit, k_per_dot, the residue position, merge_scheme, gemmsn's (S, B),
                gemv's (V, W, CC, count-down) and SPLITK_NUM.  All of these come out of
                `plan.static_plan` and moving one moves the bits.
    searched    BM, BN, BK (the Triton dot's k extent), num_warps, num_stages -- and BE and
                num_warps for the gemv families.

Three knobs named in the design are NOT searchable, and saying why is part of the result:

    GROUP_M      the original kernels compute the tile index as `pid // cdiv(N, BN)`.  There is
                 no GROUP_M constexpr to give a value to.  Adding one is a source rewrite.
    index width  every index in the original kernels is `.to(tl.int64)`, unconditionally.  There
                 is no I64 constexpr.  Adding one is a source rewrite.
    G            `_plain_gemm_k_per_dot` puts exactly KPD real k-elements in one `tl.dot` and
                 zero-pads them to BK.  Putting G groups in one dot is a source rewrite (it is
                 `_plain_gemm_kpd_wide` on the sm_103 side).  For `plain`, `split` and
                 `split_blocks`, where the dot's k extent IS the group, searching BK is searching
                 G, and it is searched.

That is the gap between the two bit-equivalent arms doing its job: those three, and dropping
the `_kcontig` copy that costs an
fp8 call its whole weight matrix, are the "it needed new code" half.

BK is searched.  A `tl.dot` of k extent `16*G` lowers to G chained k16 MMAs on one accumulator
in ascending k, so moving the loop boundary does not move any addition.

Three named exceptions are written into the space rather than left to chance:

  1. fp8 never gets BM < 64.  Below that Triton drops `tcgen05.mma.kind::f8f6f4` (k=32) for
     `mma.sync.m16n8k16` (k=16).  That is 20-36% faster and byte-identical on ordinary inputs,
     and differs on wide-exponent ones -- exactly the trap a search walks into if it can see it.
  2. gemv at vlen > 1 does not search BE or num_warps.  Measured not bit-neutral there.
  3. gemmsn keeps its accumulator narrow (fp32-per-thread is capped) and every candidate's LLVM
     IR is scanned for `.rdx`.  An unroll giving 8 or more fp32 per thread lets LLVM rewrite the
     addition chain into a horizontal reduction, which moves the result by an ulp, and
     `enable_fp_fusion=False` does not fix it.

The winner is verified byte-identical here, over the same ten draws as gb300_accelerated and torch_auto.  The
pruner is not trusted.  If the fastest configuration fails, the fastest one that passes is used
and `fallback` is set on the row.  The pre-GB300 default launch is always the first candidate, so
bitequiv_autotuning can never come out slower than the untuned original.

FIVE TIMINGS, ALL KEPT RAW
--------------------------
    device_flush_ms   graph replay per call, L2 flushed between replays   <- the headline today
    device_warm_ms    the same without the flush
    e2e_ms            an ordinary call, wall clock, no graph
    host_ms           CPU time to issue the call
    floor_ms          an empty kernel at the same grid, batched and replayed the same way

Which one becomes the headline is a decision for when the paper is written; the run should not
have to be repeated for it.

THE REPLAY FLOOR, AND WHY A ROW CARRIES `batch_n`
-------------------------------------------------
A one-call graph replay costs about 6.9 us on this box before any work of ours runs.  That is 2%
of a 289 us lm_head and 77% of a 2 us MoE expert GEMM, and because both arms pay the same 6.9 a
true 2x on the small one reads as 1.22x.  Those rows are not noisy, they are flattened toward 1,
which is the worst kind of wrong number: it reads as "the arms perform about the same" when
nothing of the kind was measured.

So a shape is timed with `batch_n` calls in ONE graph and the replay divided by `batch_n`.
`pick_batch` sets it from a three-replay probe of each arm, sized on the fastest one, and aims
to leave the fixed part of the replay at 5% of a per-call time.  A kernel already far above the
floor gets `batch_n = 1`, which is the old measurement unchanged, so only the rows the floor was
eating move.  One `batch_n` serves every arm of a shape, so the arms stay comparable.  `batch_n`
is on every row: rows taken before this existed carry none, and on a small shape those two are
not the same measurement.

The batch is `batch_n` calls on `batch_n` DISTINCT OPERAND COPIES.  The flush runs once per
replay, not per call, so a batch over ONE operand set would have call 1 cold and calls 2..N warm
-- measured here at 1.7x to 2.5x faster on a MoE expert weight, which fits in this card's 129 MiB
L2.  A copy per call means no call leaves anything in L2 for the next, so every call starts as
cold as it does at `batch_n = 1`: the batch changes what the floor costs, not what is timed.
Flushing inside the graph instead is not an option -- evicting a 129 MiB L2 takes a ~16 us
memset, a bigger floor than the 6.9 us one being removed.

`floor_ms` is an empty Triton kernel at the same grid, batched and replayed exactly like the
arms, so it is the per-call floor at that row's own `batch_n` and `near_floor` keeps its meaning
unchanged: under three times the floor, the measurement is mostly overhead.  It does not go to
zero with batching.  Graph nodes run one after another and one empty kernel completes in about
1 us on this box, which every call pays and so does a real caller.  A GEMM that small -- an
80x2048x8 LoRA merge -- stays flagged after batching, and it should: it is not distinguishable
from a launch.

`e2e_ms` and `host_ms` matter because device time flatters us.  Our TMA path spends roughly 100
us of host time per call building three tensor descriptors; graph replay captures that once and
makes it free, but a real caller pays it every call.  Reporting device time alone hides a cost
that is ours.

RUNNING IT
----------
    export PYTHONPATH=$(git rev-parse --show-toplevel)
    CUDA_VISIBLE_DEVICES=2 python -u artifact_eval/artifact.py --run gemm.perf.random --minutes 0

`--minutes 0` means run to completion; a positive value is a wall-clock cap and exits cleanly.
Everything else is an environment variable, because `artifact.py`'s CLI is shared by every step:

    PERF_RANDOM_PER_CELL     shapes per (family, dtype) cell.  Default 100, so 1400 in all.
    PERF_RANDOM_ROUNDS       measurement rounds, best of.  Default 3.
    PERF_RANDOM_MAX_CONFIGS  cap on bitequiv_autotuning's search space.  Default 0 = no cap.  A non-zero value
                             is recorded on every row it touched, so a bounded run cannot be
                             read as full coverage.
    PERF_RANDOM_SEARCH_S     wall-clock budget for bitequiv_autotuning's screening pass, per shape.  Default
                             150.  `space`, `searched` and `budget_hit` on the row say whether it
                             bit.  The sweep is dominated by Triton compilation, which the
                             on-disk cache makes free after the first shape of each kind, so the
                             budget mostly bounds the first few shapes.
    PERF_RANDOM_REFINE       how many of the screened configurations are re-timed with a real
                             CUDA-graph replay.  Default 16.
    PERF_RANDOM_CUBLASLT     which cuBLAS to match.  Default 13.1.1, the version the arch profile
                             was measured against and the one the paper's numbers use.
    PERF_RANDOM_FAMILIES     comma-separated subset, for a validation slice.
    PERF_RANDOM_LIMIT        stop after this many shapes in this process.  0 = no limit.
    PERF_RANDOM_LEASE_MIN    minutes before another worker may steal a stale claim.  Default 45.
    PERF_RANDOM_REPORT_ONLY  1 = print the table from what is on disk and exit.
    PERF_RANDOM_REPORT_TXT   also write the report to this path.

CHECKPOINTING
-------------
One JSONL record per (shape, arm), flushed as it is computed, so an interruption loses at most
one arm.  Restart skips completed work keyed on (family, dtype, M, N, K, arm); resuming is
re-running the same command.  `cache/gemm.perf.random.PAUSE` is checked at the top of each shape
and exits cleanly.  Work is claimed one shape at a time through an O_EXCL lock file with a lease,
not sliced up front, so a worker can join or leave mid-run without losing or duplicating a shape
-- which is how a third GPU joins once it is free.
"""
from __future__ import annotations

import errno
import json
import math
import os
import random
import re
import statistics
import time

from ._common import CACHE, HERE, digest, graph_ms, hot_cublas, make_inputs, pick_batch, writer

NAME = "gemm.perf.random"
ORDER = 20
DESCRIPTION = ("four-way comparison on random shapes: the bit constraint against torch, "
               "and our rewrites against a search")
IMPLEMENTED = True

# The seven sampling rules, crossed with fp16 and fp8.  The family is the SAMPLING RULE, not the
# answer: which cuBLAS plan mode a shape lands on is recorded separately and the report groups by
# that too, because a family does not map to a mode.
FAMILIES = ("any", "aligned", "unaligned", "deepk", "smallm", "gemv", "decode")

# The five arms, in the order a round measures them.  Interleaving them inside a round means a
# clock or a neighbour that drifts over the run hits all five the same way.
ARMS = ("cublas", "torch_auto", "torch_triton", "pre_search", "ours")

# What an arm is CALLED in a report. The `arm` field of a record keeps the value it has always
# had -- `pre_search`, `ours` -- so a CSV row and a table row are still the same row and a run
# started before this can be resumed by a run started after it. Only the printed name changes.
ARM_NAME = {
    "cublas": "cublas",
    "torch_auto": "torch_auto",
    "torch_triton": "torch_triton",
    "pre_search": "bitequiv_autotuning",
    "ours": "gb300_accelerated",
}

ARM_LABEL = {
    "cublas": "cuBLAS on the algorithm its own heuristic returns. THE BIT REFERENCE, and the baseline",
    "torch_auto": "torch.compile as its autotuner picks it, extern included",
    "torch_triton": "torch.compile with the extern excluded, so Inductor's own Triton template",
    "pre_search": "bit-equivalent: the kernel BEFORE the sm_103 rewrites, configuration chosen by a search",
    "ours": "bit-equivalent: the GB300-specific rewrites and the shipped fitted rule",
}

# The per-arm measurement columns.  `gemm.perf.static` imports these so the two steps cannot
# describe the same arm two different ways.  A row is one (shape, arm) pair -- not one shape with
# every arm in it -- so that an interruption loses one arm and not a whole shape.
ARM_COLS = [
    ("dtype", "str", "operand dtype: fp16 or fp8 (e4m3)"),
    ("arm", "str", "which arm this row measures: cublas (1), torch_auto (2a), torch_triton (2b), "
     "pre_search (3), ours (4)"),
    ("mode", "str", "the cuBLAS plan mode this SHAPE resolved to -- plain, k_per_dot, split, "
     "split_blocks, splitk_groups, gemmsn, gemv13, gemv_cslice or gemv14. Recorded per shape, "
     "never assumed from the family"),
    ("device_flush_ms", "float", "CUDA-graph replay PER CALL, L2 flushed between replays, "
     "median of the replays, best of the rounds. `batch_n` calls go into the graph and the "
     "replay is divided by `batch_n`"),
    ("device_warm_ms", "float", "the same replay without the flush"),
    ("batch_n", "int", "calls captured into one graph for this row, each on its own operand "
     "copy so none of them warms another's L2. 1 is one call per graph, which is what every row "
     "was before this column existed and what a kernel far above the replay floor still gets. "
     "Bigger means the ~6.9 us fixed cost of a replay was divided by this much, and on a small "
     "shape a row with a batch and a row without are not the same measurement"),
    ("e2e_ms", "float", "an ordinary call, wall clock, no graph: what a caller actually waits"),
    ("host_ms", "float", "CPU time to issue the call, no synchronise. Our TMA path builds three "
     "tensor descriptors per call, which graph replay makes free and a real caller does not"),
    ("floor_ms", "float", "an empty Triton kernel at the same grid, batched at the same "
     "`batch_n` and replayed the same way, per call. The cost of the replay before any work. It "
     "does not go to zero with batching: one empty kernel still completes in about 1 us, and "
     "graph nodes run one after another"),
    ("near_floor", "int", "1 when device_flush_ms is under three times floor_ms, i.e. the "
     "measurement is mostly launch overhead and every ratio on this row is compressed toward 1. "
     "Both sides are per call at this row's `batch_n`, so the rule means the same thing at any "
     "batch. A GEMM as small as a LoRA merge stays flagged after batching"),
    ("captured", "int", "1 if the call went into a CUDA graph. 0 is [NOT CAPTURED]: the device "
     "timings are empty and the row is kept, because a silently filtered set is a biased set"),
    ("config", "str", "the launch configuration this arm ran. For pre_search the winning one; "
     "for the torch arms the template configuration Inductor picked; empty for cuBLAS"),
    ("pick", "str", "what the torch autotuner actually chose: aten, triton_mm, decompose_k, or "
     "reduction (two reduction kernels and no GEMM template, which is what Inductor emits at "
     "M == 1)"),
    ("bit_ok", "int", "input draws whose output was byte-identical to cublas"),
    ("bit_total", "int", "input draws compared. Even draws are ordinary gaussian, odd draws "
     "spread the exponents across the dtype's range. Empty when the arm is not bit-checked"),
    ("searched", "int", "configurations bitequiv_autotuning actually measured on this shape"),
    ("space", "int", "configurations in bitequiv_autotuning's space for this shape before the search. Equal to "
     "`searched` unless PERF_RANDOM_MAX_CONFIGS bounded it or some of them failed to launch"),
    ("fallback", "int", "1 when bitequiv_autotuning's fastest configuration failed the byte check and the "
     "fastest one that passed was used instead"),
    ("budget_hit", "int", "1 when bitequiv_autotuning's search ran out of its wall-clock budget before the end "
     "of the space, so `searched` is short of `space` and the search covered the front of the "
     "order `_order_key` sets"),
    ("rdx_rejected", "int", "gemmsn configurations dropped because LLVM had turned the addition "
     "chain into a horizontal reduction (`.rdx` in the IR), which moves the result by an ulp"),
    ("rdx_unknown", "int", "gemmsn configurations whose IR could not be read, so the check above "
     "could not be made on them"),
    ("cublaslt", "str", "the libcublasLt this row was matched against"),
    ("worker", "str", "which worker produced the row, for tracing a machine-specific result"),
    ("declined", "str", "non-empty if this arm is out of scope on this shape and nothing was "
     "measured"),
    ("contended", "int", "compute processes other than ours on this GPU while the row was measured; "
     "must be 0. Non-zero means the timings are not this kernel's and the row needs re-taking: "
     "shared SMs, L2 and bandwidth make the kernel genuinely slower, and no per-process counter "
     "can subtract that back out. -1 means it could not be determined."),
    ("error", "str", "non-empty if the arm failed to run"),
]

TABLES = {
    "gemm.perf.random": {
        "doc":
        "One row per (random shape, arm). Seven sampling families crossed with fp16 and "
        "fp8, 100 shapes per cell, 1400 shapes, five arms each. All five timings are kept "
        "raw; the ratios in the paper are computed from them and are not stored, so which "
        "timing is the headline can change without re-running anything.",
        "cols": [
            ("family", "str", "sampling rule the shape was drawn from: any, aligned, unaligned, "
             "deepk, smallm, gemv or decode. This is the DRAW, not the answer"),
            ("shape_id", "int", "index into the fixed 1400-shape list, so a row can be traced "
             "back to the draw that produced it"),
            ("M", "int", "rows of A"),
            ("N", "int", "columns of B"),
            ("K", "int", "contraction length"),
            ("align", "str", "which alignment predicates hold on the operands as laid out: M16, "
             "N16, K16"),
            ("source", "str", "for the decode family, the model and projection the (N, K) came "
             "from; empty otherwise"),
        ] + ARM_COLS,
    },
}

# --------------------------------------------------------------------------------------------
# Shapes.  Generated once from a fixed seed and written to disk BEFORE anything is measured, so
# a restart cannot silently run a different experiment.
# --------------------------------------------------------------------------------------------

# Caps on the draw.  A 65536x65536x65536 fp16 GEMM is 8 GiB an operand and would take a minute a
# call, so a log-uniform draw that does not fit is rejected and redrawn.  Both numbers, and how
# many draws they rejected, are printed in the report: a cap that is not stated reads as full
# coverage.
_MAX_TFLOP = 12.0  # 2*M*N*K, so about 40 ms of a perfect GB300 fp16 GEMM
_MAX_OUT_ELEMS = 1 << 24  # M*N: bounds the fp32 split-K workspace, which is nsplit * M * N * 4


def _decode_shapes():
    """(N, K, source) for every projection in `fusion_moe/models_2026.py`.

    Reused rather than reinvented: every number in that table was read out of the model's own
    `config.json` with the key recorded, so a decode row can be traced to a weight that exists.
    """
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from fusion_moe.models_2026 import MODELS
    out = []
    for m in MODELS:
        add = lambda n, k, w: out.append((int(n), int(k), f"{m.name}.{w}"))  # noqa: E731
        if m.head_dim and m.gqa:
            add(m.qkv_out, m.hidden_size, "qkv_proj")
        if m.head_dim and m.one_o_proj:
            add(m.hidden_size, m.q_out, "o_proj")
        if m.intermediate_size:
            add(2 * m.intermediate_size, m.hidden_size, "gate_up_proj")
            add(m.hidden_size, m.intermediate_size, "down_proj")
        if m.moe_intermediate_size:
            add(2 * m.moe_intermediate_size, m.ein, "moe_gate_up")
            add(m.ein, m.moe_intermediate_size, "moe_down")
        add(m.vocab_size, m.hidden_size, "lm_head")
    return out


def _loguniform(rng, lo, hi):
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def _fits(M, N, K, kind, max_bytes):
    """Would this shape run at all?  Rejects are redrawn and the rate is reported."""
    if M * N > _MAX_OUT_ELEMS:
        return False
    if 2.0 * M * N * K > _MAX_TFLOP * 1e12:
        return False
    esz = 1 if kind == "fp8" else 2
    return (M * K + K * N) * esz + M * N * 2 <= max_bytes


def _fp8_ok(M, N, K):
    """What cuBLAS will accept for e4m3, measured on this box at 13.1.1 rather than assumed.

    Swept M and N over 1..1392 at several K: cuBLAS returns an algorithm exactly when K is a
    multiple of 16 and N is even, plus the one exception that an N of 1 works when M is even.
    M is otherwise free -- 1x1024x512 fp8 resolves fine -- so the small-M, gemv and decode fp8
    cells are real cells and not empty ones.  A shape outside this is not a measurement anybody
    could make: there is no cuBLAS to be identical to.
    """
    return K % 16 == 0 and (N % 2 == 0 or (N == 1 and M % 2 == 0))


def _draw_one(rng, family, kind, decode):
    """One shape from one sampling rule."""
    src = ""
    if family == "any":
        M, N, K = (_loguniform(rng, 1, 65536) for _ in range(3))
    elif family in ("aligned", "unaligned"):
        step = 32 if kind == "fp8" else 16
        M, N, K = (max(step, _loguniform(rng, 512, 65536) // step * step) for _ in range(3))
        if family == "unaligned":
            # "N % 16 != 0, or a stride not a multiple of 16".  The operands are contiguous, so
            # A's strides are (K, 1) and B's are (N, 1) for fp16 and (1, K) for fp8: the only
            # strides there are, are K and N.  Breaking either breaks the predicate.
            #
            # fp8 can only break N, and only to an even value: cuBLAS returns no algorithm at all
            # for an odd fp8 N or for a K that is not a multiple of 16.
            if kind == "fp8":
                N += rng.choice([2, 4, 6, 8, 10, 12, 14])
            elif rng.random() < 0.5:
                N += rng.choice([1, 3, 5, 7, 9, 11, 13, 15])
            else:
                K += rng.choice([1, 2, 4, 8])
    elif family == "deepk":
        K = _loguniform(rng, 8192, 65536)
        M = _loguniform(rng, 1, 512)
        N = _loguniform(rng, 1, 512)
    elif family == "smallm":
        M = rng.randint(2, 16)
        N = _loguniform(rng, 256, 65536)
        K = _loguniform(rng, 16, 4096)
    elif family == "gemv":
        K = _loguniform(rng, 16, 60000)
        if rng.random() < 0.5:
            M, N = 1, _loguniform(rng, 1, 65536)
        else:
            M, N = _loguniform(rng, 1, 65536), 1
    else:
        N, K, src = decode[rng.randrange(len(decode))]
        M = rng.randint(1, 8)
    M, N, K = max(1, M), max(1, N), max(1, K)
    if kind == "fp8":  # see `_fp8_ok`: outside this there is no cuBLAS to be identical to
        K = max(16, K // 16 * 16)
        if N == 1:
            M += M % 2
        elif N % 2:
            N += 1
    return M, N, K, src


def _build_shapes(seed, per_cell, max_bytes):
    """The whole shape list, deterministic in (seed, per_cell, max_bytes)."""
    decode = _decode_shapes()
    shapes, rejected, drawn = [], 0, 0
    for family in FAMILIES:
        for kind in ("fp16", "fp8"):
            rng = random.Random(f"{seed}:{family}:{kind}")
            got, seen = 0, set()
            while got < per_cell:
                drawn += 1
                M, N, K, src = _draw_one(rng, family, kind, decode)
                if not _fits(M, N, K, kind, max_bytes):
                    rejected += 1
                    continue
                if kind == "fp8" and not _fp8_ok(M, N, K):
                    rejected += 1
                    continue
                # The family is a rule, so check the rule rather than trusting the draw.
                if family == "aligned" and (N % 16 or K % 16):
                    continue
                if family == "unaligned" and not (N % 16 or K % 16):
                    continue
                if (M, N, K) in seen and family != "decode":
                    continue
                seen.add((M, N, K))
                got += 1
                align = (("M16", M % 16 == 0), ("N16", N % 16 == 0), ("K16", K % 16 == 0))
                shapes.append(
                    dict(family=family, dtype=kind, M=M, N=N, K=K, source=src,
                         align=" ".join(x for x, ok in align if ok)))
    for i, s in enumerate(shapes):
        s["shape_id"] = i
    return shapes, drawn, rejected


def _shapes_path():
    return os.path.join(CACHE, "gemm.perf.random.shapes.json")


def _load_or_make_shapes(seed, per_cell, max_bytes):
    """Written to disk before anything is measured.  If the file exists it wins, even if the
    parameters differ -- a resumed run has to be the same experiment, not a fresh draw."""
    path = _shapes_path()
    if not os.path.exists(path):
        shapes, drawn, rejected = _build_shapes(seed, per_cell, max_bytes)
        meta = {
            "seed": seed, "per_cell": per_cell, "max_bytes": max_bytes, "drawn": drawn, "rejected": rejected,
            "max_tflop": _MAX_TFLOP, "max_out_elems": _MAX_OUT_ELEMS, "written": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        os.makedirs(CACHE, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"meta": meta, "shapes": shapes}, f)
        os.replace(tmp, path)  # atomic, so two workers starting at once cannot see half a file
    with open(path) as f:
        blob = json.load(f)
    return blob["shapes"], blob["meta"]


# --------------------------------------------------------------------------------------------
# bitequiv_autotuning: the pre-GB300 launchers, and the space a search may move in.
#
# Everything in this section is importable. `gemm.perf.static` runs the same arm.
# --------------------------------------------------------------------------------------------

_POW2 = (16, 32, 64, 128, 256)


def _np2(x):
    return 1 << max(0, int(x - 1)).bit_length()


def _tile_choices(M, N, min_bm):
    """Tiles worth trying at this shape.  A tile may exceed the extent -- the rows past M are
    masked, load zeros and contribute nothing, and the shipped launcher already runs a 128-row
    tile at every M below 128 -- but a tile four times the extent only wastes machine, so the
    ladder stops one step above."""
    bms = [b for b in _POW2 if min_bm <= b <= max(min_bm, 2 * _np2(M))]
    bns = [b for b in _POW2 if b <= max(16, 2 * _np2(N))]
    return (bms or [min_bm]), (bns or [16])


def _smem_ok(BM, BN, BK, esz, stages):
    """A cheap pre-filter, so the search does not spend its time catching OutOfResources."""
    return (BM * BK + BK * BN) * esz * max(1, stages) <= 220000


def default_pre_gb300_config(mode, plan, M, N, K, kind):
    """The launch the pre-GB300 launcher would have used with no search at all.

    Put at the front of the space for two reasons.  It makes the pruning cutoff sensible from the
    first candidate instead of after a slow one poisons it; and it means bitequiv_autotuning can never come out
    slower than the untuned original, so its ratio against cuBLAS is a lower bound on what a
    search buys and not an
    artefact of a space that happened to miss the shipped point.
    """
    from bitequiv.cublas_match.arch import platform
    from bitequiv.cublas_match.kernels import _tile
    if mode == "plain":
        return {"BM": 128, "BN": 128, "BK": 64, "num_warps": 8, "num_stages": 3}
    if mode == "k_per_dot":
        return {"BM": 128, "BN": 128, "BK": 16, "num_warps": 8, "num_stages": 3}
    if mode in ("split", "split_blocks", "splitk_groups"):
        import torch
        dt = torch.float8_e4m3fn if kind == "fp8" else torch.float16
        BM, BN = _tile(M, N, dt, platform().fp8_min_bm)
        return {
            "BM": BM, "BN": BN, "BK": {"split": 64, "split_blocks": 32, "splitk_groups": 16}[mode], "num_warps": 4,
            "num_stages": 3
        }
    if mode == "gemmsn":
        import triton
        return {"BM": min(128, max(16, triton.next_power_of_2(M))), "BN": 64, "num_warps": 4, "num_stages": 3}
    return {"BE": 16, "num_warps": 4}


def _gemv_w(mode, plan):
    """Lanes cooperating on one output element, which sizes the (BE, W) accumulator."""
    return plan.gemv[0][1] if mode == "gemv13" else (plan.gemv[0] if mode == "gemv_cslice" else 128)


def _gemv_pin(mode, plan, K):
    """Exception 2: at vlen > 1 the gemv launches are NOT bit-neutral in BE or num_warps, so the
    space collapses to the one shipped launch.  Returns that config, or None to search."""
    from bitequiv.cublas_match.kernels import _gemv_vlen
    if mode == "gemv13":
        recipe, nsplit = plan.gemv
        vlen = _gemv_vlen(recipe[0], -(-K // nsplit) if nsplit > 1 else K, recipe[1])
    elif mode == "gemv14":
        vlen = plan.gemv[1]
    else:
        vlen = 1  # `_gemv_cslice` has no lane vector: a lane owns one contiguous slice
    return {"BE": 16, "num_warps": 4, "pinned": "vlen>1"} if vlen > 1 else None


def config_space(mode, plan, M, N, K, kind, cap=0):
    """bitequiv_autotuning's search space for one shape.  Returns (configs, size_before_cap).

    Everything the cuBLAS plan fixes is absent from this list by construction: the chunking, the
    k per dot, the residue position, the merge scheme, (S, B) and the gemv recipe are read off
    `plan` at launch time and never varied.  Only launch parameters are here.  The three named
    exceptions are applied as they are built, not filtered out afterwards.
    """
    esz = 1 if kind == "fp8" else 2
    min_bm = 64 if kind == "fp8" else 16  # exception 1: fp8 below BM 64 changes the MMA
    out = []
    if mode in ("plain", "k_per_dot", "split", "split_blocks", "splitk_groups"):
        bms, bns = _tile_choices(M, N, min_bm)
        kpd = plan.k_per_dot if mode in ("k_per_dot", "splitk_groups") else None
        bks = [b for b in (16, 32, 64, 128) if kpd is None or b >= kpd]
        for BM in bms:
            for BN in bns:
                for BK in bks:
                    for nw in (4, 8):
                        for ns in (2, 3, 4):
                            if not _smem_ok(BM, BN, BK, esz, ns) or BM * BN < 32 * nw:
                                continue
                            out.append({"BM": BM, "BN": BN, "BK": BK, "num_warps": nw, "num_stages": ns})
    elif mode == "gemmsn":
        # Exception 3, first half: keep the accumulator narrow.  `_simt_chain_gemm` carries three
        # (BM, BN) fp32 planes, so fp32-per-thread is 3*BM*BN/(32*num_warps); the cap keeps it
        # where the shipped kernel already is and away from the horizontal-reduction rewrite.
        for BM in (16, 32, 64, 128):
            for BN in (16, 32, 64, 128):
                for nw in (1, 2, 4, 8):
                    for ns in (1, 2, 3):
                        if BM > max(16, 2 * _np2(M)) or BN > max(16, 2 * _np2(N)):
                            continue
                        if 3 * BM * BN > 64 * 32 * nw or BM * BN < 32 * nw:
                            continue
                        out.append({"BM": BM, "BN": BN, "num_warps": nw, "num_stages": ns})
    elif mode in ("gemv13", "gemv_cslice", "gemv14"):
        pin = _gemv_pin(mode, plan, K)  # exception 2
        if pin:
            return [pin], 1
        nel, W = (M if N == 1 else N), _gemv_w(mode, plan)
        for BE in (4, 8, 16, 32, 64):
            for nw in (1, 2, 4, 8):
                if BE > max(4, 2 * _np2(nel)) or BE * W > 64 * 32 * nw:
                    continue
                out.append({"BE": BE, "num_warps": nw})
    d = default_pre_gb300_config(mode, plan, M, N, K, kind)
    out = [d] + sorted((c for c in out if c != d), key=lambda c: _order_key(c, M, N))
    size = len(out)
    return (out[:cap] if cap and size > cap else out), size


def _order_key(cfg, M, N):
    """The order the search walks the space in, so that stopping early still leaves a good subset.

    Tiles that fill the machine come first -- the same thing every autotuner's config prune does
    -- then the wider k step, then more warps, then more stages.  The order is a heuristic and it
    is stated here rather than hidden: a run that hit its budget covered the front of THIS order,
    and `space` and `searched` on the row say how much of it.
    """
    import triton
    BM, BN = cfg.get("BM", cfg.get("BE", 16)), cfg.get("BN", 1)
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, max(1, BN))
    return (abs(math.log(max(tiles, 1) / 148.0)), -cfg.get("BK", 0), -cfg.get("num_warps", 0),
            -cfg.get("num_stages", 0))


def pre_gb300_launcher(mode, plan, cfg, kind=None):
    """A callable `(a, b, out_dtype, scale) -> c` running the ORIGINAL kernel for `mode` at `cfg`.

    Each body below is the corresponding launcher in `bitequiv/cublas_match/kernels.py` with its
    sm_103 branch removed and its hard-coded launch constants replaced by `cfg`.  Nothing else
    moves: `_kcontig`, the grid shape, the workspace dtype, the argument order and the runtime
    plan arguments are all as they are there.  Verified: at `default_pre_gb300_config` every mode
    returns exactly what the package's own pre-GB300 body returns.
    """
    import torch
    import triton
    from bitequiv.cublas_match import kernels as KK
    from bitequiv.cublas_match.ltapi import DEVICE

    cdiv = triton.cdiv

    def plain(a, b, out_dtype, scale):
        BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
        b = KK._kcontig(b)
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
        KK._plain_gemm[(cdiv(M, BM) * cdiv(N, BN), )](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                                                      b.stride(1), c.stride(0), c.stride(1), scale, BM=BM, BN=BN, BK=BK,
                                                      num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        return c

    def k_per_dot(a, b, out_dtype, scale):
        BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
        b = KK._kcontig(b)
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
        KK._plain_gemm_k_per_dot[(cdiv(M, BM) * cdiv(N, BN), )](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                                                                b.stride(1), c.stride(0), c.stride(1), scale,
                                                                plan.k_per_dot, plan.leading_group_k, BM=BM, BN=BN,
                                                                BK=BK, num_warps=cfg["num_warps"],
                                                                num_stages=cfg["num_stages"])
        return c

    def split(a, b, out_dtype, scale):
        BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
        b = KK._kcontig(b)
        M, K = a.shape
        N = b.shape[1]
        nsplit = (K + plan.k_chunk - 1) // plan.k_chunk
        ntile = cdiv(M, BM) * cdiv(N, BN)
        w = torch.empty(nsplit, M, N, device=DEVICE, dtype=torch.float32)
        KK._splitk_partial[(ntile, nsplit)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                            w.stride(0), w.stride(1), w.stride(2), plan.k_chunk, BM=BM, BN=BN, BK=BK,
                                            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
        KK._splitk_combine[(ntile, )](w, c, M, N, w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1),
                                      nsplit, scale, BM=BM, BN=BN, num_warps=cfg["num_warps"])
        return c

    def split_blocks(a, b, out_dtype, scale):
        BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
        b = KK._kcontig(b)
        M, K = a.shape
        N = b.shape[1]
        nsplit = (K + plan.k_chunk - 1) // plan.k_chunk
        ntile = cdiv(M, BM) * cdiv(N, BN)
        w = torch.empty(nsplit, M, N, device=DEVICE, dtype=torch.float32)
        KK._splitk_partial_blocks[(ntile, nsplit)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                                   w.stride(0), w.stride(1), w.stride(2), plan.k_chunk, plan.block_k,
                                                   BM=BM, BN=BN, BK=BK, num_warps=cfg["num_warps"],
                                                   num_stages=cfg["num_stages"])
        c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
        KK._splitk_combine[(ntile, )](w, c, M, N, w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1),
                                      nsplit, scale, BM=BM, BN=BN, num_warps=cfg["num_warps"])
        return c

    def splitk_groups(a, b, out_dtype, scale):
        BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
        b = KK._kcontig(b)
        M, K = a.shape
        N = b.shape[1]
        nsplit = (K + plan.k_chunk - 1) // plan.k_chunk
        ntile = cdiv(M, BM) * cdiv(N, BN)
        w = torch.empty(nsplit, M, N, device=DEVICE, dtype=torch.float32)
        KK._splitk_partial_k_per_dot[(ntile, nsplit)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                                                      b.stride(1), w.stride(0), w.stride(1), w.stride(2), plan.k_chunk,
                                                      plan.k_per_dot, plan.block_k, BM=BM, BN=BN, BK=BK,
                                                      num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
        KK._splitk_combine_modes[(ntile, )](w, c, M, N, w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1),
                                            nsplit, scale, CMODE=plan.merge_scheme, BM=BM, BN=BN,
                                            num_warps=cfg["num_warps"])
        return c

    def gemmsn(a, b, out_dtype, scale):
        BM, BN = cfg["BM"], cfg["BN"]
        S, sub = plan.simt
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
        k = KK._simt_chain_gemm[(cdiv(M, BM), cdiv(N, BN))](a, b, c, M, N, K, -(-K // S), sub or K, a.stride(0),
                                                            a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                                                            c.stride(1), BM=BM, BN=BN, S=S, num_warps=cfg["num_warps"],
                                                            num_stages=cfg["num_stages"])
        gemmsn.kernel = k  # exception 3's second half reads its LLVM IR back out; see `_rdx`
        return c

    def gemv13(a, b, out_dtype, scale):
        BE, nw = cfg["BE"], cfg["num_warps"]
        recipe, nsplit = plan.gemv
        _, W, CC, down = recipe
        K = a.shape[1]
        c = torch.empty(a.shape[0], b.shape[1], device=DEVICE, dtype=out_dtype)
        nel, *strides, sc = KK._gemv_axis(a, b, c)
        chunk = -(-K // nsplit) if nsplit > 1 else K
        vlen = KK._gemv_vlen(recipe[0], chunk, W)

        def lane(out, k0, ke, sc_):
            ntile = -(-(ke - k0) // (vlen * W))
            per = CC or ntile
            KK._gemv_lane[(cdiv(nel, BE), )](a, b, out, nel, k0, ke, *strides, sc_, -(-ntile // per), per, V=vlen, W=W,
                                             DOWN=down, BE=BE, num_warps=nw)

        if nsplit <= 1:
            lane(c, 0, K, sc)
            return c
        w = torch.zeros(nsplit, nel, device=DEVICE, dtype=torch.float32)
        for sl in range(nsplit):
            k0 = sl * chunk
            if k0 >= K:
                continue  # the workspace is zeroed, so a slice past K contributes an exact 0
            lane(w[sl], k0, min(k0 + chunk, K), 1)
        KK._gemv_slice_combine[(cdiv(nel, BE), )](w, c, nel, w.stride(0), sc, nsplit, BE=BE, num_warps=nw)
        return c

    def gemv_cslice(a, b, out_dtype, scale):
        BE, nw = cfg["BE"], cfg["num_warps"]
        w_lanes, chunk = plan.gemv
        K = a.shape[1]
        c = torch.empty(a.shape[0], b.shape[1], device=DEVICE, dtype=out_dtype)
        nel, *strides, sc = KK._gemv_axis(a, b, c)
        slice_k = -(-K // w_lanes)
        KK._gemv_cslice[(cdiv(nel, BE), )](a, b, c, nel, K, *strides, sc, slice_k, -(-slice_k // chunk), C=chunk,
                                           W=w_lanes, BE=BE, num_warps=nw)
        return c

    def gemv14(a, b, out_dtype, scale):
        BE, nw = cfg["BE"], cfg["num_warps"]
        nblock, v = plan.gemv
        K = a.shape[1]
        c = torch.empty(a.shape[0], b.shape[1], device=DEVICE, dtype=out_dtype)
        nel, *strides, sc = KK._gemv_axis(a, b, c)
        ntile = -(-K // (v * 128))
        w = torch.zeros(nblock, nel, device=DEVICE, dtype=torch.float32)
        KK._gemv_block_dot[(cdiv(nel, BE), nblock)](a, b, w, nel, K, *strides, w.stride(0), nblock, -(-ntile // nblock),
                                                    V=v, BE=BE, num_warps=nw)
        KK._gemv_block_reduce[(cdiv(nel, BE), )](w, c, nel, w.stride(0), sc, nblock, -(-nblock // 128), BE=BE,
                                                 num_warps=nw)
        return c

    return {
        "plain": plain, "k_per_dot": k_per_dot, "split": split, "split_blocks": split_blocks, "splitk_groups":
        splitk_groups, "gemmsn": gemmsn, "gemv13": gemv13, "gemv_cslice": gemv_cslice, "gemv14": gemv14
    }[mode]


def config_str(cfg):
    """A configuration as one sortable string, for the record."""
    return " ".join(f"{k}={v}" for k, v in sorted(cfg.items()))


def _rdx(fn):
    """Exception 3's second half: did LLVM turn the addition chain into a horizontal reduction?

    A wide enough accumulator lets LLVM rewrite the chain into `llvm.vector.reduce.fadd`, whose
    operands it leaves named `.rdx` in the IR.  That moves the result by an ulp and
    `enable_fp_fusion=False` does not fix it.  Reading the IR catches it on every configuration,
    where a byte check only catches it on a draw that happens to expose the ulp.

    True, False, or None when the IR could not be read -- which is counted, not called clean.
    """
    try:
        return ".rdx" in fn.kernel.asm["llir"]
    except Exception:
        return None


def _screen_ms(torch, call, iters=3):
    """One launch to compile it, then `iters` back-to-back launches timed with CUDA events.

    For RANKING only: no graph, no L2 flush.  On a kernel shorter than its own launch this
    measures the launch, but it measures the same launch for every candidate, so the order it
    produces is still usable -- and the top of that order is then re-timed properly by
    `graph_ms`.  Screening this way instead of capturing a graph per candidate cuts the sweep's
    device work about sevenfold, which matters at 500 candidates a shape.
    """
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    call()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        call()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def search_pre_gb300(torch, a, b, plan, kind, out_dtype, flush, cap=0, budget_s=150.0, refine=16):
    """Screen the whole space, then measure the best `refine` of it properly.  Ranked, fastest
    first.

    Returns (ranked, info).  `info` carries what the record needs to be read honestly: `space` is
    how many configurations there were, `searched` how many were screened, and `budget_hit` says
    the wall-clock budget stopped it before the end.  Both numbers land on every arm-3 row, so a
    truncated search cannot be read as an exhaustive one.

    Nothing here decides correctness.  The caller byte-checks the winner, because a pruner that
    also judged the bits would be marking its own work.

    Cost, and why there is a budget at all: the sweep is dominated by Triton COMPILATION, not by
    running the kernels -- a 562-configuration space costs about 0.3 s a configuration the first
    time and almost nothing once Triton's on-disk cache has it.  The budget bounds the first
    shape of each new kind without bounding the rest.  For scale: Inductor's own mm autotune
    considers 55 candidates.
    """
    cfgs, size = config_space(plan.mode, plan, a.shape[0], b.shape[1], a.shape[1], kind, cap)
    info = {"space": size, "searched": 0}
    screened, rejected, unknown, stop = [], 0, 0, time.time() + budget_s
    for i, cfg in enumerate(cfgs):
        if i and time.time() > stop:
            info["budget_hit"] = 1
            break
        try:
            fn = pre_gb300_launcher(plan.mode, plan, cfg, kind)
            call = lambda f=fn: f(a, b, out_dtype, 1.0)  # noqa: E731
            if plan.mode == "gemmsn":  # exception 3
                call()
                torch.cuda.synchronize()
                r = _rdx(fn)
                if r is True:
                    rejected += 1
                    continue
                unknown += 1 if r is None else 0
            screened.append((_screen_ms(torch, call), cfg))
        except Exception:
            torch.cuda.empty_cache()
            continue
    screened.sort(key=lambda x: x[0])
    ranked, nb = [], None
    for _, cfg in screened[:refine]:
        try:
            fn = pre_gb300_launcher(plan.mode, plan, cfg, kind)
            call = lambda f=fn: f(a, b, out_dtype, 1.0)  # noqa: E731
            if nb is None:
                # Batched for the same reason the arms are: on a small shape the 6.9 us replay
                # floor is most of the number and every candidate reads the same, so the order
                # comes out of noise. One operand set is enough here -- this only ORDERS the
                # candidates and the winner is re-timed properly, cold, with the arms.
                nb = pick_batch(torch, [call], flush, floor_ms(torch, a.shape[0], b.shape[1], flush, reps=6))
            t = graph_ms(torch, call, flush, reps=6, batch=nb)
            if t is not None:
                ranked.append((t, cfg))
        except Exception:
            torch.cuda.empty_cache()
            continue
    ranked.sort(key=lambda x: x[0])
    info["searched"] = len(screened)
    if rejected:
        info["rdx_rejected"] = rejected
    if unknown:
        info["rdx_unknown"] = unknown
    if not ranked:
        info["error"] = "no configuration in the space ran"
    return ranked, info


# --------------------------------------------------------------------------------------------
# Arm 2: torch.compile
# --------------------------------------------------------------------------------------------

_PICK_RE = re.compile(
    r"BLOCK_M['\"]?\s*[:=]\s*(\d+).{0,400}?BLOCK_N['\"]?\s*[:=]\s*(\d+).{0,400}?"
    r"BLOCK_K['\"]?\s*[:=]\s*(\d+)", re.S)
_WARPS_RE = re.compile(r"num_warps['\"]?\s*[:=]\s*(\d+)")
_STAGES_RE = re.compile(r"num_stages['\"]?\s*[:=]\s*(\d+)")


def classify_inductor_pick(code):
    """What the Inductor autotuner actually produced, read off the code it generated.

    `aten` is the extern call, `triton_mm` a GEMM template, `decompose_k` the split-K rewrite,
    and `reduction` is two reduction kernels and no GEMM template at all -- what Inductor emits
    at M == 1.  That last case is the honest torch_triton there, so it is measured, not skipped.
    Returns (pick, template configuration).
    """
    blob = "\n".join(code)
    picks = []
    if "decompose_k" in blob:
        picks.append("decompose_k")
    if re.search(r"extern_kernels\.(mm|addmm|bmm|_scaled_mm)|aten\._scaled_mm", blob):
        picks.append("aten")
    if re.search(r"triton_tem_fused|triton_mm|@triton_heuristics\.template", blob):
        picks.append("triton_mm")
    if not picks and re.search(r"triton_red_fused|triton_per_fused|triton_poi_fused", blob):
        picks.append("reduction")
    cfg = ""
    m = _PICK_RE.search(blob)
    if m:
        cfg = f"BLOCK_M={m.group(1)} BLOCK_N={m.group(2)} BLOCK_K={m.group(3)}"
        w, s = _WARPS_RE.search(blob), _STAGES_RE.search(blob)
        cfg += f" num_warps={w.group(1)}" if w else ""
        cfg += f" num_stages={s.group(1)}" if s else ""
    return ("+".join(picks) or "unknown"), cfg


class TorchCannotExpress(Exception):
    """torch has no way to run this GEMM, so the two torch arms do not exist on this shape.

    That is a decline and not a failure, and the row keeps it: `torch._scaled_mm` refuses an fp8
    operand whose N or K is not a multiple of 16, while cuBLAS itself only asks for K a multiple
    of 16 and N even.  So there are fp8 shapes with a cuBLAS answer, our answer and a pre-GB300
    answer, and no unconstrained torch baseline at all -- which is worth recording rather than
    quietly leaving out of the fp8 column.
    """


def torch_can_express(kind, M, N, K):
    """Will `torch.compile` accept this shape?  See `TorchCannotExpress`."""
    return kind != "fp8" or (N % 16 == 0 and K % 16 == 0)


def compile_torch_arm(torch, a, b, kind, out_dtype, backends):
    """Compile one torch.compile arm on this shape.  Returns (callable(x, y), pick, config).

    The callable takes its operands rather than closing over `a` and `b`, so the timing can run
    it on one operand copy per call in the batch.  `dynamic=False` guards on shape, dtype and
    stride, not on identity, so a copy of the same shape reuses the same compiled kernel.

    `backends` is what goes into `max_autotune_gemm_backends`: "ATEN,TRITON" is torch_auto, the thing
    a user actually gets, and "TRITON" is torch_triton, the extern excluded so the number is a Triton
    kernel and not cuBLAS wearing a Triton hat.

    `mm` is defined here, so 2a and 2b hand dynamo two different function objects and get two
    independent caches.  `torch._dynamo.reset()` is deliberately NOT called here: it would throw
    away the arm compiled a moment ago.  `measure_shape` resets once per shape instead.
    """
    import torch._inductor.config as icfg
    from torch._inductor.utils import run_and_get_code
    M, K = a.shape
    N = b.shape[1]
    if not torch_can_express(kind, M, N, K):
        raise TorchCannotExpress(f"torch._scaled_mm needs both dimensions of mat2 divisible by 16, "
                                 f"and mat2 here is [{K}, {N}]")
    if kind == "fp8":
        sa = torch.ones((), device="cuda", dtype=torch.float32)
        sb = torch.ones((), device="cuda", dtype=torch.float32)

        def mm(x, y):
            return torch._scaled_mm(x, y, sa, sb, out_dtype=out_dtype)
    else:

        def mm(x, y):
            return torch.matmul(x, y)

    _quiet_inductor()
    with icfg.patch({"max_autotune": True, "max_autotune_gemm": True, "max_autotune_gemm_backends": backends}):
        fn = torch.compile(mm, mode="max-autotune-no-cudagraphs", dynamic=False)
        _, code = run_and_get_code(fn, a, b)
    pick, cfg = classify_inductor_pick(code)
    return (lambda x, y: fn(x, y)), pick, cfg


def _quiet_inductor():
    """Stop Inductor printing its whole candidate list per shape.

    1400 shapes times two arms times 55 candidates is tens of megabytes of log that says nothing
    the record does not already carry -- the winner and its configuration are in `pick` and
    `config`.  This only silences a `sys.stderr.write`; nothing measured changes.  Set
    PERF_RANDOM_INDUCTOR_LOG=1 to keep it.
    """
    if os.environ.get("PERF_RANDOM_INDUCTOR_LOG"):
        return
    import logging
    try:
        import torch._inductor.select_algorithm as sa
        sa.PRINT_AUTOTUNE = False
    except Exception:
        pass
    # A shape torch refuses is logged with a full traceback at error level from inside dynamo's
    # fake-tensor pass. `torch_can_express` catches those before they get there, but a new one
    # would otherwise bury the run log in stack traces.
    logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)


# --------------------------------------------------------------------------------------------
# Timing.  `graph_ms` from artifact.py is the only device timer; the two host numbers below are
# the ones it deliberately does not make.
# --------------------------------------------------------------------------------------------

_NOOP = {}


def _noop_kernel():
    import triton
    if "k" not in _NOOP:

        @triton.jit
        def _empty(X):
            pass

        _NOOP["k"] = _empty
    return _NOOP["k"]


def floor_ms(torch, M, N, flush, reps=25, batch=1):
    """An empty Triton kernel at a grid of the same order, batched and replayed like an arm.

    This is the cost of the replay before any work: about 6.9 us for one call on a GB300.  gemv,
    decode and small-M kernels are 5-30 us, so it is a large fraction of them and compresses
    every ratio toward 1.  A 128x128 tile is the grid the shipped launcher uses on most shapes;
    for an empty kernel the grid barely matters, because what is being measured is the launch.

    Measured at the arm's own `batch`, so it is the floor per call that that row actually paid.
    It falls with the batch but not to zero: about 6 us of it is per replay and about 1 us is
    the completion of one empty kernel, which every call in the graph pays.
    """
    import triton
    k = _noop_kernel()
    grid = (max(1, triton.cdiv(M, 128) * triton.cdiv(N, 128)), )
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    return graph_ms(torch, lambda: k[grid](x), flush, reps=reps, batch=batch)


def host_and_e2e_ms(torch, fn, calls=20):
    """(e2e_ms, host_ms): an ordinary call end to end, and only the time to issue it.

    The gap between them is what a real caller pays that a graph replay hides -- our TMA path
    spends roughly 100 us of host time per call building three tensor descriptors.
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    host, e2e = [], []
    for _ in range(calls):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        host.append((t1 - t0) * 1e3)
        e2e.append((t2 - t0) * 1e3)
    return statistics.median(e2e), statistics.median(host)


def _reps_for(torch, fn, flush, batch=1):
    """Enough replays for a stable median, few enough that a 10 ms GEMM does not cost a second."""
    t = graph_ms(torch, fn, flush, reps=3, batch=batch)
    return None if t is None else max(5, min(25, int(20.0 / max(t * batch, 1e-3))))


def time_arms(torch, series, flush, rounds=3, batch=1):
    """The five timings for every arm, arms interleaved inside a round, best of the rounds.

    `series[arm]` is a LIST of callables: the same arm on `batch` distinct operand copies, which
    the graph cycles through so no call in the batch warms the next one's L2.  A one-element
    list is the ordinary un-batched call.

    Interleaving matters: a clock or a neighbouring process that drifts over the run then hits
    all the arms the same way instead of penalising whichever one is measured last.  A call that
    cannot be captured into a graph still gets its host and end-to-end numbers, because those are
    exactly what a real caller pays, and the row records `captured = 0` rather than disappearing.
    An arm whose batched capture fails -- an allocation the graph pool will not take, say -- is
    retimed on its own at `batch = 1` and says so in `batch_n`, rather than losing its timings.

    Returns {arm: {timing: value}}, with the missing device timings simply absent.
    """
    noflush = torch.empty(0, device="cuda", dtype=torch.int32)
    res = {arm: {} for arm in series}
    reps, nb = {}, {}
    for arm, fns in series.items():
        nb[arm] = batch
        try:
            reps[arm] = _reps_for(torch, fns, flush, batch)
            if reps[arm] is None and batch > 1:  # the batch would not capture; time one call
                nb[arm] = 1
                reps[arm] = _reps_for(torch, fns[0], flush, 1)
        except Exception as e:
            res[arm]["error"] = f"{type(e).__name__}: {e}"[:200]
            reps[arm] = None
        res[arm]["batch_n"] = nb[arm]
    for _ in range(rounds):
        for arm in ARMS:
            if arm not in series or reps.get(arm) is None:
                continue
            fns = series[arm] if nb[arm] > 1 else series[arm][0]
            try:
                pair = (("device_flush_ms", graph_ms(torch, fns, flush, reps=reps[arm], batch=nb[arm])),
                        ("device_warm_ms", graph_ms(torch, fns, noflush, reps=reps[arm], batch=nb[arm])))
                for key, val in pair:
                    if val is not None and (res[arm].get(key) is None or val < res[arm][key]):
                        res[arm][key] = val
            except Exception as e:
                res[arm].setdefault("error", f"{type(e).__name__}: {e}"[:200])
    for arm, fns in series.items():
        try:
            res[arm]["e2e_ms"], res[arm]["host_ms"] = host_and_e2e_ms(torch, fns[0])
        except Exception as e:
            res[arm].setdefault("error", f"{type(e).__name__}: {e}"[:200])
    return res


# --------------------------------------------------------------------------------------------
# The four-arm measurement of one shape.  This is the part `gemm.perf.static` shares.
# --------------------------------------------------------------------------------------------


def measurement_options(args):
    """The knobs `measure_shape` reads, from the environment and the shared CLI."""
    return {
        "rounds": int(os.environ.get("PERF_RANDOM_ROUNDS", 3)),
        "max_configs": int(os.environ.get("PERF_RANDOM_MAX_CONFIGS", 0)),
        "search_s": float(os.environ.get("PERF_RANDOM_SEARCH_S", 150)),
        "refine": int(os.environ.get("PERF_RANDOM_REFINE", 16)),
        "draws": max(2, args.reps),
        "worker": f"{os.uname().nodename}:{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}:{os.getpid()}",
        "per_cell": int(os.environ.get("PERF_RANDOM_PER_CELL", 100)),
        "lease_s": float(os.environ.get("PERF_RANDOM_LEASE_MIN", 45)) * 60,
        "limit": int(os.environ.get("PERF_RANDOM_LIMIT", 0)),
    }


def make_flush_buffer(torch, mib=256):
    """The buffer `graph_ms` zeroes between replays to evict L2.  One per process."""
    return torch.empty(mib * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)


def others_on_our_gpu():
    """How many compute processes are on our GPU besides us.

    A timing number taken while another process is resident is not a measurement of this kernel.
    It is not an attribution problem that a per-process counter could fix -- without MPS the driver
    swaps our context out entirely and the wall clock between the two CUDA events keeps running,
    and with MPS the kernels genuinely share SMs, L2 and bandwidth, so the kernel really is slower.
    Nothing can subtract that out afterwards, so the only honest move is to record that it happened
    and re-take the row.

    Returns -1 when it cannot tell, which is recorded rather than treated as zero.
    """
    import os
    import subprocess
    try:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        q = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
        uuids = {i.strip(): u.strip() for i, u in (l.split(",") for l in _sh(q).splitlines() if "," in l)}
        # CUDA_VISIBLE_DEVICES renumbers the devices we see, but nvidia-smi reports the real index.
        ours = uuids.get(vis.split(",")[0]) if vis else None
        if ours is None:
            return -1
        apps = _sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
        mine = str(os.getpid())
        return sum(1 for l in apps.splitlines() if l.strip().startswith(ours) and l.rsplit(",", 1)[-1].strip() != mine)
    except Exception:
        return -1


def _sh(cmd):
    import subprocess
    return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout


def _operand_series(torch, a, b, build, calls, n):
    """The same arms on `n` distinct operand copies.  Returns (series, copies, keep).

    The batch in `graph_ms` cycles through these, so no call in a batch can leave its operands in
    L2 for the next one and every call starts cold -- which is the only thing that makes a
    batched timing the same measurement as an un-batched one on a shape whose weights fit in the
    129 MiB L2.  See `graph_ms`.

    `copies` is what was actually built, which is the batch the caller must use: if an arm cannot
    be built on a copy, that copy is dropped for EVERY arm rather than kept for some, because a
    series shorter than the batch would silently start reusing operands.

    `keep` exists because `hot_cublas` holds its operands only as raw device pointers.  Drop the
    tensors and those pointers dangle, and cuBLAS then reads whatever the allocator handed out
    next.  The caller has to hold `keep` until the timing is over.
    """
    series = {arm: [fn] for arm, fn in calls.items()}
    keep = []
    if n <= 1:
        return series, 1, keep
    for _ in range(n - 1):
        aa = a.clone() if a.is_contiguous() else a.t().clone().t()
        bb = b.clone() if b.is_contiguous() else b.t().clone().t()
        try:
            more = {arm: build[arm](aa, bb) for arm in series}
        except Exception:
            torch.cuda.empty_cache()
            break
        keep += [aa, bb]
        for arm, fn in more.items():
            series[arm].append(fn)
    return series, min(len(v) for v in series.values()) if series else 1, keep


def measure_shape(torch, M, N, K, kind, tags, out, flush, opts):
    """Run all four arms on one shape and stream ONE RECORD PER ARM into `out`.  Never raises.

    `tags` is copied onto every record, so a caller decides what identifies its rows: this step
    passes the family and the shape id, `gemm.perf.static` passes the model and the layer.  `out`
    is a `writer()` handle; each record is written and flushed as soon as it is computed, so an
    interruption loses one arm and not a shape.

    The order is: resolve the cuBLAS plan, compile the two torch arms, build
    gb300_accelerated, search bitequiv_autotuning, byte-check those two and torch_auto over
    `opts["draws"]` input draws, restore the first draw's data,
    and only then measure.  The byte check comes before the timing on purpose -- a row whose
    `bit_ok` is short of `bit_total` is a bug report and its timings mean nothing, because the
    arms are then not computing the same thing.

    Returns {phase: seconds} so a caller can see where a long shape went.
    """
    from bitequiv.cublas_match import cublas_equivalent_gemm
    from bitequiv.cublas_match import ltapi as L
    from bitequiv.cublas_match.errors import CublasUnsupportedShape
    from bitequiv.cublas_match.gemm import _resolve
    from bitequiv.cublas_match.ltapi import _kind_of

    torch.cuda.empty_cache()  # the previous shape's operands are out of scope by now
    spent, clock = {}, time.time()

    def phase(name):
        nonlocal clock
        spent[name] = round(time.time() - clock, 1)
        clock = time.time()

    out_dtype = torch.float16
    base = dict(tags)
    base.update({"M": M, "N": N, "K": K, "dtype": kind, "worker": opts["worker"]})
    seed = M * 1000003 + N * 10007 + K
    draws = opts["draws"]

    others_before = others_on_our_gpu()

    def emit(arm, **kw):
        rec = dict(base)
        rec["arm"] = arm
        rec.update(kw)
        # Non-zero means somebody else was on this GPU while the row was measured, so its timings
        # are not this kernel's. -1 means we could not tell. Either way the row needs re-taking.
        rec["contended"] = max(others_before, others_on_our_gpu())
        out.write(json.dumps(rec) + "\n")
        out.flush()

    def bail(msg):
        for arm in ARMS:
            emit(arm, error=msg[:300])
        return spent

    # -- operands, held for the whole shape so the cuBLAS closure's pointers stay valid ---------
    try:
        a, b = make_inputs(torch, M, N, K, kind, 0, seed)
    except Exception as e:
        return bail(f"inputs: {type(e).__name__}: {e}")
    try:
        hot = hot_cublas(L, torch, a, b, kind, out_dtype)
    except Exception as e:
        # No cuBLAS algorithm means there is nothing to be bit-identical to, so the whole row is
        # out of scope rather than broken. It is a decline, and it is kept: dropping it would
        # quietly bias the family it came from.
        msg = str(e)
        for arm in ARMS:
            emit(arm, **({"declined": msg[:300]} if "no algorithm" in msg else {"error": f"cublas: {msg}"[:300]}))
        return spent

    plan, declined = None, ""
    try:
        plan = _resolve(a, b, _kind_of(a), out_dtype)
        base["mode"] = plan.mode
    except CublasUnsupportedShape as e:
        declined = str(e)
    except Exception as e:
        declined = f"{type(e).__name__}: {e}"

    # An arm is kept as a BUILDER, `build[arm](x, y) -> a callable of no arguments`, not as one
    # closure over `a` and `b`. The timing needs the same arm on one operand copy per call in the
    # batch, and rebuilding it on other operands is the only way to get that; `calls` below is
    # just the builder applied to the shape's own operands, which is what the byte check uses.
    build, extra = {"cublas": lambda x, y: hot_cublas(L, torch, x, y, kind, out_dtype).run}, {arm: {} for arm in ARMS}
    calls = {"cublas": hot.run}

    # -- the two torch arms ------------------------------------------------------------------------
    import torch._dynamo
    torch._dynamo.reset()  # once per shape, so the two arms below keep their own caches
    for arm, backends in (("torch_auto", "ATEN,TRITON"), ("torch_triton", "TRITON")):
        try:
            fn, pick, cfg = compile_torch_arm(torch, a, b, kind, out_dtype, backends)
            build[arm] = lambda x, y, f=fn: (lambda: f(x, y))
            calls[arm], extra[arm] = build[arm](a, b), {"pick": pick, "config": cfg}
        except TorchCannotExpress as e:
            extra[arm] = {"declined": str(e)[:300]}
        except Exception as e:
            extra[arm] = {"error": f"{type(e).__name__}: {e}"[:300]}
    phase("torch")

    # -- gb300_accelerated ---------------------------------------------------------------------------------
    if plan is None:
        extra["ours"] = {"declined": declined}
    else:
        try:
            cublas_equivalent_gemm(a, b, out_dtype)
            build["ours"] = lambda x, y: (lambda: cublas_equivalent_gemm(x, y, out_dtype))
            calls["ours"] = build["ours"](a, b)
        except Exception as e:
            extra["ours"] = {"error": f"{type(e).__name__}: {e}"[:300]}

    # -- bitequiv_autotuning: the search ----------------------------------------------------------------------
    ranked = []
    if plan is None:
        extra["pre_search"] = {"declined": declined}
    else:
        try:
            ranked, extra["pre_search"] = search_pre_gb300(torch, a, b, plan, kind, out_dtype, flush,
                                                           opts["max_configs"], opts["search_s"], opts["refine"])
        except Exception as e:
            extra["pre_search"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    phase("search")

    # -- the byte check, which also picks bitequiv_autotuning's winner ------------------------------------------
    a0 = a.clone()
    b0 = b.clone() if b.is_contiguous() else b.t().clone().t()

    def bit_pass(fns):
        """`draws` draws against cuBLAS.  Even draws ordinary gaussian, odd draws with the
        exponents spread across the dtype's range -- with narrow exponents almost every
        regrouping rounds to the same bits, so an ordinary-input-only check proves very little."""
        okc = {k: 0 for k in fns}
        for r in range(draws):
            aa, bb = make_inputs(torch, M, N, K, kind, r, seed)
            a.copy_(aa)
            b.copy_(bb)
            del aa, bb
            if hot.run() != 0:
                return None
            torch.cuda.synchronize()
            ref = digest(torch, hot.out)
            for k, f in fns.items():
                try:
                    okc[k] += 1 if digest(torch, f()) == ref else 0
                except Exception:
                    pass
        return okc

    checks = {k: calls[k] for k in ("ours", "torch_auto") if k in calls}
    fallback, win = 0, None
    for i, (_, cfg) in enumerate(ranked[:6]):
        fn = pre_gb300_launcher(plan.mode, plan, cfg, kind)
        trial = dict(checks)
        trial["pre_search"] = lambda f=fn: f(a, b, out_dtype, 1.0)
        okc = bit_pass(trial)
        if okc is None:
            break
        # gb300_accelerated and torch_auto are checked once; a further pass only re-checks
        # bitequiv_autotuning.
        if checks:
            for k in checks:
                extra[k].update(bit_ok=okc[k], bit_total=draws)
            checks = {}
        extra["pre_search"].update(bit_ok=okc["pre_search"], bit_total=draws)
        if okc["pre_search"] == draws:
            win, fallback = fn, (1 if i else 0)
            extra["pre_search"]["config"] = config_str(cfg)
            break
        fallback = 1
    if checks:  # every bitequiv_autotuning candidate failed, so gb300_accelerated and torch_auto were never checked
        okc = bit_pass(checks)
        for k in (checks if okc else {}):
            extra[k].update(bit_ok=okc[k], bit_total=draws)
    if win is not None:
        build["pre_search"] = lambda x, y, f=win: (lambda: f(x, y, out_dtype, 1.0))
        calls["pre_search"] = build["pre_search"](a, b)
    elif ranked:
        extra["pre_search"].setdefault("error", "no configuration in the space was byte-identical")
    extra["pre_search"]["fallback"] = fallback

    a.copy_(a0)
    b.copy_(b0)
    del a0, b0
    hot.run()
    torch.cuda.synchronize()
    phase("bits")

    # -- the five timings -------------------------------------------------------------------------
    # How many calls go into one graph, and the floor at that batch. Both floors are kept: an arm
    # whose batched capture failed was timed one call at a time and must be judged against the
    # one-call floor, not against a floor it never paid.
    el = 1 if kind == "fp8" else 2
    fl = {1: floor_ms(torch, M, N, flush, reps=6)}
    want = pick_batch(torch, list(calls.values()), flush, fl[1], bytes_per_call=(M * K + N * K) * el + M * N * 4)
    series, n, keep = _operand_series(torch, a, b, build, calls, want)
    if n > 1:
        fl[n] = floor_ms(torch, M, N, flush, batch=n)
    res = time_arms(torch, series, flush, opts["rounds"], batch=n)
    for arm in ARMS:
        rec = dict(extra[arm])
        rec.update(res.get(arm, {}))
        rec["floor_ms"] = fl.get(rec.get("batch_n", 1))
        captured = 1 if rec.get("device_flush_ms") is not None else 0
        if arm in calls:
            rec["captured"] = captured
            if not captured and not rec.get("error"):
                rec["error"] = "[NOT CAPTURED]"
        d, f = rec.get("device_flush_ms"), rec.get("floor_ms")
        rec["near_floor"] = 1 if (d and f and d < 3 * f) else 0
        emit(arm, **rec)
    phase("time")
    calls.clear()
    series.clear()
    del keep
    return spent


# --------------------------------------------------------------------------------------------
# Work claiming.  One shape at a time through an O_EXCL lock with a lease, so adding or removing
# a worker mid-run neither loses nor duplicates a shape.
# --------------------------------------------------------------------------------------------


def _claims_dir():
    d = os.path.join(CACHE, "gemm.perf.random.claims")
    os.makedirs(d, exist_ok=True)
    return d


def _try_claim(idx, me, lease_s):
    """Take shape `idx` if nobody holds it.  A claim older than the lease is stale -- its worker
    was killed -- and may be stolen, which is why a claim is refreshed as the shape runs."""
    path = os.path.join(_claims_dir(), str(idx))
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        try:
            if time.time() - os.path.getmtime(path) < lease_s:
                return False
            os.utime(path, None)  # take the lease first, so two thieves cannot both win
            with open(path, "w") as f:
                f.write(f"{me} stolen {time.strftime('%H:%M:%S')}\n")
            return True
        except OSError:
            return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{me} {time.strftime('%H:%M:%S')}\n")
    return True


def _touch_claim(idx):
    try:
        os.utime(os.path.join(_claims_dir(), str(idx)), None)
    except OSError:
        pass


def _done_keys():
    """(family, dtype, M, N, K, arm) already on disk, so a restart skips finished work."""
    path = os.path.join(CACHE, "gemm.perf.random.jsonl")
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue  # a torn last line from a kill; that shape is simply redone
            done.add((r.get("family"), r.get("dtype"), r.get("M"), r.get("N"), r.get("K"), r.get("arm")))
    return done


def _pause_path():
    return os.path.join(CACHE, "gemm.perf.random.PAUSE")


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


def _geo(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def _rows():
    path = os.path.join(CACHE, "gemm.perf.random.jsonl")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def _dist(vals, top=4):
    c = {}
    for v in vals:
        c[v or "-"] = c.get(v or "-", 0) + 1
    return " ".join(f"{k}:{n}" for k, n in sorted(c.items(), key=lambda kv: -kv[1])[:top])


# Every arm against ONE baseline, cuBLAS, as cuBLAS time over that arm's time, so above 1 means
# faster than cuBLAS. Ratios BETWEEN arms are deliberately not reported. They hide the levels --
# one arm over another says nothing about whether either is any good against the reference -- and
# where two arms differ in more than one way they invite a causal reading this measurement cannot
# support. Four numbers on one baseline let a reader form whatever comparison they want.
BASELINE = "cublas"
VS_BASELINE = ("torch_auto", "torch_triton", "pre_search", "ours")


def _report(meta, timing="device_flush_ms"):
    """One block per (family, dtype), then the same grouped by the resolved plan mode.

    One row per (group, arm), so the four numbers on the cuBLAS baseline sit under each other
    and the levels are visible; an arm per column hid them.
    """
    rows = _rows()
    by_shape = {}
    for r in rows:
        key = (r.get("family"), r.get("dtype"), r.get("M"), r.get("N"), r.get("K"))
        by_shape.setdefault(key, {})[r.get("arm")] = r
    out = []
    P = out.append
    P("")
    want = (meta or {}).get("per_cell", 0) * len(FAMILIES) * 2
    state = f"of {want} " if want and len(by_shape) < want else ""
    P(f"gemm.perf.random -- {len(by_shape)} {state}shapes measured, {len(rows)} arm records" +
      ("  (RUN STILL IN PROGRESS)" if state else ""))
    P(f"  written {time.strftime('%Y-%m-%d %H:%M:%S')}. Regenerate with PERF_RANDOM_REPORT_ONLY=1.")
    for k, v in ARM_LABEL.items():
        P(f"    {ARM_NAME[k]:20} {v}")
    P("    Every arm is against cuBLAS and against nothing else, as cuBLAS time over that arm's")
    P("    time, SO ABOVE 1 MEANS FASTER THAN CUBLAS. bitequiv_autotuning and gb300_accelerated")
    P("    are both byte-identical to cuBLAS and differ in how they got there, so a comparison")
    P("    between them covers two things at once: the kernel and how its configuration is")
    P("    chosen. gb300_accelerated against a torch arm is a product comparison, NOT the price")
    P("    of bit-exactness -- they are different kernels, ours with TMA descriptors, a")
    P("    persistent grid and warp specialization that Inductor's mm_template does not have.")
    if meta:
        P(f"  shape list: seed {meta.get('seed')}, {meta.get('per_cell')} per (family, dtype) cell, "
          f"{meta.get('drawn')} draws, {meta.get('rejected')} rejected by the size caps "
          f"({meta.get('max_tflop')} TFLOP, {meta.get('max_out_elems')} output elements)")
    P(f"  timing column: {timing}")

    def block(title, groups):
        P("")
        P(f"  {title}")
        P(f"    {'group':22} {'arm':20} {'n':>4} {'x cuBLAS geo':>12} {'[p25':>6} {'med':>6} "
          f"{'p75]':>6} {'worst':>6} {'best':>6} {'us/call geo':>11} {'byte-identical':>16} "
          f"{'near floor':>10}")
        for gname, shapes in sorted(groups.items()):
            rat = {k: [] for k in VS_BASELINE}
            tms = {k: [] for k in ARMS}
            bits = {"ours": [0, 0], "pre_search": [0, 0], "torch_auto": [0, 0]}
            near = {k: 0 for k in ARMS}
            modes, picks, batches, nc, ndec = [], [], [], 0, 0
            for arms in shapes:
                t = {k: (arms.get(k) or {}).get(timing) for k in ARMS}
                for k in VS_BASELINE:
                    if t.get(k) and t.get(BASELINE):
                        rat[k].append(t[BASELINE] / t[k])
                for k in ARMS:
                    r = arms.get(k) or {}
                    if t.get(k):
                        tms[k].append(t[k] * 1e3)
                    near[k] += 1 if r.get("near_floor") else 0
                    if k in bits and r.get("bit_total"):
                        bits[k][0] += int(r.get("bit_ok") or 0)
                        bits[k][1] += int(r["bit_total"])
                modes.append(next((a.get("mode") for a in arms.values() if a.get("mode")), None))
                picks.append((arms.get("torch_auto") or {}).get("pick"))
                batches.append(next((a.get("batch_n") for a in arms.values() if a.get("batch_n")), None))
                nc += 1 if any((arms.get(k) or {}).get("captured") == 0 for k in ARMS) else 0
                ndec += 1 if (arms.get("ours") or {}).get("declined") else 0

            def num(x, w=6):
                return f"{x:{w}.3f}" if x is not None else f"{'-':>{w}}"

            for k in ARMS:
                v = rat.get(k, [])
                o, tot = bits[k] if k in bits else (0, 0)
                P(f"    {gname:22} {ARM_NAME[k]:20} {len(shapes):4d} {num(_geo(v), 12)} "
                  f"{num(_pct(v, .25))} {num(_pct(v, .5))} {num(_pct(v, .75))} "
                  f"{num(min(v) if v else None)} {num(max(v) if v else None)} "
                  f"{num(_geo(tms[k]), 11)} {(f'{o}/{tot}' if tot else '-'):>16} {near[k]:10d}")
            notes = [f"calls per graph {_dist(batches)}", f"plan modes {_dist(modes)}", f"torch picks {_dist(picks)}"]
            if nc:
                notes.append(f"uncaptured arm {nc}")
            if ndec:
                notes.append(f"declined {ndec}")
            P(f"    {'':22} {'; '.join(notes)}")

    fam = {}
    for (f_, d_, *_), arms in by_shape.items():
        fam.setdefault(f"{f_}/{d_}", []).append(arms)
    block("by sampling family and dtype", fam)
    mod = {}
    for (_, d_, *_), arms in by_shape.items():
        m = next((a.get("mode") for a in arms.values() if a.get("mode")), "declined")
        mod.setdefault(f"{m}/{d_}", []).append(arms)
    block("by the cuBLAS plan mode the shape actually resolved to", mod)
    P("")
    P("  x cuBLAS is cuBLAS time over that arm's time, so above 1 means faster than cuBLAS.")
    P("  byte-identical = draws byte-identical to cuBLAS / draws compared, summed over the group.")
    P("  A group whose bitequiv_autotuning or gb300_accelerated is short of the total is a bug")
    P("  report, not a measurement: those two arms must be byte-identical or their times mean")
    P("  nothing. torch is unconstrained and is not expected to match.")
    P("  near-floor rows are mostly launch overhead, so every ratio on them is pulled toward 1.")
    P("  calls per graph is `batch_n`: how many calls were captured into one graph, each on its")
    P("  own operand copy, with the replay divided by that. `-` is a row taken before the batch")
    P("  existed, when a 6.9 us replay floor was added whole to every call; on a small shape a")
    P("  `-` row and a batched row are not the same measurement and the `-` one reads slower.")
    P("")
    P("  Watching, pausing and resuming this run")
    P("    watch     tail -f artifact_eval/run_perf_random.gpu2.log      (and .gpu3.log)")
    P("    progress  PERF_RANDOM_REPORT_ONLY=1 PYTHONPATH=$(git rev-parse --show-toplevel) \\")
    P("                python artifact_eval/artifact.py --run gemm.perf.random")
    P(f"    pause     touch {_pause_path()}")
    P("    resume    rm that file, then re-run the same command. Resuming IS re-running it:")
    P("              a finished (family, dtype, M, N, K, arm) is skipped, and a shape in flight")
    P("              when a worker died is picked up again once its claim goes stale.")
    P("    workers   start or stop one at any time, on any free GPU. Work is claimed one shape")
    P("              at a time, not sliced up front, so nothing is lost and nothing is repeated.")
    return out


# --------------------------------------------------------------------------------------------


def run(args, env):
    import torch
    from bitequiv.cublas_match import cublaslt_version, set_cublaslt

    opts = measurement_options(args)
    only = [x for x in os.environ.get("PERF_RANDOM_FAMILIES", "").split(",") if x]
    txt = os.environ.get("PERF_RANDOM_REPORT_TXT", "")

    def show(meta):
        lines = _report(meta)
        print("\n".join(lines))
        if txt:
            with open(txt, "w") as f:
                f.write("\n".join(lines) + "\n")

    if os.environ.get("PERF_RANDOM_REPORT_ONLY"):
        meta = json.load(open(_shapes_path()))["meta"] if os.path.exists(_shapes_path()) else {}
        return show(meta)

    set_cublaslt(os.environ.get("PERF_RANDOM_CUBLASLT", "13.1.1") or None)
    ltver = ".".join(map(str, cublaslt_version()))
    shapes, meta = _load_or_make_shapes(args.seed, opts["per_cell"], args.max_bytes)
    if only:
        shapes = [s for s in shapes if s["family"] in only]
    worker = opts["worker"]

    print(f"\n[gemm.perf.random] {len(shapes)} shapes, matching cuBLASLt {ltver}, worker {worker}")
    print(f"  shape list  {_shapes_path()}")
    print(f"  records     {os.path.join(CACHE, 'gemm.perf.random.jsonl')}")
    print(f"  pause with  touch {_pause_path()}")
    print(f"  bitequiv_autotuning: space cap {opts['max_configs'] or 'none'}, {opts['search_s']}s budget, "
          f"top {opts['refine']} re-timed; {opts['rounds']} rounds, {opts['draws']} byte-check draws")
    if opts["max_configs"]:
        print("  WARNING: bitequiv_autotuning's space is BOUNDED in this run. Every row it touched records both "
              "`space` and `searched`, so the bound is visible in the data.")

    out = writer("gemm.perf.random")
    flush = make_flush_buffer(torch)
    deadline = time.time() + args.minutes * 60 if args.minutes and args.minutes > 0 else None
    n, stop, idle = 0, False, 0
    # Sweep the list until nothing is left. One pass skips a shape another worker holds, so a
    # worker that was killed mid-shape would strand it if the pass never came back -- the second
    # pass takes it once its claim goes stale. An idle pass waits for that to happen; enough idle
    # passes in a row and the remaining shapes belong to workers that are still alive, so stop.
    while not stop:
        before, done = n, _done_keys()
        for spec in shapes:
            if os.path.exists(_pause_path()):
                print("  PAUSE sentinel present; stopping cleanly. Remove it and re-run to resume.")
                stop = True
                break
            if deadline and time.time() > deadline:
                print("  --minutes budget reached; stopping cleanly.")
                stop = True
                break
            if opts["limit"] and n >= opts["limit"]:
                print("  PERF_RANDOM_LIMIT reached; stopping cleanly.")
                stop = True
                break
            key = (spec["family"], spec["dtype"], spec["M"], spec["N"], spec["K"])
            if all((*key, arm) in done for arm in ARMS):
                continue
            if not _try_claim(spec["shape_id"], worker, opts["lease_s"]):
                continue
            tags = {k: spec[k] for k in ("family", "shape_id", "align", "source")}
            tags["cublaslt"] = ltver
            t0 = time.time()
            _touch_claim(spec["shape_id"])
            try:
                spent = measure_shape(torch, spec["M"], spec["N"], spec["K"], spec["dtype"], tags, out, flush,
                                      opts) or {}
            except KeyboardInterrupt:
                raise
            except Exception as e:
                spent = {"failed": round(time.time() - t0, 1)}
                for arm in ARMS:
                    rec = {
                        **tags, "M": spec["M"], "N": spec["N"], "K": spec["K"], "dtype": spec["dtype"], "worker":
                        worker, "arm": arm, "error": f"{type(e).__name__}: {e}"[:300]
                    }
                    out.write(json.dumps(rec) + "\n")
                out.flush()
            n += 1
            shape = f"{spec['M']}x{spec['N']}x{spec['K']}"
            print(
                f"  [{n}] {spec['family']:9} {spec['dtype']:4} {shape:<22} {time.time() - t0:6.1f}s  "
                f"{' '.join(f'{k}={v}' for k, v in spent.items())}", flush=True)
            _touch_claim(spec["shape_id"])
        if n == before:
            idle += 1
            left = sum(1 for s in shapes
                       if not all((s['family'], s['dtype'], s['M'], s['N'], s['K'], arm) in done for arm in ARMS))
            if stop or not left or idle > 12:
                break
            print(
                f"  {left} shapes left, all claimed by another worker; waiting 5 min for a"
                f" claim to go stale (idle pass {idle}/12)", flush=True)
            time.sleep(300)
        else:
            idle = 0
    out.close()
    show(meta)
