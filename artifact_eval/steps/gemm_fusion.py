"""gemm.fusion -- for an epilogue cuBLAS cannot fuse, how much faster is our bit-exact GEMM with
the epilogue folded in than the two-kernel path a determinism user gets today?

Four arms, one baseline, real model shapes.

    cublas_unfused            cuBLAS through the hot closure, then the epilogue as its own
                              kernel.  THE BASELINE AND THE BIT REFERENCE -- it is what a user
                              who needs cuBLAS's bytes can reach today.
    torch_fused               what `torch.compile(mode="max-autotune-no-cudagraphs")` produces
                              for the same expression, with no numerics requirement at all.  What
                              its autotuner picked is recorded; picking the unfused extern is a
                              result, not a failure.
    bitequiv_autotuner_fused  the bit-matching kernel as it was BEFORE the sm_103 rewrites, with
                              the epilogue folded in, at a configuration chosen by a search under
                              the bit constraint.
    gb300_fused               the GB300 bit-matching kernel -- TMA, a persistent grid, warp
                              specialization, the shipped fitted tile rule -- with the epilogue
                              folded in.

Every arm is reported as its speed RELATIVE TO `cublas_unfused`: baseline time over the arm's
time, so above 1 means faster than the baseline.  Ratios between two non-baseline arms are
deliberately not reported -- they hide the level both arms sit at, and the quantity a reader wants
is what a user gains or loses against what they have.

WHY THE LAST TWO ARMS EXIST
---------------------------
They are bit-equivalent to cuBLAS BY CONSTRUCTION, because the cuBLAS plan defines the addition
order and the kernel implements that order.  `torch_fused` cannot be made bit-exact by any
mechanism torch offers.  An earlier attempt patched the epilogue text of Inductor's generated
kernel, and that only worked because Inductor's mm template happens to satisfy the same mainloop
contract as plan mode `plain` -- which holds on 361 of the 405 model cases and fails on the 17
this step is built around.

WHAT MAKES A FUSED ARM EQUAL TO THE BASELINE
--------------------------------------------
The unfused path writes the GEMM's output to memory in the output dtype and reads it back, so a
fused kernel has to round the accumulator through that dtype before the epilogue sees it:

    c = acc.to(out_dtype)            # the round cuBLAS did when it wrote to memory
    y = epilogue(c.to(tl.float32))   # the epilogue in fp32, as torch eager does it
    store(y.to(out_dtype))

The epilogue spellings are written neither here nor in the kernels.  They are
`fusion_oracle/inductor_kernel.EPI_SRC`, verified over their complete input domain --
`fusion_oracle/probe_pairs.txt` has the counts -- and spliced into the kernel templates as text,
so that there is one copy of them and it is the verified one.  Two traps recorded there bite here
too: `tl.minimum`/`tl.maximum` return the non-NaN operand where `torch.clamp` keeps the NaN, and a
`.to(fp16)` does not survive the compiler unless the launch passes `enable_fp_fusion=False`.

THE SHAPES, AND THE SELECTION RULE
----------------------------------
`fusion_moe/models_2026.py` holds 405 reachable (model, layer, epilogue) cases, read from each
model's own config.json on Hugging Face on 2026-08-16.  On this machine cuBLAS resolves 361 to
plan mode `plain`, 17 to `split`, and declines the other 27 (their output dtype is fp8).  A
361-row `plain` table would be one answer repeated, so:

  * ALL 17 `split` cases are measured.  They are the only cases in the table where cuBLAS does
    not pick `plain`, and so the only ones where the bit constraint reaches past the epilogue
    into the mainloop.  15 are an MoE expert `up_proj`, 2 a dense `mlp.down_proj`.
  * about 31 `plain` cases, one per (model, layer group), taking models in the order of the
    ranking fact `models_2026.py` records for each -- trending position first, then 30-day
    downloads descending -- and within a model taking the layers in the order they cost: routed
    expert FFN, dense FFN, attention output projection, LoRA merge.

`fusion_oracle/fusion_cases.py` is that rule written out at length, together with what it leaves
out on purpose.  A reviewer pruning rows should read it first: this list is already a pruning, and
the axis it pruned hardest -- routing balance and token count -- is the one it says least about.

Both dtypes.  Every case runs at fp16 and at bf16: the models ship in bf16, and a cuBLAS plan is a
function of the dtype, so the mode is resolved per (case, dtype) rather than assumed from one of
them.  The exhaustive evidence behind the epilogue spellings was taken at fp16; a bf16 row rests
on this step's own byte check, which is why the dtype is on every row.

ONE ROW PER CASE
----------------
A record holds one (case, dtype) with all four arms in it, so a reviewer prunes a case by deleting
one line.  Nothing is stored aggregated: every geometric mean in the report is computed at print
time from whatever rows are on disk.

OPTIONS, all environment variables
----------------------------------
    FUSION_DTYPES     fp16,bf16    which dtypes to run
    FUSION_MAX_PLAIN  35           cap on the `plain` half of the selection (does not bind today)
    FUSION_DRAWS      10           input draws for the byte check; odd draws spread the exponents
    FUSION_SEARCH_S   45           seconds of configuration search per (case, dtype)
    FUSION_ROUNDS     3            timing rounds, best of
    FUSION_ONLY       ""           substring filter on "<model> <layer>", to run one case
    FUSION_SKIP       ""           comma-separated arms to skip, e.g. torch_fused
    FUSION_REDO       0            1 to re-measure rows already on disk
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time

from ._common import CACHE, digest, graph_ms, hot_cublas, make_inputs, pick_batch, writer

NAME = "gemm.fusion"
ORDER = 40
DESCRIPTION = "a byte-identical fused epilogue against cuBLAS-plus-a-kernel, on real model layers"
IMPLEMENTED = True

BASELINE = "cublas_unfused"
ARMS = ("cublas_unfused", "torch_fused", "bitequiv_autotuner_fused", "gb300_fused")

ARM_LABEL = {
    "cublas_unfused": "cuBLAS, then the epilogue as its own kernel. THE BASELINE AND THE BIT REFERENCE",
    "torch_fused": "torch.compile max-autotune, no numerics requirement; `torch_pick` says what it chose",
    "bitequiv_autotuner_fused": "bit-exact: the kernel BEFORE the sm_103 rewrites, fused, configuration searched",
    "gb300_fused": "bit-exact: the GB300 rewrites and the shipped fitted tile rule, fused",
}

_TIMINGS = [
    ("{a}_ms", "float", "device time per call for {a}: `batch_n` calls captured into one CUDA "
     "graph, each on its own operand copy, L2 flushed between replays, median replay divided by "
     "`batch_n`, best of `FUSION_ROUNDS` rounds"),
    ("{a}_warm_ms", "float", "the same replay for {a} without the L2 flush"),
    ("{a}_e2e_ms", "float", "an ordinary call of {a}, wall clock, no graph: what a caller waits"),
    ("{a}_host_ms", "float", "CPU time to issue {a}, no synchronise"),
]

_COLS = [
    ("model", "str", "model the shape comes from"),
    ("layer", "str", "layer within that model"),
    ("pairing", "str", "layer group: moe_up, moe_down, mlp_up, mlp_down, attn, lora"),
    ("epi", "str", "the epilogue that actually follows this GEMM in this model"),
    ("M", "int", "rows"),
    ("N", "int", "columns"),
    ("K", "int", "contraction length"),
    ("dtype", "str", "operand dtype: fp16 or bf16"),
    ("out_dtype", "str", "output dtype, which is also the dtype every epilogue op rounds through"),
    ("mode", "str", "the cuBLAS plan mode THIS (shape, dtype) resolved to, resolved on the "
     "machine the row was taken on and never assumed from another dtype"),
    ("k_chunk", "int", "split-K slice length, for mode=split; empty otherwise"),
    ("selected_by", "str", "why this case is in the list: the rule in "
     "fusion_oracle/fusion_cases.py, applied to this case"),
]
for _a in ARMS:
    for _n, _t, _d in _TIMINGS:
        _COLS.append((_n.format(a=_a), _t, _d.format(a=_a)))
for _a in ARMS:
    if _a != BASELINE:
        _COLS.append(
            (f"{_a}_over_{BASELINE}", "float", f"{BASELINE} time over {_a} time. ABOVE 1 MEANS {_a} IS FASTER THAN THE "
             f"BASELINE. There is deliberately no column comparing two non-baseline arms"))
for _a in ARMS:
    _COLS.append((f"{_a}_bit_ok", "int", f"input draws on which {_a} was byte-identical to "
                  f"{BASELINE}. For the two bit-exact arms this must equal `bit_total`, or their "
                  f"times are not a measurement"))
_COLS += [
    ("bit_total", "int", "input draws compared. Even draws are ordinary gaussian, odd draws "
     "spread the exponents across the dtype's range, which is what makes a changed ORDER of "
     "additions visible"),
    ("batch_n", "int", "calls captured into one CUDA graph for this row, each on its own operand "
     "copy so none warms another's L2. 1 means one call per graph"),
    ("floor_ms", "float", "an empty Triton kernel batched and replayed the same way, per call: "
     "the cost of a replay before any work of ours runs"),
    ("near_floor", "int", "1 when the baseline is under three times `floor_ms`, i.e. the row is "
     "mostly launch overhead and every ratio on it is compressed toward 1"),
    ("torch_pick", "str", "what torch.compile's autotuner produced: aten (the unfused extern), "
     "triton_mm (a fused mm template), decompose_k, or a pointwise/reduction pair"),
    ("torch_kernels", "int", "how many kernels torch.compile's generated code launches"),
    ("bitequiv_autotuner_cfg", "str", "the winning configuration of the searched arm"),
    ("search_space", "int", "configurations in that arm's space"),
    ("search_seen", "int", "how many of them the budget allowed to be screened. A row where this "
     "is below `search_space` covered the front of the order stated in fused_plain.config_space"),
    ("search_bit_rejected", "int", "candidates that ran but were not byte-identical, so were "
     "refused. Zero is the expected answer: the plan fixes the arithmetic and a tile does not"),
    ("gb300_cfg", "str", "the configuration the shipped GB300 rule picked"),
    ("gb300_path", "str", "tma or notma for mode=plain -- whether the operands could carry a "
     "tensor descriptor, decided exactly as the shipped launcher decides it; twopass for "
     "mode=split"),
    ("others_on_gpu", "int", "compute processes on this GPU besides us while the row was timed. "
     "Anything but 0 means the timings on the row are not a measurement; -1 means it could not "
     "be told"),
    ("notes", "str", "why an arm is missing or what went wrong, one entry per problem"),
]

TABLES = {
    "gemm.fusion": {
        "doc":
        "One row per (model layer, dtype). Four arms on the same inputs: cuBLAS plus a separate "
        "epilogue kernel (the baseline and the bit reference), torch.compile with no numerics "
        "requirement, and two bit-exact fused kernels. Every `*_over_cublas_unfused` column is "
        "baseline time over that arm's time, so above 1 means faster than the baseline; there is "
        "no column comparing two non-baseline arms. Shapes are the layer dimensions of the "
        "open-weight models current in August 2026, chosen by the rule written out in "
        "`fusion_oracle/fusion_cases.py`.",
        "cols":
        _COLS,
    },
}

_JSONL = os.path.join(CACHE, "gemm.fusion.jsonl")


# --------------------------------------------------------------------------------------------
# Small local helpers.  The measurement primitives come from `_common`; these are the two the
# perf steps keep private, restated here rather than imported so this step cannot break when a
# step somebody else owns is edited.
# --------------------------------------------------------------------------------------------
def _flush_buffer(torch, mib=256):
    return torch.empty(mib * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)


_NOOP = [None]


def _noop_kernel():
    if _NOOP[0] is None:
        import triton
        import triton.language as tl  # noqa: F401

        @triton.jit
        def noop(X):
            pass

        _NOOP[0] = noop
    return _NOOP[0]


def _floor_ms(torch, M, N, flush, reps=25, batch=1):
    """An empty Triton kernel at a grid of the same order, batched and replayed like an arm: the
    cost of a replay before any work of ours runs."""
    import triton
    grid = (max(1, triton.cdiv(M, 128) * triton.cdiv(N, 128)), )
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    return graph_ms(torch, lambda: _noop_kernel()[grid](x), flush, reps=reps, batch=batch)


def _host_and_e2e_ms(torch, fn, calls=8):
    """(e2e, host): an ordinary call end to end, and only the time to issue it.  The gap is what a
    graph replay hides -- the TMA path builds three tensor descriptors on every call."""
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


def _others_on_our_gpu():
    """Compute processes on our GPU besides us; -1 when it cannot be told.  A row timed beside
    another process is not a measurement, and that has to be on the row rather than assumed."""
    import subprocess
    try:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                                       text=True, timeout=20)
        uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True,
                                        timeout=20).split()
        if not vis.isdigit() or int(vis) >= len(uuids):
            return -1
        mine, me = uuids[int(vis)], os.getpid()
        return sum(1 for ln in apps.strip().splitlines()
                   if ln.strip() and ln.split(",")[1].strip() == mine and int(ln.split(",")[0]) != me)
    except Exception:
        return -1


def _geo(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def _beat(tag, what):
    """A timestamped line at every phase boundary inside a case.

    A worker that computes hard and prints nothing is a known failure mode on this machine -- two
    of them sat at 100% for hours with a silent log.  A case here takes tens of seconds and has
    five phases, so a log that has not moved in a few minutes is a stall and says so, instead of
    looking like a slow shape.
    """
    print(f"        {time.strftime('%H:%M:%S')}  {tag}  {what}", flush=True)


_TORCH_DT = {"fp16": "float16", "bf16": "bfloat16"}


def _dt(torch, name):
    return getattr(torch, _TORCH_DT[name])


def _inputs(torch, M, N, K, dtype_name, rep, seed):
    """The shape's operands for draw `rep`.

    `_common.make_inputs` is the draw every other step uses and it produces fp16; a bf16 row casts
    it, which keeps the same exponent spread and the same random stream.  Even reps are ordinary
    gaussian, odd reps spread the exponents across the range -- with narrow exponents almost any
    regrouping rounds to the same bits and a byte check passes things it should not.
    """
    a, b = make_inputs(torch, M, N, K, "fp16", rep, seed)
    if dtype_name == "bf16":
        a, b = a.to(torch.bfloat16).contiguous(), b.to(torch.bfloat16).contiguous()
    return a, b


# --------------------------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------------------------


def _epi_text(epi, params):
    from inductor_kernel import EPI_SRC
    return [ln.format(**params) if "{" in ln else ln for ln in EPI_SRC[epi]]


def _params_of(case):
    """The scalars this case bakes into the kernel.  `model_cases.params_of` is the authority --
    it reads LoRA's alpha/rank and the clamp out of the model table -- so they are not restated."""
    from cases import params_for
    from model_cases import params_of
    return params_for(case.epi, params_of(case))


def _classify_torch(codes):
    """(pick, kernel count) read off the code Inductor generated.  `aten` is the extern call, i.e.
    the autotuner decided not to fuse the GEMM at all."""
    import re
    blob = "\n".join(codes)
    picks = []
    if "decompose_k" in blob:
        picks.append("decompose_k")
    if re.search(r"extern_kernels\.(mm|addmm|bmm)", blob):
        picks.append("aten")
    if re.search(r"triton_tem_fused|@triton_heuristics\.template", blob):
        picks.append("triton_mm")
    if not picks and re.search(r"triton_red_fused|triton_per_fused|triton_poi_fused", blob):
        picks.append("pointwise_only")
    n = len(re.findall(r"^\s+\w+\.run\(", blob, re.M)) + len(re.findall(r"extern_kernels\.\w+\(", blob))
    return ("+".join(picks) or "unknown"), n


def _compile_torch(torch, epi, params, a, b, extras):
    """arm `torch_fused`: compile the whole expression with max-autotune and no numerics
    requirement.  Returns (compiled callable, output, pick, kernel count).

    ONE compiled object serves the whole case.  Every later operand copy has the same shapes and
    dtypes, so it reuses this compilation instead of adding a guard entry -- which matters,
    because dynamo's cache size limit is small and blowing it silently falls back to eager.
    """
    import torch._inductor.config as ic
    from torch._inductor.utils import run_and_get_code

    from cases import EPILOGUES
    fn = EPILOGUES[epi]
    with ic.patch(max_autotune_gemm_backends="ATEN,TRITON"):
        compiled = torch.compile(lambda x, w, *rest: fn(x @ w, *rest, **params), mode="max-autotune-no-cudagraphs")
        out, codes = run_and_get_code(compiled, a, b, *extras)
    torch.cuda.synchronize()
    pick, nkern = _classify_torch(codes)
    return compiled, out, pick, nkern


def _make_baseline(torch, L, epi, kinds, params, a, b, extras, out_dtype, cache_dir, dtype_name):
    """arm `cublas_unfused`: cuBLAS through the hot closure, then the epilogue as one kernel.

    The epilogue kernel comes from the same `EPI_SRC` text the fused kernels are spliced with, via
    `inductor_kernel.standalone`, so the baseline and the fused arms cannot drift apart in the
    arithmetic -- only in where the value came from.  Here it comes out of memory already at the
    output dtype, so the leading round in every chain is a no-op, and that is exactly what makes
    this arm the reference the others have to match.

    `round_lines` is what points the epilogue's rounds at THIS row's dtype.  The fused kernels go
    through the same call, so the baseline and the arms it is the reference for cannot round
    through different dtypes -- which, when it happened, made every configuration of every fused
    arm come out byte-different and read like a broken kernel.

    Returns (callable, the epilogue's output, the hot closure).  The closure must be kept alive:
    it holds its operands as raw device pointers only.
    """
    from inductor_kernel import EPI_SRC, launch_standalone, standalone
    from bitequiv.cublas_match.fused_plain import round_lines
    from bitequiv.cublas_match.ltapi import _kind_of
    M, N = a.shape[0], b.shape[1]
    hot = hot_cublas(L, torch, a, b, _kind_of(a), out_dtype)
    fn, form = standalone(epi, kinds, params, cache_dir=cache_dir, body=round_lines(EPI_SRC[epi], dtype_name))
    y = torch.empty(M, N, device="cuda", dtype=out_dtype)
    ex = list(extras)

    def call():
        hot.run()
        launch_standalone(fn, form, hot.out, ex, y)

    return call, y, hot


def _screen_ms(torch, call, iters=3):
    """One launch to compile it, then `iters` back-to-back launches timed with events.  For
    RANKING only: it measures the same launch overhead for every candidate, so the order it
    produces is usable, and the top of that order is re-timed properly."""
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    call()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        call()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _search(torch, make_call, cfgs, flush, floor, budget_s, refine=12):
    """Screen the space, re-time the best `refine` properly, return them fastest first.

    Nothing here judges correctness.  The caller byte-checks the winner against the baseline and
    walks down the ranking until one is byte-identical, so the bit constraint is enforced on the
    configuration that is actually used and a pruner never marks its own work.

    The budget exists because the sweep is dominated by Triton COMPILATION rather than by running
    the kernels; it bounds the first case of each new kind without bounding the rest, since the
    on-disk cache serves the repeats.  `search_space` and `search_seen` are on every row so a
    truncated search cannot be read as an exhaustive one.  For scale: Inductor's own mm autotune
    considers 55 candidates, and the shipped launch is the first thing screened here, so the arm
    can never come out slower than no search at all.
    """
    info = {"space": len(cfgs), "seen": 0}
    screened, stop = [], time.time() + budget_s
    for i, cfg in enumerate(cfgs):
        if i and time.time() > stop:
            info["budget_hit"] = 1
            break
        try:
            screened.append((_screen_ms(torch, make_call(cfg)), cfg))
        except Exception:
            torch.cuda.empty_cache()
            continue
    info["seen"] = len(screened)
    screened.sort(key=lambda x: x[0])
    ranked, nb = [], None
    for _s, cfg in screened[:refine]:
        try:
            call = make_call(cfg)
            if nb is None:
                nb = pick_batch(torch, [call], flush, floor)
            t = graph_ms(torch, call, flush, reps=6, batch=nb)
            if t is not None:
                ranked.append((t, cfg))
        except Exception:
            torch.cuda.empty_cache()
            continue
    ranked.sort(key=lambda x: x[0])
    return ranked, info


def _plan_for(M, N, K, dtype_name, out_dtype):
    """The cuBLAS plan for this (shape, dtype), from cuBLAS's own heuristic.  No kernel is run and
    no data pointer is read: the heuristic query needs the shape and nothing else."""
    from plan_modes import plan_of
    return plan_of(M, N, K, dtype_name, out_dtype)


def _fused_pair(mode, plan, kern, kinds, out_dtype):
    """`(searched(cfg, ...), fit_gb300(...), gb300(cfg, ...))` for one plan mode.

    Each is the corresponding launcher in `bitequiv/cublas_match/kernels.py` with the epilogue
    folded in.  The plan's own parameters -- the chunking -- are read off `plan` and never varied,
    so nothing the search touches can change the arithmetic.

    `fit_gb300` is called once per case and returns the configuration the GB300 arm actually ran
    at: the shipped rule's tile is fitted for a kernel with no epilogue, and on some shapes the
    fused kernel does not fit in shared memory at it.
    """
    from bitequiv.cublas_match import fused_plain as FP
    from bitequiv.cublas_match import fused_split as FS
    from bitequiv.cublas_match.arch import platform
    if mode == "plain":

        def searched(cfg, a, b, extras):
            return FP.launch_pre(kern, a, b, out_dtype, kinds, extras, cfg)

        def fit(a, b, extras):
            want = FP.gb300_config(a.shape[0], b.shape[1])
            used, how, c = FP.fit_gb300(kern, a, b, out_dtype, kinds, extras, want)
            tag = FP.config_str(used) + ("" if used == want else "  stepped down from " + FP.config_str(want))
            return used, how, tag, c

        def gb300(cfg, a, b, extras):
            return FP.launch_gb300(kern, a, b, out_dtype, kinds, extras, cfg)[0]

        return searched, fit, gb300

    min_bm = platform().fp8_min_bm

    def searched(cfg, a, b, extras):
        return FS.launch_pre(kern, a, b, out_dtype, kinds, extras, plan, cfg, fp8_min_bm=min_bm)

    def gb300(cfg, a, b, extras):
        return FS.launch_gb300(kern, a, b, out_dtype, kinds, extras, plan, fp8_min_bm=min_bm)

    def fit(a, b, extras):
        c = gb300(None, a, b, extras)
        if c is None:
            raise RuntimeError("the shipped split launcher declined this shape")
        return None, "twopass", "shipped pass-1 tile rule", c

    return searched, fit, gb300


# --------------------------------------------------------------------------------------------
# one case
# --------------------------------------------------------------------------------------------


def _one_case(torch, L, case, dtype_name, opts, flush, why):
    """Measure one (case, dtype).  Always returns a record: a case that could not be measured
    comes back with `notes` saying why rather than disappearing from the table."""
    from cases import EPI_OPERANDS, EPILOGUES, make_epi_args, out_dtype_for
    from bitequiv.cublas_match import fused_plain as FP
    from bitequiv.cublas_match import fused_split as FS

    M, N, K, epi = case.M, case.N, case.K, case.epi
    dt = _dt(torch, dtype_name)
    params = _params_of(case)
    kinds = EPI_OPERANDS.get(epi, ())
    out_dtype = out_dtype_for(epi, dt)
    rec = {
        "model": case.model, "layer": case.layer, "pairing": case.pairing, "epi": epi, "M": M, "N": N, "K": K, "dtype":
        dtype_name, "out_dtype": str(out_dtype).replace("torch.", ""), "selected_by": why, "notes": ""
    }
    notes = []
    tag = f"{case.model[:14]} {case.pairing} {M}x{N}x{K} {dtype_name}"

    plan, reason = _plan_for(M, N, K, dtype_name, out_dtype)
    if plan is None:
        rec["notes"] = f"declined: {reason}"[:300]
        return rec
    rec["mode"] = plan.mode
    if plan.k_chunk:
        rec["k_chunk"] = plan.k_chunk
    if plan.mode not in ("plain", "split"):
        rec["notes"] = (f"mode {plan.mode} is out of scope: a fused kernel is built only for the "
                        f"plan modes the selected shapes were measured to reach")
        return rec

    _beat(tag, f"mode {plan.mode}, building the fused kernel")
    cache_dir = os.path.join(CACHE, "gemm.fusion.kernels")
    lines = _epi_text(epi, params)
    build_kern = FP.build if plan.mode == "plain" else FS.build
    kern = build_kern(lines, kinds, dtype_name, cache_dir, tag=f"{epi}_{dtype_name}")
    searched, fit_gb300, gb300 = _fused_pair(plan.mode, plan, kern, kinds, out_dtype)

    seed = M * 1000003 + N * 10007 + K
    a, b = _inputs(torch, M, N, K, dtype_name, 0, seed)
    extras = make_epi_args(epi, M, N, dt, seed=seed)

    # -- the baseline, which is also the reference ---------------------------------------------
    try:
        base_call, base_out, hot = _make_baseline(torch, L, epi, kinds, params, a, b, extras, out_dtype, cache_dir,
                                                  dtype_name)
        base_call()
        torch.cuda.synchronize()
    except Exception as e:
        rec["notes"] = f"cublas_unfused: {type(e).__name__}: {e}"[:300]
        return rec

    # The baseline has to be what torch eager computes, or it is not a reference for anything.
    if digest(torch, base_out) != digest(torch, EPILOGUES[epi](hot.out, *extras, **params)):
        notes.append("cublas_unfused is not byte-identical to torch eager on draw 0")
    want0 = digest(torch, base_out)

    # -- the searched arm: rank by speed, then take the fastest byte-identical one --------------
    _beat(tag, "searching the pre-sm_103 configuration space")
    floor1 = _floor_ms(torch, M, N, flush, reps=6)
    win_cfg = None
    if "bitequiv_autotuner_fused" not in opts["skip"]:
        cfgs = (FP.config_space(M, N, K, dtype_name) if plan.mode == "plain" else FS.config_space(
            M, N, K, dtype_name, plan))
        ranked = []
        try:
            ranked, sinfo = _search(torch, lambda c: (lambda: searched(c, a, b, extras)), cfgs, flush, floor1,
                                    opts["search_s"])
            rec["search_space"] = sinfo["space"]
            rec["search_seen"] = sinfo["seen"]
        except Exception as e:
            notes.append(f"bitequiv_autotuner_fused search: {type(e).__name__}: {e}"[:200])
        rejected = 0
        for _t, cfg in ranked:
            try:
                got = searched(cfg, a, b, extras)
            except Exception:
                continue
            if got is not None and digest(torch, got) == want0:
                win_cfg = cfg
                break
            rejected += 1
        rec["search_bit_rejected"] = rejected
        if win_cfg is not None:
            rec["bitequiv_autotuner_cfg"] = FP.config_str(win_cfg)
        elif ranked:
            notes.append("bitequiv_autotuner_fused: no configuration in the space was byte-identical")
        else:
            notes.append("bitequiv_autotuner_fused: no configuration in the space ran")

    # -- the GB300 arm -------------------------------------------------------------------------
    _beat(tag, "fitting the GB300 launch")
    gb_cfg, gb_ok = None, False
    if "gb300_fused" not in opts["skip"]:
        try:
            gb_cfg, how, cfg_str, c = fit_gb300(a, b, extras)
            rec["gb300_path"], rec["gb300_cfg"] = how, cfg_str
            gb_ok = True
            del c
        except Exception as e:
            notes.append(f"gb300_fused: {type(e).__name__}: {e}"[:200])

    # -- the torch arm -------------------------------------------------------------------------
    _beat(tag, "torch.compile max-autotune")
    compiled = None
    if "torch_fused" not in opts["skip"]:
        try:
            compiled, _tout, pick, nkern = _compile_torch(torch, epi, params, a, b, extras)
            rec["torch_pick"], rec["torch_kernels"] = pick, nkern
        except Exception as e:
            notes.append(f"torch_fused: {type(e).__name__}: {e}"[:200])
            compiled = None

    # -- builders: one per arm, so every arm can be rebuilt on a fresh operand copy -------------
    # A builder returns (callable, output, anything that must be kept alive).
    build = {
        BASELINE:
        (lambda x, y, ex: _make_baseline(torch, L, epi, kinds, params, x, y, ex, out_dtype, cache_dir, dtype_name))
    }
    if win_cfg is not None:
        build["bitequiv_autotuner_fused"] = lambda x, y, ex: ((lambda: searched(win_cfg, x, y, ex)), None, None)
    if gb_ok:
        build["gb300_fused"] = lambda x, y, ex: ((lambda: gb300(gb_cfg, x, y, ex)), None, None)
    if compiled is not None:
        build["torch_fused"] = lambda x, y, ex: ((lambda: compiled(x, y, *ex)), None, None)

    # -- the byte check ------------------------------------------------------------------------
    _beat(tag, f"byte check, {opts['draws']} draws")
    draws = opts["draws"]
    ok = {arm: 0 for arm in ARMS}
    ran = {arm: 0 for arm in ARMS}
    for rep in range(draws):
        try:
            aa, bb = _inputs(torch, M, N, K, dtype_name, rep, seed)
            ee = make_epi_args(epi, M, N, dt, seed=seed + rep, wide=(rep % 2 == 1))
            bc, bout, _h = _make_baseline(torch, L, epi, kinds, params, aa, bb, ee, out_dtype, cache_dir, dtype_name)
            bc()
            torch.cuda.synchronize()
            want = digest(torch, bout)
            ok[BASELINE] += 1
            ran[BASELINE] += 1
            for arm in ARMS[1:]:
                if arm not in build:
                    continue
                try:
                    call, _o, _k = build[arm](aa, bb, ee)
                    got = call()
                    ran[arm] += 1
                    if got is not None and digest(torch, got) == want:
                        ok[arm] += 1
                except Exception as e:
                    notes.append(f"{arm} draw {rep}: {type(e).__name__}: {e}"[:120])
            del aa, bb, ee, bout, bc
        except Exception as e:
            notes.append(f"byte check draw {rep}: {type(e).__name__}: {e}"[:150])
    for arm in ARMS:
        if arm in build:
            rec[f"{arm}_bit_ok"] = ok[arm]
            if ran[arm] != draws:
                # `bit_ok` counts agreements, so a draw the arm never ran reads the same as a draw
                # it got wrong. Say which it was rather than letting the count carry both.
                notes.append(f"{arm} ran {ran[arm]} of {draws} draws")
    rec["bit_total"] = draws
    torch.cuda.empty_cache()

    # -- the five timings ----------------------------------------------------------------------
    _beat(tag, "timing the four arms")
    calls, keep = {}, [hot]
    for arm, mkb in build.items():
        try:
            call, _o, h = mkb(a, b, extras)
            calls[arm] = call
            keep.append(h)
        except Exception as e:
            notes.append(f"{arm} build: {type(e).__name__}: {e}"[:150])
    esz = a.element_size()
    per_call = (M * K + N * K) * esz + M * N * esz * (2 + len(kinds)) + M * N * 4
    want_n = pick_batch(torch, list(calls.values()), flush, floor1, bytes_per_call=per_call)
    series, n, hold = _series(torch, M, N, K, dtype_name, seed, epi, dt, build, calls, want_n)
    keep.append(hold)
    rec["batch_n"] = n
    rec["floor_ms"] = floor1 if n == 1 else _floor_ms(torch, M, N, flush, batch=n)
    rec["others_on_gpu"] = _others_on_our_gpu()

    noflush = torch.empty(0, device="cuda", dtype=torch.int32)
    best = {arm: {} for arm in series}
    for _r in range(opts["rounds"]):
        for arm in ARMS:  # arms interleaved inside a round, so a drifting clock hits them alike
            if arm not in series:
                continue
            fns = series[arm] if n > 1 else series[arm][0]
            try:
                for key, val in (("ms", graph_ms(torch, fns, flush, reps=15,
                                                 batch=n)), ("warm_ms", graph_ms(torch, fns, noflush, reps=15,
                                                                                 batch=n))):
                    if val is not None and (best[arm].get(key) is None or val < best[arm][key]):
                        best[arm][key] = val
            except Exception as e:
                notes.append(f"{arm} timing: {type(e).__name__}: {e}"[:150])
    for arm in series:
        try:
            best[arm]["e2e_ms"], best[arm]["host_ms"] = _host_and_e2e_ms(torch, series[arm][0])
        except Exception:
            pass
    for arm, vals in best.items():
        for key, val in vals.items():
            rec[f"{arm}_{key}"] = val

    base = rec.get(f"{BASELINE}_ms")
    for arm in ARMS[1:]:
        t = rec.get(f"{arm}_ms")
        if base and t:
            rec[f"{arm}_over_{BASELINE}"] = base / t
    fl = rec["floor_ms"]
    rec["near_floor"] = 1 if (base and fl and base < 3 * fl) else 0
    rec["notes"] = " | ".join(dict.fromkeys(notes))[:900]
    del series, calls, keep, hold
    torch.cuda.empty_cache()
    return rec


def _series(torch, M, N, K, dtype_name, seed, epi, dt, build, calls, want):
    """The same arms on `want` distinct operand copies, so no call in a batch warms another's L2.

    A copy any arm cannot be built on is dropped for EVERY arm: a series shorter than the batch
    would silently start reusing operands, which is the thing the copies exist to prevent.  `hold`
    keeps the tensors alive -- the hot cuBLAS closure holds its operands as raw pointers only.
    """
    from cases import make_epi_args
    series = {arm: [c] for arm, c in calls.items()}
    hold = []
    if want <= 1:
        return series, 1, hold
    for i in range(1, want):
        try:
            aa, bb = _inputs(torch, M, N, K, dtype_name, 2 * (i + 1), seed + 7919 * i)
            ee = make_epi_args(epi, M, N, dt, seed=seed + 7919 * i)
            fresh = {}
            for arm in series:
                call, _o, h = build[arm](aa, bb, ee)
                fresh[arm] = call
                hold.append(h)
            for arm, call in fresh.items():
                series[arm].append(call)
            hold += [aa, bb, ee]
        except Exception:
            torch.cuda.empty_cache()
            break
    made = min(len(v) for v in series.values()) if series else 1
    for arm in series:
        series[arm] = series[arm][:made]
    return series, made, hold


# --------------------------------------------------------------------------------------------
# the report.  Everything here is computed from the rows on disk at print time, so deleting a
# case's line and re-running gives the same table without it.
# --------------------------------------------------------------------------------------------


def _rows():
    rows = []
    if not os.path.exists(_JSONL):
        return rows
    with open(_JSONL, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue  # a torn last line from a kill; that case is simply redone
    return rows


def _latest(rows):
    """One row per (case, dtype), the most recent measurement winning."""
    seen, out = set(), []
    for r in reversed(rows):
        k = (r.get("model"), r.get("layer"), r.get("M"), r.get("N"), r.get("K"), r.get("dtype"))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    out.reverse()
    return out


def _bit_mark(r, arm):
    v = r.get(f"{arm}_bit_ok")
    if v is None:
        return "-"
    return "." if v == r.get("bit_total") else "x"


def _report(selected=None, dtypes=None):
    """`selected` and `dtypes` are what the run set out to do, so the table can say what it
    covered against what it attempted.  Both are optional: `--export` and a report-only re-read
    have no selection to hand, and then the coverage line reports only what is on disk."""
    lines = []
    P = lines.append
    uniq = _latest(_rows())
    good = [r for r in uniq if r.get(BASELINE + "_ms")]
    P("")
    P("  COVERAGE.  This table is what was measured, not the whole model list. Read the counts")
    P("  before the numbers: a partial sweep and a complete one look the same once averaged.")
    want = len(selected) * len(dtypes) if selected and dtypes else None
    P(f"    rows attempted           {len(uniq)}" + (f" of {want} (case, dtype) rows selected" if want else ""))
    P(f"    rows with all four arms  {sum(1 for r in good if all(r.get(a + '_ms') for a in ARMS))}")
    P(f"    rows with a baseline     {len(good)}   <- a row without one has no ratio at all")
    by_mode = {}
    for r in uniq:
        k = r.get("mode") or "not resolved"
        by_mode.setdefault(k, [0, 0])
        by_mode[k][0] += 1
        by_mode[k][1] += 1 if r.get(BASELINE + "_ms") else 0
    P("    by cuBLAS plan mode      " +
      "   ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by_mode.items(), key=lambda kv: -kv[1][0])))
    P("    layer groups reached     " + "   ".join(f"{p} {sum(1 for r in good if r.get('pairing') == p)}"
                                                   for p in sorted({r.get("pairing")
                                                                    for r in uniq
                                                                    if r.get("pairing")})))
    P("    epilogues reached        " + "   ".join(f"{e} {sum(1 for r in good if r.get('epi') == e)}"
                                                   for e in sorted({r.get("epi")
                                                                    for r in uniq
                                                                    if r.get("epi")})))
    P(f"    rows with a reason recorded against them  {sum(1 for r in uniq if r.get('notes'))}"
      "   (listed at the end; a case that failed is a row with a reason, never an absence)")
    P("")
    P("  EVERY NUMBER BELOW IS SPEED RELATIVE TO cublas_unfused -- baseline time over the arm's")
    P("  time, so ABOVE 1 MEANS FASTER THAN THE BASELINE. cublas_unfused is cuBLAS followed by the")
    P("  epilogue as a second kernel: what a determinism user gets today. No ratio between two")
    P("  non-baseline arms is reported.")
    P("")
    for arm in ARMS:
        P(f"    {arm:26} {ARM_LABEL[arm]}")
    P("")
    P(f"  {'model':20} {'layer':25} {'shape':>21} {'dt':5}{'mode':7}"
      f"{'torch':>8}{'autotnr':>9}{'gb300':>8}  bits  {'base us':>9}")
    P("  " + "-" * 116)
    for r in uniq:
        shape = "{}x{}x{}".format(r.get("M"), r.get("N"), r.get("K"))
        head = f"  {str(r.get('model'))[:20]:20} {str(r.get('layer'))[:25]:25} {shape:>21} " \
               f"{r.get('dtype', ''):5}{r.get('mode') or '-':7}"
        if not r.get(BASELINE + "_ms"):
            P(head + "  not measured: " + (r.get("notes") or "")[:45])
            continue

        def rat(a, w, r=r):
            v = r.get(f"{a}_over_{BASELINE}")
            return (f"{v:.3f}" if v else "-").rjust(w)

        bits = "".join(_bit_mark(r, a) for a in ARMS[1:])
        base_us = r[BASELINE + "_ms"] * 1000
        P(head + rat("torch_fused", 8) + rat("bitequiv_autotuner_fused", 9) + rat("gb300_fused", 8) +
          f"  {bits}  {base_us:9.1f}" + ("*" if r.get("near_floor") else ""))
    P("")
    P("  bits: one character per arm, in the order torch / autotuner / gb300. `.` means every")
    P("        draw was byte-identical to cublas_unfused, `x` at least one was not, `-` the arm")
    P("        did not run. A time from an arm marked `x` in the LAST TWO positions is a bug")
    P("        report, not a speedup. `*` marks a row whose baseline is under three times the")
    P("        replay floor, so every ratio on it is compressed toward 1.")
    P("")
    P("  geometric mean of the speed relative to cublas_unfused, over the rows on disk:")
    P("")
    P(f"  {'group':30}{'rows':>5}{'torch':>10}{'autotuner':>12}{'gb300':>10}")
    P("  " + "-" * 67)
    groups = [("ALL", lambda r: True), ("mode=split", lambda r: r.get("mode") == "split"),
              ("mode=plain", lambda r: r.get("mode") == "plain"), ("dtype=fp16", lambda r: r.get("dtype") == "fp16"),
              ("dtype=bf16", lambda r: r.get("dtype") == "bf16"),
              ("above the replay floor", lambda r: not r.get("near_floor"))]
    for p in sorted({r.get("pairing") for r in good if r.get("pairing")}):
        groups.append((f"layer={p}", lambda r, p=p: r.get("pairing") == p))
    for e in sorted({r.get("epi") for r in good if r.get("epi")}):
        groups.append((f"epilogue={e}", lambda r, e=e: r.get("epi") == e))
    for name, sel in groups:
        sub = [r for r in good if sel(r)]
        if not sub:
            continue
        cells = ""
        for arm, w in (("torch_fused", 10), ("bitequiv_autotuner_fused", 12), ("gb300_fused", 10)):
            v = _geo([r.get(f"{arm}_over_{BASELINE}") for r in sub])
            cells += (f"{v:.3f}" if v else "-").rjust(w)
        P(f"  {name:30}{len(sub):>5}{cells}")
    P("")
    P("  byte check against cublas_unfused, summed over the rows on disk:")
    for a in ARMS:
        n_ok = sum(r.get(f"{a}_bit_ok") or 0 for r in good)
        n_all = sum(r.get("bit_total") or 0 for r in good if r.get(f"{a}_bit_ok") is not None)
        tail = "   <- must be all of them" if a not in (BASELINE, "torch_fused") else ""
        P(f"    {a:26} {n_ok:6} of {n_all:6} draws{tail}")
    picks = {}
    for r in good:
        k = r.get("torch_pick") or "-"
        picks[k] = picks.get(k, 0) + 1
    P("")
    P("  what torch.compile's autotuner produced:  " + "   ".join(f"{k} {v}" for k, v in sorted(picks.items())))
    bad = [r for r in uniq if r.get("notes")]
    if bad:
        P("")
        P(f"  {len(bad)} row(s) with something recorded against them:")
        for r in bad:
            P(f"    {str(r.get('model'))[:18]:18} {str(r.get('layer'))[:22]:22} {r.get('dtype', ''):5} "
              f"{(r.get('notes') or '')[:110]}")
    return lines


# --------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------


def _opts():

    def env(k, d, cast=str):
        v = os.environ.get(k, "")
        return cast(v) if v.strip() else d

    return {
        "dtypes": [d.strip() for d in env("FUSION_DTYPES", "fp16,bf16").split(",") if d.strip()],
        "max_plain": env("FUSION_MAX_PLAIN", 35, int),
        "draws": env("FUSION_DRAWS", 10, int),
        "search_s": env("FUSION_SEARCH_S", 45.0, float),
        "rounds": env("FUSION_ROUNDS", 3, int),
        "only": env("FUSION_ONLY", ""),
        "skip": {s.strip()
                 for s in env("FUSION_SKIP", "").split(",")
                 if s.strip()},
        "redo": env("FUSION_REDO", 0, int),
    }


def _add_paths():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys
    for p in (os.path.join(here, "fusion_oracle"), os.path.join(here, "fusion_moe")):
        if p not in sys.path:
            sys.path.insert(0, p)


def run(args, env):
    import torch
    _add_paths()

    import bitequiv.cublas_match.ltapi as L
    from fusion_cases import select
    from plan_modes import mode_of

    from cases import out_dtype_for
    from model_cases import reachable_cases
    from models_2026 import GROUPS

    # One compiled entry per case, and there are more cases than dynamo's default cache limit.
    # Blowing it does not raise -- it silently stops compiling and runs eager, which would time
    # as a torch arm that never was one.
    import torch._dynamo
    for knob, want in (("cache_size_limit", 1024), ("accumulated_cache_size_limit", 4096)):
        if hasattr(torch._dynamo.config, knob):
            setattr(torch._dynamo.config, knob, max(getattr(torch._dynamo.config, knob), want))

    opts = _opts()
    print(f"\n[{NAME}] fusing an epilogue cuBLAS cannot express, on real model layer dimensions")
    print(f"  dtypes {','.join(opts['dtypes'])}   draws {opts['draws']}   "
          f"search {opts['search_s']:.0f}s/row   rounds {opts['rounds']}")

    # The plan mode is resolved HERE, on this machine, not read from a file: which kernel cuBLAS
    # picks depends on the architecture and the library version, so a shipped mode table would be
    # a claim about somebody else's machine.  The query reads the shape only -- no kernel runs.
    t0 = time.time()
    allc, _skipped = reachable_cases(tuple(GROUPS))
    modes = {}
    for c in allc:
        for d in opts["dtypes"]:
            m, _w = mode_of(c.M, c.N, c.K, d, out_dtype_for(c.epi, _dt(torch, d)))
            modes[(c.M, c.N, c.K, c.epi, d)] = m
    # Counted over CASES, not over the keys of `modes`: two cases of different models can be the
    # same (M, N, K, epilogue), and a tally of the dictionary would silently merge them.
    tally = {}
    for c in allc:
        for d in opts["dtypes"]:
            m = modes[(c.M, c.N, c.K, c.epi, d)] or "declined"
            tally[m] = tally.get(m, 0) + 1
    print(f"  cuBLAS plan modes over {len(allc)} reachable cases x {len(opts['dtypes'])} dtypes, "
          f"resolved in {time.time() - t0:.1f}s")
    print("    " + "   ".join(f"{k} {v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1])))

    chosen, why = select(modes, opts["max_plain"])
    if opts["only"]:
        chosen = [c for c in chosen if opts["only"].lower() in f"{c.model} {c.layer}".lower()]
    n_split = sum(1 for c in chosen if why[c.key].startswith("split"))
    print(f"  selected {len(chosen)} cases: {n_split} split (every one of them) and "
          f"{len(chosen) - n_split} plain -- see fusion_oracle/fusion_cases.py for the rule")

    done = set()
    if not opts["redo"]:
        for r in _rows():
            done.add((r.get("model"), r.get("layer"), r.get("M"), r.get("N"), r.get("K"), r.get("dtype")))
    todo = [(c, d) for c in chosen for d in opts["dtypes"] if (c.model, c.layer, c.M, c.N, c.K, d) not in done]
    print(f"  {len(todo)} of {len(chosen) * len(opts['dtypes'])} (case, dtype) rows still to do\n", flush=True)

    fh = writer("gemm.fusion")
    flush = _flush_buffer(torch)
    started = time.time()
    for i, (case, dtype_name) in enumerate(todo):
        t = time.time()
        try:
            rec = _one_case(torch, L, case, dtype_name, opts, flush, why[case.key])
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec = {
                "model": case.model, "layer": case.layer, "pairing": case.pairing, "epi": case.epi, "M": case.M, "N":
                case.N, "K": case.K, "dtype": dtype_name, "selected_by": why[case.key], "notes":
                f"case failed: {type(e).__name__}: {e}"[:300]
            }
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        r = rec.get(f"gb300_fused_over_{BASELINE}")
        eta = (time.time() - started) / (i + 1) * (len(todo) - i - 1) / 60
        print(
            f"  [{i + 1:>3}/{len(todo)}] {case.model[:18]:18} {case.layer[:23]:23} "
            f"{case.M}x{case.N}x{case.K:<7} {dtype_name:5} {rec.get('mode') or '-':6} "
            f"gb300 {f'{r:.3f}x' if r else '  -   '}  {time.time() - t:5.1f}s  eta {eta:5.1f}m"
            f"{'  ' + rec['notes'][:70] if rec.get('notes') else ''}", flush=True)
        torch.cuda.empty_cache()
    fh.close()

    for line in _report(chosen, opts["dtypes"]):
        print(line)
