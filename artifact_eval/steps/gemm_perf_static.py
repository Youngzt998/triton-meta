"""gemm.perf.static -- the five arms of `gemm.perf.random`, on the fixed layer shapes of the
open-weight models that are current in August 2026.

Same five arms, same operands, same five timings, same byte checks.  The ONLY difference is
where the shapes come from: `fusion_moe/models_2026.py` instead of a random draw.

    cublas               cuBLAS through `hot_cublas`.  NVIDIA's heuristic picks the kernel.
                         THE BIT REFERENCE, and the baseline every ratio is taken against.
    torch_auto           torch.compile(mode="max-autotune-no-cudagraphs") as its autotuner
                         picks, with the extern included.  Unconstrained.
    torch_triton         the same with `max_autotune_gemm_backends="TRITON"`, so it is
                         Inductor's own Triton template.  Unconstrained.
    bitequiv_autotuning  bit-equivalent to cuBLAS: the kernel AS IT STOOD BEFORE the sm_103
                         rewrites, with its configuration chosen by a search.
    gb300_accelerated    bit-equivalent to cuBLAS: the GB300-specific rewrites and the shipped
                         fitted rule.

The last two are both byte-identical to cuBLAS and differ in HOW THEY GOT THERE -- one searches
a configuration space, the other is rewrites written for this architecture plus a fitted rule.
They therefore differ in two ways at once, the kernel and the way its configuration is chosen,
and a comparison between them covers both.

WHAT IS REPORTED, AND WHAT IS NOT
---------------------------------
Every arm is reported against cuBLAS and nothing else: `torch_auto / cublas`,
`torch_triton / cublas`, `bitequiv_autotuning / cublas`, `gb300_accelerated / cublas`, each
oriented so that above 1 means faster than cuBLAS.  Ratios BETWEEN arms are deliberately not
printed.  They hide the levels -- one arm over another says nothing about whether either is any
good against the reference -- and between two arms that differ in more than one way they invite
a causal reading the measurement cannot support.  A reader with four numbers on one baseline can
form whatever comparison they want.

In particular, `gb300_accelerated` against either torch arm is NOT the price of bit-exactness.
The two are different kernels: ours has TMA descriptors, a persistent grid and warp
specialization, and Inductor's `mm_template` has none of them.  It is a product comparison --
what a user gets by switching -- and the kernel difference dominates whatever the bit constraint
costs.

THE MEASUREMENT IS IMPORTED, NOT COPIED
---------------------------------------
Every arm, every timing and every byte check is `gemm_perf_random`'s per-shape function, called
from here with a different shape list.  Nothing about the measurement is re-implemented in this
file; if the two steps ever disagreed about what an arm is, the comparison between them would be
worthless.  What this file owns is the shapes, the checkpointing and the report.

The random draw covers a space and can support a statement about that space.  These fixed shapes
cannot -- 349 shapes are not a sample of anything -- but they are the GEMMs a reader cares about.
Neither step substitutes for the other.

THE SHAPES
----------
`fusion_moe/models_2026.py` builds 461 cases over 391 distinct (M, N, K).  Every number in that
table was read out of the model's own `config.json` on Hugging Face on 2026-08-16 with the source
key recorded, and the layer-to-GEMM mapping was checked against official modeling code.  Nothing
is added here from memory.  The five pairings:

    moe_up     66 cases   the routed expert's gate/up projection, per expert, at a realistic
                          token count for that expert (even, hot and cold routing)
    moe_down   72 cases   the routed expert's down projection
    lora      168 cases   the LoRA merge, fused into lora_B, so K IS the rank: 8, 16, 32, 64
    lmhead     56 cases   lm_head at decode token counts 256, 64, 8 and 1
    attn       57 cases   o_proj and the fused qkv_proj at 16k, 4k and 256 tokens

That spread of K is the reason to read the report per pairing rather than as one mean: K is 8 on
a LoRA adapter and 8192 on an lm_head, and there is no reason the bit constraint should cost the
same on both.

This step is GEMM only.  Each case also carries the epilogue that really follows it in that
model; that is `gemm.fusion`'s question, and `epi` is ignored here.

A shape shared by several models is MEASURED ONCE.  Deduplication is on (M, N, K, dtype); the
full case list is kept, so a measurement is reported under every model and layer it belongs to,
and both counts are printed.  `cases` and `shared_with` on a row say when that happened.

DTYPE
-----
fp16 on every case.  fp8 (e4m3) additionally on the four FFN pairings and `attn`, nowhere else.
The rule is what an fp8-served open-weight checkpoint actually quantizes: DeepSeek ships native
fp8 weights for the transformer's linear layers, and the `qkv_proj` case in the model table
already carries "scale then cast to fp8" as its epilogue -- that case only exists in an fp8-served
model.  The two exclusions:

    lmhead   the same recipes keep the embedding and the output head out of fp8; the head feeds
             the sampling distribution.
    lora     adapters are served in bf16/fp16, and K is the rank, so 42 of the 168 cases have
             K = 8.  cuBLAS returns no algorithm at all for an fp8 K that is not a multiple of
             16, so those rows could not exist even if the rule allowed them.  The multiple-of-16
             check is applied to every fp8 candidate anyway and the skips are counted, so a model
             added later cannot slip past it.

That is 195 fp8 cases over 157 more distinct shapes: 506 (M, N, K, dtype) shapes measured in all,
45% more than fp16 alone.  `PERF_STATIC_DTYPES=fp16` drops the fp8 half.

ORDER, SO A TRUNCATED RUN IS STILL READABLE
-------------------------------------------
Shapes are ordered round-robin across the five pairings.  A run that stops early therefore
covers all seven roughly equally instead of finishing `moe_up` and never reaching `attn`.  Within
a pairing the order is sorted and fixed, so a prefix is a deterministic slice and NOT a random
sample of that pairing -- read a partial run as "these cases", never as "this pairing".

RUNNING IT
----------
    export PYTHONPATH=$(git rev-parse --show-toplevel)
    CUDA_VISIBLE_DEVICES=0 python -u artifact_eval/artifact.py --run gemm.perf.static --minutes 0

`--minutes 0` runs to completion; a positive value is a wall-clock cap and exits cleanly.
Everything else is an environment variable, because `artifact.py`'s CLI is shared by every step:

    PERF_STATIC_DTYPES       comma-separated, default `fp16,fp8`.
    PERF_STATIC_PAIRINGS     comma-separated subset of the five, for a validation slice.
    PERF_STATIC_ROUNDS       measurement rounds, best of.  Default 3.
    PERF_STATIC_MAX_CONFIGS  cap on bitequiv_autotuning's search space.  Default 0 = no cap.  A non-zero value
                             is recorded on every row it touched, so a bounded run cannot be
                             read as full coverage.
    PERF_STATIC_SEARCH_S     bitequiv_autotuning's per-shape search budget, in seconds.
    PERF_STATIC_REFINE       how many of bitequiv_autotuning's fastest configurations are re-timed.
                             Both default to whatever `gemm.perf.random` uses rather than to a
                             number of their own: a different search budget would make the two
                             steps' numbers incomparable.  Where the budget stopped the sweep
                             early, the row's `searched` is below its `space` and says so.
    PERF_STATIC_CUBLASLT     which cuBLAS to match.  Default 13.1.1, the same version
                             `gemm.perf.random` pins, so the two steps are comparable.
    PERF_STATIC_LIMIT        stop after this many shapes in this process.  0 = no limit.
    PERF_STATIC_LEASE_MIN    minutes before another worker may steal a stale claim.  Default 45.
    PERF_STATIC_REDO_UNBATCHED
                             1 = take back the shapes measured one call per graph, before the
                             batch existed, whose own recorded floor says a batch would change
                             them, and measure them again.  See THE LAUNCH FLOOR below.  A shape
                             the rule would leave at one call per graph is left alone: its row
                             already IS what the new code produces.  Idempotent, so the switch
                             can be left on and an interrupted run resumed with the same
                             command.
    PERF_STATIC_REDO_CONTENDED
                             1 = take back the shapes whose latest rows record `contended`, i.e.
                             another process was on this GPU while they were timed, and measure
                             them again.  Also idempotent: a shape re-taken on a quiet card
                             stops being selected.
    PERF_STATIC_REPORT_ONLY  1 = print the table from what is on disk and exit.
    PERF_STATIC_REPORT_TXT   also write the report to this path.

THE LAUNCH FLOOR
----------------
These are small GEMMs.  A CUDA-graph replay costs about 6.9 us on this box before any work of
ours runs, and an expert down-projection at 16 tokens is 2 us of work, so measured one call at a
time it read 8.7 us -- 77% of the number was the replay, both arms paid the same 6.9, and a true
2x came out as 1.22x.  That flattens the answer toward "the arms are about the same" on exactly
the layers this step exists to report: 74 of 74 `moe_down` cases, 68 of 70 `moe_up` and 49 of 49
`lora` were flagged `near_floor`, against 0 of 49 `lmhead`.

`gemm.perf.random`'s measurement now puts `batch_n` calls in one graph, each on its own operand
copy so no call warms the next one's L2, and divides.  A kernel far above the floor still gets
`batch_n = 1` and its rows did not move.  `bat` on the per-case table and `calls per graph` under
each pairing say which rows were taken which way; a row with no `bat` predates the batch and is
not comparable to a batched row on a small shape.

CHECKPOINTING
-------------
One JSONL record per (shape, arm), fanned out to one per (case, arm) and flushed as it is
computed, so an interruption loses at most one arm.  Restart skips completed work keyed on
(M, N, K, dtype, arm); resuming is re-running the same command.
`cache/gemm.perf.static.PAUSE` is checked at the top of each shape and exits cleanly.  Shapes are
claimed one at a time through an O_EXCL lock with a lease, so a second GPU can join a run in
progress without losing or duplicating a shape.
"""
from __future__ import annotations

import errno
import json
import math
import os
import statistics
import sys
import time

from ._common import CACHE, GRAPH_FLOOR_SHARE, HERE, writer
from .gemm_perf_random import ARM_COLS, ARM_LABEL, ARM_NAME, ARMS, make_flush_buffer

NAME = "gemm.perf.static"
ORDER = 30
DESCRIPTION = "price of the bit constraint on the fixed shapes of real 2026 open-weight models"
IMPLEMENTED = True

PAIRINGS = ("moe_up", "moe_down", "mlp_up", "mlp_down", "lora", "lmhead", "attn")

# The dtype rule, in one place.  See DTYPE in the module docstring for why these three and not
# the other two.
FP8_PAIRINGS = ("moe_up", "moe_down", "mlp_up", "mlp_down", "attn")

TABLES = {
    "gemm.perf.static": {
        "doc":
        "One row per (model layer, dtype, arm). The same four arms as `gemm.perf.random` on "
        "the layer shapes of the open-weight models current in August 2026, taken from "
        "`fusion_moe/models_2026.py`, so every row traces to a weight that exists. A shape "
        "shared by several models was measured once and appears under each of them; `cases` "
        "and `shared_with` say when. All five timings are kept raw and the ratios are "
        "computed from them, so which timing is the headline can change without re-running.",
        "cols": [
            ("model", "str", "model the shape comes from"),
            ("layer", "str", "layer within that model, e.g. expert.up_proj T16k/even, "
             "q_proj.lora_B r=16, lm_head"),
            ("pairing", "str", "which of the seven layer kinds this case belongs to: moe_up, "
             "moe_down, lora, lmhead or attn. The report is aggregated on this"),
            ("note", "str", "what the layer is, carried over from the model table"),
            ("M", "int", "rows of A; the token count this case is measured at"),
            ("N", "int", "columns of B; out_features"),
            ("K", "int", "contraction length; in_features, and the LoRA rank on a lora row"),
            ("shape_id", "int", "index into this step's (M, N, K, dtype) shape list, so a row "
             "can be traced back to the one measurement it came from"),
            ("cases", "int", "how many (model, layer) cases share this shape. The measurement "
             "was made ONCE and copied to each of them, so `cases` rows carry equal timings"),
            ("shared_with", "str", "the other cases sharing this measurement; empty when the "
             "shape belongs to this case alone"),
        ] + ARM_COLS,
    },
}

# --------------------------------------------------------------------------------------------
# The shapes
# --------------------------------------------------------------------------------------------


def _case_list(pairings, dtypes):
    """Every (case, dtype) this step measures, plus the fp8 skips, as (cases, skipped).

    `build` is the model table's own entry point, so the case list here is the same object
    `gemm.fusion` measures and the two steps cannot drift onto different shapes.
    """
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from fusion_moe.models_2026 import GROUPS as MODEL_GROUPS
    from fusion_moe.models_2026 import build
    out, skipped = [], []
    for c in build(list(MODEL_GROUPS)):
        if pairings and c.pairing not in pairings:
            continue
        for kind in dtypes:
            if kind == "fp8":
                if c.pairing not in FP8_PAIRINGS:
                    continue
                if c.K % 16:  # cuBLAS returns no algorithm at all; not a judgement call
                    skipped.append(c)
                    continue
            out.append((c, kind))
    return out, skipped


def _shape_list(cases):
    """Deduplicate to (M, N, K, dtype), keeping every case that shares a shape.

    Ordered round-robin across the pairings so a run that stops early is spread over all five.
    A shape belonging to cases in two pairings is ordered under the first one, and still reports
    under both.
    """
    by = {}
    for c, kind in cases:
        by.setdefault((c.M, c.N, c.K, kind), []).append(c)
    lanes = {}
    for key, cs in sorted(by.items()):
        lanes.setdefault(cs[0].pairing, []).append(key)
    # Iterate the lanes that exist, not the five named ones: a pairing added to the model table
    # later must join the rotation rather than spin this loop forever.
    order = sorted(lanes, key=lambda p: (PAIRINGS.index(p) if p in PAIRINGS else len(PAIRINGS), p))
    shapes, i = [], 0
    while any(lanes.values()):
        for p in order:
            if lanes.get(p):
                M, N, K, kind = lanes[p].pop(0)
                shapes.append(
                    {"shape_id": i, "M": M, "N": N, "K": K, "dtype": kind, "pairing": p, "cases": by[(M, N, K, kind)]})
                i += 1
    return shapes


# --------------------------------------------------------------------------------------------
# Calling the random step's measurement
# --------------------------------------------------------------------------------------------

# The name the per-shape measurement goes by, newest first. It was `_one_shape` before it was
# given a public name; keeping the older names means a rename over there does not silently take
# this step with it, and a rename to something not in this list fails at startup with a message
# that says what to do rather than an AttributeError in the middle of a twenty-hour run.
_MEASURE_NAMES = ("measure_shape", "one_shape", "_one_shape")


def _measure_fn():
    """`gemm.perf.random`'s per-shape measurement: all four arms, five timings, byte checks.

    Signature `(torch, M, N, K, kind, tags, out, flush, opts)`. `tags` is copied onto every
    record it writes, and `out` only has to look like a file, which is what `_PerCase` exploits.
    """
    from . import gemm_perf_random as R
    for name in _MEASURE_NAMES:
        fn = getattr(R, name, None)
        if fn is not None:
            return fn
    raise RuntimeError("steps/gemm_perf_random.py no longer exposes its per-shape measurement under any of "
                       f"{_MEASURE_NAMES}. gemm.perf.static must call it, not copy it -- add the new name here.")


def _measure_opts(args, cfgenv, worker):
    """The option dict the measurement reads.

    Built ON TOP of the random step's own `measurement_options` rather than beside it, so a knob
    added over there arrives here already carrying its default instead of becoming a KeyError in
    the middle of a run. Only the knobs this step names itself are overridden. `search_s` and
    `refine` -- bitequiv_autotuning's per-shape search budget and how many of its fastest configurations are
    re-timed -- are deliberately left at the random step's value unless PERF_STATIC_* sets them,
    because a different search budget would make the two steps' numbers incomparable.
    """
    from . import gemm_perf_random as R
    opts = R.measurement_options(args)
    opts.update({
        "rounds": cfgenv["rounds"], "max_configs": cfgenv["max_configs"], "draws": max(2, args.reps), "worker": worker
    })
    if os.environ.get("PERF_STATIC_SEARCH_S"):
        opts["search_s"] = float(os.environ["PERF_STATIC_SEARCH_S"])
    if os.environ.get("PERF_STATIC_REFINE"):
        opts["refine"] = int(os.environ["PERF_STATIC_REFINE"])
    return opts


class _PerCase:
    """Stands in for the output file and turns each (shape, arm) record into one per case.

    The measurement identifies a row with the `tags` its caller passes, which is one dict and so
    one case. A shape here can belong to several (model, layer) cases and is measured once for
    all of them, so the fan-out happens on the way to disk. `cases` on the row says how many rows
    carry that one measurement, and `shared_with` names the others.
    """

    def __init__(self, fh, shape):
        self.fh, self.shape, self.n = fh, shape, 0

    def write(self, line):
        rec = json.loads(line)
        cs = self.shape["cases"]
        for c in cs:
            r = dict(rec)
            r.update({
                "model": c.model, "layer": c.layer, "pairing": c.pairing, "note": c.note, "cases": len(cs),
                "shared_with": "; ".join(f"{o.model} {o.layer}" for o in cs if o is not c)
            })
            self.fh.write(json.dumps(r) + "\n")
        self.n += 1

    def flush(self):
        self.fh.flush()


# --------------------------------------------------------------------------------------------
# Checkpointing: done keys, the pause sentinel, and one claim per shape
# --------------------------------------------------------------------------------------------


def _jsonl_path():
    return os.path.join(CACHE, "gemm.perf.static.jsonl")


def _pause_path():
    return os.path.join(CACHE, "gemm.perf.static.PAUSE")


def _would_batch(arms):
    """Would the batch rule put more than one call in a graph for this shape?

    The same arithmetic as `pick_batch`, run on the floor and the fastest arm the shape ALREADY
    RECORDED instead of on a fresh probe. It is only used to decide whether a row taken before
    the batch existed is worth taking again, so an estimate is enough; the re-measurement probes
    properly. A `near_floor` row is a subset of what this selects -- near the floor means the
    floor is over a third of the number, and the rule batches from about a twentieth.
    """
    fl = next((r.get("floor_ms") for r in arms.values() if r.get("floor_ms")), None)
    ts = [r.get("device_flush_ms") for r in arms.values() if r.get("device_flush_ms")]
    if not fl or not ts:
        return False
    return math.ceil(fl / (GRAPH_FLOOR_SHARE * max(min(ts) - fl, 1e-4))) > 1


def _done_keys(redo_unbatched=False, redo_contended=False):
    """(M, N, K, dtype, arm) already on disk. A shape whose every arm is here is skipped.

    Two switches take work back. Both are IDEMPOTENT -- what they select stops being selected
    once the shape has been re-measured -- so either can be left on and an interrupted run
    resumed with the same command. Both KEEP the old rows: the report reads the LAST record for
    an arm, so a re-measurement wins and the row it replaced stays on disk to compare against.

    `redo_unbatched` takes back the shapes that were measured one call per graph, before the
    batch existed, AND whose own recorded floor says the batch would now change them. A shape
    the rule would leave at one call per graph is left alone: its existing row IS what the new
    code would produce, so re-measuring it would only add noise. `_would_batch` is the test.

    `redo_contended` takes back the shapes whose latest rows were measured while another process
    was on this GPU. Those are not measurements -- shared SMs, L2 and bandwidth make the kernel
    genuinely slower and no per-process counter can subtract that back out -- so the only honest
    move is to take them again on a quiet card.
    """
    last = {}
    if not os.path.exists(_jsonl_path()):
        return set()
    with open(_jsonl_path(), errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue  # a torn last line from a kill; that shape is simply redone
            last[(r.get("M"), r.get("N"), r.get("K"), r.get("dtype"), r.get("arm"))] = r
    by_shape = {}
    for key, r in last.items():
        by_shape.setdefault(key[:4], {})[key[4]] = r
    drop = set()
    for shape, arms in by_shape.items():
        rs = arms.values()
        if redo_unbatched and not any(r.get("batch_n") for r in rs) and _would_batch(arms):
            drop.add(shape)
        if redo_contended and any(r.get("contended") for r in rs):
            drop.add(shape)
    return {k for k in last if k[:4] not in drop}


def _claim(idx, me, lease_s):
    """Take shape `idx`, or report that someone else holds it.

    Same lock-file-with-a-lease as the random step, on this step's own directory. Work is
    claimed one shape at a time rather than sliced up front, so a second GPU can join a run
    already in progress -- which is what happens here, because the GPUs free up one at a time.
    """
    d = os.path.join(CACHE, "gemm.perf.static.claims")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, str(idx))
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        try:
            if time.time() - os.path.getmtime(path) < lease_s:
                return False
            os.utime(path, None)  # stale: steal it, and take the lease with it
            with open(path, "w") as f:
                f.write(f"{me} stolen {time.strftime('%H:%M:%S')}\n")
            return True
        except OSError:
            return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{me} {time.strftime('%H:%M:%S')}\n")
    return True


def _touch(idx):
    try:
        os.utime(os.path.join(CACHE, "gemm.perf.static.claims", str(idx)), None)
    except OSError:
        pass


# --------------------------------------------------------------------------------------------
# The report. Two levels: every case, then the five pairings.
# --------------------------------------------------------------------------------------------

# Deliberately local rather than imported from the random step: these three decide how this
# step's own report prints, and a rename in the other file must not be able to break the report
# of a run that has already been paid for.


def _geo(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def _dist(vals, top=4):
    c = {}
    for v in vals:
        c[v or "-"] = c.get(v or "-", 0) + 1
    return " ".join(f"{k}:{n}" for k, n in sorted(c.items(), key=lambda kv: -kv[1])[:top])


# Every arm against ONE baseline, cuBLAS, as cuBLAS time over that arm's time, so above 1 means
# faster than cuBLAS. Ratios between two arms are deliberately not reported: see the module
# docstring. `BASELINE` is not in the list because its own ratio is 1 by construction.
BASELINE = "cublas"
VS_BASELINE = ("torch_auto", "torch_triton", "pre_search", "ours")


def _rows():
    if not os.path.exists(_jsonl_path()):
        return []
    rows = []
    with open(_jsonl_path(), errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _by_case(rows):
    """(model, layer, pairing, M, N, K, dtype) -> {arm: record}, latest record per arm wins."""
    out = {}
    for r in rows:
        key = (r.get("model"), r.get("layer"), r.get("pairing"), r.get("M"), r.get("N"), r.get("K"), r.get("dtype"))
        out.setdefault(key, {})[r.get("arm")] = r
    return out


def _ratios(arms, timing):
    """{arm: cuBLAS time / that arm's time}, so above 1 means faster than cuBLAS."""
    t = {k: (arms.get(k) or {}).get(timing) for k in ARMS}
    if not t.get(BASELINE):
        return {}
    return {k: t[BASELINE] / t[k] for k in VS_BASELINE if t.get(k)}


# One letter per arm whose byte check came up short, for the flags column.
_BIT_FLAG = {"ours": "G", "pre_search": "B", "torch_auto": "T"}


def _flags(arms):
    """One character per thing that makes a row's numbers weaker than they look."""
    f = ""
    if any((arms.get(k) or {}).get("near_floor") for k in ARMS):
        f += "F"
    if any((arms.get(k) or {}).get("captured") == 0 for k in ARMS):
        f += "!"
    for k, letter in _BIT_FLAG.items():
        r = arms.get(k) or {}
        if r.get("bit_total") and int(r.get("bit_ok") or 0) < int(r["bit_total"]):
            f += letter
    if any((arms.get(k) or {}).get("declined") for k in ARMS):
        f += "d"
    if any((arms.get(k) or {}).get("error") for k in ARMS):
        f += "e"
    return f


def _bits(arms, key):
    r = arms.get(key) or {}
    return f"{int(r.get('bit_ok') or 0)}/{int(r['bit_total'])}" if r.get("bit_total") else "-"


def _report(lines, meta, timing="device_flush_ms"):
    """Per case, then per pairing. The per-pairing view is the one to read.

    A single mean over all of it would average an lm_head (K = 8192, N = 200k) with a LoRA
    adapter (K = 8) and say nothing about either.
    """
    P = lines.append
    rows = _rows()
    cases = _by_case(rows)
    P("")
    P(f"gemm.perf.static -- {len(cases)} (model, layer, dtype) cases over "
      f"{len({(k[3], k[4], k[5], k[6]) for k in cases})} distinct (M, N, K, dtype) shapes, "
      f"{len(rows)} arm records")
    if meta:
        P(f"  planned: {meta.get('cases')} cases over {meta.get('shapes')} shapes "
          f"({meta.get('fp16_shapes')} fp16, {meta.get('fp8_shapes')} fp8); "
          f"{meta.get('fp8_skipped')} fp8 candidates dropped for K % 16")
        P(f"  dtype rule: fp16 on every case, fp8 additionally on {', '.join(FP8_PAIRINGS)}")
        if meta.get("max_configs"):
            P(f"  WARNING: bitequiv_autotuning's space was BOUNDED at {meta['max_configs']} configurations "
              f"in this run; `space` and `searched` on each row say by how much")
    P(f"  timing column: {timing}")
    P("")
    P("  The arms, and what a ratio here is")
    for k in ARMS:
        P(f"    {ARM_NAME[k]:22} {ARM_LABEL[k]}")
    P("    Every arm is reported against cuBLAS and against nothing else, as cuBLAS time over")
    P("    that arm's time, SO ABOVE 1 MEANS FASTER THAN CUBLAS. Ratios between two arms are not")
    P("    printed: they hide the levels, and where two arms differ in more than one way they")
    P("    invite a causal reading this measurement cannot support. Four numbers on one baseline")
    P("    let a reader form whatever comparison they want.")
    P("    bitequiv_autotuning and gb300_accelerated are BOTH byte-identical to cuBLAS and differ")
    P("    in how they got there: a search over a configuration space, against rewrites written")
    P("    for this architecture plus a fitted rule. They differ in two ways at once -- the")
    P("    kernel and the way its configuration is chosen -- so a comparison covers both.")
    P("    gb300_accelerated against a torch arm is NOT the price of bit-exactness. They are")
    P("    different kernels: ours has TMA descriptors, a persistent grid and warp specialization")
    P("    and Inductor's mm_template has none of them. It is a product comparison.")

    # ---- per case ---------------------------------------------------------------------------
    P("")
    P("  per case. Times are microseconds per call; `x cuBLAS` is cuBLAS over that arm, above 1")
    P("  faster than cuBLAS. No G, B or T in flags means every draw of that arm was byte-identical.")
    P(f"    {'model':22} {'layer':30} {'pairing':8} {'dt':4} {'M':>6} {'N':>7} {'K':>6} {'mode':13} "
      f"| microseconds per call {'cublas':>11} {'torch_auto':>10} {'torch_triton':>12} "
      f"{'bitequiv_autotuning':>19} {'gb300_accelerated':>17} "
      f"| x cuBLAS {'torch_auto':>10} {'torch_triton':>12} {'bitequiv_autotuning':>19} "
      f"{'gb300_accelerated':>17} | {'batch':>5} {'floor':>7} flags")
    for key in sorted(cases, key=lambda k: (PAIRINGS.index(k[2]) if k[2] in PAIRINGS else 9, k[6], k[0], k[1])):
        model, layer, pairing, M, N, K, dt = key
        arms = cases[key]
        rat = _ratios(arms, timing)
        t = {k: (arms.get(k) or {}).get(timing) for k in ARMS}
        any_row = next(iter(arms.values()), {})

        def us(x, w):
            return f"{x*1e3:{w}.3f}" if x else f"{'-':>{w}}"

        def rr(k, w):
            return f"{rat[k]:{w}.3f}" if k in rat else f"{'-':>{w}}"

        bat = next((a.get("batch_n") for a in arms.values() if a.get("batch_n")), None)
        flr = next((a.get("floor_ms") for a in arms.values() if a.get("floor_ms")), None)
        P(f"    {str(model)[:22]:22} {str(layer)[:30]:30} {str(pairing):8} {str(dt):4} {M:6d} {N:7d} {K:6d} "
          f"{str(any_row.get('mode') or '-')[:13]:13} "
          f"| {'':21} {us(t.get('cublas'), 11)} {us(t.get('torch_auto'), 10)} {us(t.get('torch_triton'), 12)} "
          f"{us(t.get('pre_search'), 19)} {us(t.get('ours'), 17)} "
          f"| {'':8} {rr('torch_auto', 10)} {rr('torch_triton', 12)} {rr('pre_search', 19)} "
          f"{rr('ours', 17)} | {(str(bat) if bat else '-'):>5} "
          f"{(f'{flr*1e3:7.3f}' if flr else '      -')} {_flags(arms)}")

    # ---- per pairing ------------------------------------------------------------------------
    def block(title, groups):
        """One row per (group, arm), every arm on the cuBLAS baseline.

        One row per group with an arm in every column was the older shape of this table and it
        made the levels hard to see; a row per arm puts the four numbers under each other.
        """
        P("")
        P(f"  {title}")
        P(f"    {'group':16} {'arm':20} {'cases':>5} {'shapes':>6} {'x cuBLAS geo':>12} {'[p25':>6} "
          f"{'med':>6} {'p75]':>6} {'worst':>6} {'best':>6} {'us/call geo':>11} {'byte-identical':>16} "
          f"{'near floor':>10} {'uncaptured':>10}")
        for gname in sorted(groups, key=lambda g: (PAIRINGS.index(g.split("/")[0])
                                                   if g.split("/")[0] in PAIRINGS else 9, g)):
            keys = groups[gname]
            rat = {k: [] for k in VS_BASELINE}
            tms = {k: [] for k in ARMS}
            bits = {k: [0, 0] for k in _BIT_FLAG}
            modes, picks, bats = [], [], []
            near = {k: 0 for k in ARMS}
            unc = {k: 0 for k in ARMS}
            for key in keys:
                arms = cases[key]
                for k, v in _ratios(arms, timing).items():
                    rat[k].append((v, key))
                for k in ARMS:
                    r = arms.get(k) or {}
                    if r.get(timing):
                        tms[k].append(r[timing] * 1e3)
                    near[k] += 1 if r.get("near_floor") else 0
                    unc[k] += 1 if r.get("captured") == 0 else 0
                    if k in bits and r.get("bit_total"):
                        bits[k][0] += int(r.get("bit_ok") or 0)
                        bits[k][1] += int(r["bit_total"])
                modes.append(next((a.get("mode") for a in arms.values() if a.get("mode")), None))
                picks.append((arms.get("torch_auto") or {}).get("pick"))
                bats.append(next((a.get("batch_n") for a in arms.values() if a.get("batch_n")), None))
            nsh = len({(k[3], k[4], k[5], k[6]) for k in keys})

            def num(x, w=6):
                return f"{x:{w}.3f}" if x is not None else f"{'-':>{w}}"

            for k in ARMS:
                v = [x for x, _ in rat.get(k, [])]
                o, tot = bits[k] if k in bits else (0, 0)
                P(f"    {gname:16} {ARM_NAME[k]:20} {len(keys):5d} {nsh:6d} "
                  f"{num(_geo(v), 12)} {num(_pct(v, .25))} {num(_pct(v, .5))} {num(_pct(v, .75))} "
                  f"{num(min(v) if v else None)} {num(max(v) if v else None)} "
                  f"{num(_geo(tms[k]), 11)} {(f'{o}/{tot}' if tot else '-'):>16} "
                  f"{near[k]:10d} {unc[k]:10d}")
                if rat.get(k):
                    lo, hi = min(rat[k]), max(rat[k])
                    P(f"    {'':16} {'':20} worst {lo[0]:6.3f} {lo[1][0]} {lo[1][1]} {lo[1][6]} "
                      f"{lo[1][3]}x{lo[1][4]}x{lo[1][5]}   best {hi[0]:6.3f} {hi[1][0]} {hi[1][1]} "
                      f"{hi[1][6]} {hi[1][3]}x{hi[1][4]}x{hi[1][5]}")
            P(f"    {'':16} plan modes {_dist(modes)} | torch picks {_dist(picks)}")
            P(f"    {'':16} calls per graph {_dist(bats, top=6)}")

    grp = {}
    for key in cases:
        grp.setdefault(key[2] or "-", []).append(key)
    block("per pairing", grp)
    grp = {}
    for key in cases:
        grp.setdefault(f"{key[2] or '-'}/{key[6]}", []).append(key)
    block("per pairing and dtype", grp)

    # ---- what did not measure -----------------------------------------------------------------
    bad = {}
    for r in rows:
        for k in ("declined", "error"):
            if r.get(k):
                bad.setdefault(f"{r.get('arm')} {k}: {str(r[k])[:70]}", 0)
                bad[f"{r.get('arm')} {k}: {str(r[k])[:70]}"] += 1
    if bad:
        P("")
        P("  arms that measured nothing (rows kept, never dropped: a filtered set is a biased set)")
        for k, n in sorted(bad.items(), key=lambda kv: -kv[1])[:25]:
            P(f"    {n:5d}  {k}")
    P("")
    P("  Check `byte-identical` equals the total first: where an arm was not byte-identical to")
    P("  cuBLAS its time is not a measurement of anything. It counts input draws, ten per case.")
    P("  flags  F near the launch floor (under 3x floor, every ratio compressed toward 1)")
    P("         ! an arm the CUDA graph would not capture   d declined   e error")
    P("         G gb300_accelerated, B bitequiv_autotuning, T torch_auto was not byte-identical")
    P("           to cuBLAS on every draw")
    P("")
    P("  batch is `batch_n`: how many calls went into one CUDA graph, each on its own operand")
    P("  copy, with the replay divided by that. A one-call replay costs about 6.9 us on this box")
    P("  before any work of ours runs, which is 2% of an lm_head and 77% of a 2 us expert GEMM,")
    P("  and every arm pays the same 6.9, so a true 2x on a small shape reads as 1.22x. The batch")
    P("  divides that fixed cost; a kernel already far above the floor gets batch 1, unchanged.")
    P("  `-` is a row taken before the batch existed and it is NOT comparable to a batched row on")
    P("  a small shape. floor is floor_ms at that row's own batch, so F means the same thing at")
    P("  every batch. It does not fall to zero: graph nodes run one after another and an empty")
    P("  kernel still takes about 1 us, which a real caller pays too. A GEMM the size of a LoRA")
    P("  merge stays flagged after batching, and it should -- it is not distinguishable from a")
    P("  launch.")
    return lines


# --------------------------------------------------------------------------------------------


def run(args, env):
    import torch
    from bitequiv.cublas_match import cublaslt_version, set_cublaslt

    cfgenv = {
        "rounds": int(os.environ.get("PERF_STATIC_ROUNDS", 3)),
        "max_configs": int(os.environ.get("PERF_STATIC_MAX_CONFIGS", 0)),
    }
    lease_s = float(os.environ.get("PERF_STATIC_LEASE_MIN", 45)) * 60
    limit = int(os.environ.get("PERF_STATIC_LIMIT", 0))
    dtypes = tuple(x for x in os.environ.get("PERF_STATIC_DTYPES", "fp16,fp8").split(",") if x)
    pairings = tuple(x for x in os.environ.get("PERF_STATIC_PAIRINGS", "").split(",") if x)
    txt = os.environ.get("PERF_STATIC_REPORT_TXT", "")

    cases, skipped = _case_list(pairings, dtypes)
    shapes = _shape_list(cases)
    meta = {
        "cases": len(cases), "shapes": len(shapes), "fp8_skipped": len(skipped), "fp16_shapes":
        sum(1 for s in shapes if s["dtype"] == "fp16"), "fp8_shapes": sum(1 for s in shapes if s["dtype"] == "fp8"),
        "max_configs": cfgenv["max_configs"]
    }

    if os.environ.get("PERF_STATIC_REPORT_ONLY"):
        lines = _report([], meta)
        print("\n".join(lines))
        if txt:
            open(txt, "w").write("\n".join(lines) + "\n")
        return

    set_cublaslt(os.environ.get("PERF_STATIC_CUBLASLT", "13.1.1") or None)
    ltver = ".".join(map(str, cublaslt_version()))
    worker = f"{os.uname().nodename}:{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}:{os.getpid()}"
    measure = _measure_fn()
    opts = _measure_opts(args, cfgenv, worker)

    print(f"\n[gemm.perf.static] {meta['cases']} cases over {meta['shapes']} shapes "
          f"({meta['fp16_shapes']} fp16, {meta['fp8_shapes']} fp8), matching cuBLASLt {ltver}, worker {worker}")
    print("  shapes from fusion_moe/models_2026.py, deduplicated on (M, N, K, dtype)")
    print(f"  fp8 on {', '.join(FP8_PAIRINGS)} only; {len(skipped)} fp8 candidates dropped for K % 16")
    print(f"  records     {_jsonl_path()}")
    print(f"  pause with  touch {_pause_path()}")
    print(f"  bitequiv_autotuning: space cap {opts['max_configs'] or 'none'}, {opts['search_s']}s budget, "
          f"top {opts['refine']} re-timed; {opts['rounds']} rounds, {opts['draws']} byte-check draws")
    if cfgenv["max_configs"]:
        print("  WARNING: bitequiv_autotuning's space is BOUNDED in this run; every row it touched records both "
              "`space` and `searched`.")

    done = _done_keys(redo_unbatched=bool(os.environ.get("PERF_STATIC_REDO_UNBATCHED")),
                      redo_contended=bool(os.environ.get("PERF_STATIC_REDO_CONTENDED")))
    fh = writer("gemm.perf.static")
    flush = make_flush_buffer(torch)
    deadline = time.time() + args.minutes * 60 if args.minutes and args.minutes > 0 else None
    n, spent, todo = 0, [], 0
    for shape in shapes:
        key = (shape["M"], shape["N"], shape["K"], shape["dtype"])
        if not all((*key, arm) in done for arm in ARMS):
            todo += 1
    print(f"  {todo} of {len(shapes)} shapes still to do\n", flush=True)

    for shape in shapes:
        if os.path.exists(_pause_path()):
            print("  PAUSE sentinel present; stopping cleanly. Remove it and re-run to resume.")
            break
        if deadline and time.time() > deadline:
            print("  --minutes budget reached; stopping cleanly.")
            break
        if limit and n >= limit:
            print("  PERF_STATIC_LIMIT reached; stopping cleanly.")
            break
        key = (shape["M"], shape["N"], shape["K"], shape["dtype"])
        if all((*key, arm) in done for arm in ARMS):
            continue
        if not _claim(shape["shape_id"], worker, lease_s):
            continue
        sink = _PerCase(fh, shape)
        tags = {"shape_id": shape["shape_id"], "cublaslt": ltver}
        t0 = time.time()
        _touch(shape["shape_id"])
        phases = None
        try:
            phases = measure(torch, shape["M"], shape["N"], shape["K"], shape["dtype"], tags, sink, flush, opts)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # `measure_shape` does not raise, so this is the measurement itself breaking. The
            # shape still gets its five rows: a shape that vanishes from the table is a shape
            # nobody goes back to look at.
            for arm in ARMS:
                sink.write(
                    json.dumps({
                        **tags, "M": shape["M"], "N": shape["N"], "K": shape["K"], "dtype": shape["dtype"], "arm": arm,
                        "worker": worker, "error": f"{type(e).__name__}: {e}"[:300]
                    }))
            sink.flush()
        n += 1
        spent.append(time.time() - t0)
        eta = statistics.median(spent) * (todo - n) / 3600.0
        # Where the shape went, when the measurement reports it: a long shape should be readable
        # as "the search took it" or "torch.compile took it" without re-running anything.
        where = " ".join(f"{k}={v:.0f}s" for k, v in sorted((phases or {}).items()) if v >= 1.0)
        print(
            f"  [{n}/{todo}] {shape['pairing']:9} {shape['dtype']:4} "
            f"{shape['M']}x{shape['N']}x{shape['K']} {len(shape['cases'])} case(s) "
            f"{spent[-1]:6.1f}s   eta {eta:5.1f}h   {where}", flush=True)
        _touch(shape["shape_id"])
    fh.close()
    lines = _report([], meta)
    print("\n".join(lines))
    if txt:
        open(txt, "w").write("\n".join(lines) + "\n")
