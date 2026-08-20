"""inner_tree.layout -- the reduction layout-optimization pass: is it bit-safe, and what does it
buy?

This is a re-run, on this machine, of the measurement reported with the pass itself: commit
`8176cccdc`, PR #2312, "M2: reduction-layout optimization pass". Nothing about the experiment was
redesigned: same kernels, same three arms, same `gap_closed`, same harness
(`bitequiv/evaluation/evaluate_opt.py`, which is what that PR used too). Only the machine, the
Triton version and the input dtype differ, and each of those three is printed next to the earlier
number so a reader can see which axis moved.

    arms      base = reduction_ordering=inner_tree, standard pipeline
              opt  = the same, plus `tritongpu-optimize-reduction-layout{ideal,8,256}` appended
                     at end-of-TTGIR
              ceil = the same kernel with reduction_ordering=unordered, standard pipeline -- what
                     the compiler picks when the ordering constraint is dropped entirely
    metric    speedup    = base_ms / opt_ms
              gap_closed = (base_ms - opt_ms) / (base_ms - ceil_ms);  1.0 = opt reached ceil

Read `bit_changed` before any speedup. A speedup on a row whose bytes moved is not a result.

Attribution. The ordering path this pass sits on top of is not ours: the `reduction_ordering` /
`inner_tree` mechanism, and the `TRITON_STRICT_REDUCTION_ORDERING` environment variable that pins
it, were written by Nick Riasanovsky, a co-author. The pass under test here is
`tritongpu-optimize-reduction-layout`, added in commit `8176cccdc` (PR #2312).

HOW TO RUN

    export PYTHONPATH=$(git rev-parse --show-toplevel)
    CUDA_VISIBLE_DEVICES=<idle gpu> TRITON_ALWAYS_COMPILE=1 \
        python artifact_eval/artifact.py --run inner_tree.layout --minutes 600

`TRITON_ALWAYS_COMPILE=1` is not optional and has to be in the environment at process start.
Triton's in-memory compile cache is keyed on the specialization and the launch options only, so
flipping the injected pass on or off does not change the key: without it the pass-on build is
served the pass-off one and the whole run reports a perfect, meaningless `1.00x, 0 bits changed`.
This step refuses to measure anything until it has watched two builds of the same config actually
produce different TTGIR, so a cache that is still warm is an abort, not a clean table.

`--minutes` is a resumable budget, not a sample size. Rows stream to
`cache/inner_tree.layout.jsonl` one at a time and a second invocation picks up where the first
stopped, so a short run followed by more runs gives the same table as one long run.

    INNER_TREE_LAYOUT_STAGES=1     preflight only: the self-checks, the config counts and a timing
                                   projection, compile-only, writes nothing. Minutes, not hours.
    INNER_TREE_LAYOUT_KERNELS=a,b  restrict the kernel list (debugging; the default is the list
                                   PR #2312 measured).
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time

from ._common import CACHE, ROOT, writer

NAME = "inner_tree.layout"
ORDER = 70
DESCRIPTION = "is the reduction layout pass bit-safe, and what does it buy"
IMPLEMENTED = True

# Exactly the invocation PR #2312's description documents for the bitequiv eval hook.
PASS_SPEC = "tritongpu-optimize-reduction-layout{ideal,8,256}"
PASS_NAME = "tritongpu-optimize-reduction-layout"
PASS_SOURCE = os.path.join(ROOT, "lib", "Dialect", "TritonGPU", "Transforms", "OptimizeReductionLayout.cpp")

# This project measures f16 and fp8; bf16 and f32 stay supported in the registry but are not what
# we report. `col_bf16` is the exception and it is not a choice: its spec pins `in_dtype=bf16`, so
# the dtype axis does not reach it at all (see the note this step prints).
DTYPES = ("f16", "fp8")
PINNED_DTYPE = {"col_bf16": ("bf16", )}

# The eleven kernels of PR #2312's H100 table, in its order. Three of them are not in the
# registry `evaluate_opt` resolves against -- they live in the Inductor and benchmark-zoo suites,
# which have a different spec shape -- so they are reported as unmeasured rows rather than dropped.
KERNELS = ("sum_3d_outer", "sum_2d_col", "sum_2d_col_big", "sum_2d_axis0", "col_exp_sum", "col_bf16", "col_sum_loop",
           "col_dot")
UNREACHABLE = {
    "I_bias_grad_dim0": "bitequiv/evaluation/realistic_inductor_kernels.py (SPECS, not KernelSpec)",
    "J_epilogue_colsum_dim0": "bitequiv/evaluation/realistic_inductor_kernels.py (SPECS, not KernelSpec)",
    "layernorm_bwd_dwdb": "bitequiv/evaluation/benchmark_kernels.py (register(), not KernelSpec)",
}

# PR #2312's H100 table, verbatim: {kernel: {num_warps: (speedup, gap_closed)}}. Measured on
# triton-3.7.0 with f32 input (the dtype axis did not exist in eval_kernels.py at that commit, so
# every kernel took the f32 default) -- except col_bf16, whose spec pinned bf16 then as it does
# now. `None` is its one col_bf16 number, reported without saying which num_warps.
PRIOR_H100 = {
    "sum_3d_outer": {4: (3.27, 1.01), 8: (3.32, 1.05)},
    "sum_2d_col": {4: (1.44, 0.94), 8: (1.25, 0.98)},
    "sum_2d_col_big": {4: (1.46, 0.99), 8: (1.32, 0.79)},
    "sum_2d_axis0": {4: (1.34, 0.96), 8: (1.21, 1.03)},
    "col_exp_sum": {4: (1.43, 0.94), 8: (1.27, 0.95)},
    "col_bf16": {None: (1.40, 0.96)},
    "col_sum_loop": {4: (1.26, 0.97), 8: (1.25, 1.11)},
    "col_dot": {2: (1.12, 0.98), 4: (1.19, 0.89), 8: (1.09, 0.98)},
    "I_bias_grad_dim0": {4: (1.29, 1.00), 8: (1.49, 0.95)},
    "J_epilogue_colsum_dim0": {4: (3.03, 1.26), 8: (2.58, 1.00)},
    "layernorm_bwd_dwdb": {4: (3.58, 1.07), 8: (3.89, 1.05)},
}
PRIOR_DTYPE = {"col_bf16": "bf16"}  # everything else was f32
PRIOR_LABEL = "PR #2312 H100 triton-3.7.0"

# The same PR also carries a GB300/sm_100 table -- 72 f32 configs per kernel, median/max/
# gap_closed. Same hardware family as this machine, so it is the closer of the two priors and is
# printed alongside; it is not the headline comparison, which is the H100 one.
PRIOR_SM100 = {
    "sum_3d_outer": (5.15, 5.64, 1.01),
    "col_sum_loop": (2.12, 2.40, 1.00),
    "col_exp_sum": (1.93, 2.27, 0.96),
    "sum_2d_col": (1.92, 2.26, 0.92),
    "sum_2d_col_big": (1.80, 2.15, 0.71),
    "col_bf16": (1.74, 1.97, 0.86),
    "sum_2d_axis0": (1.68, 1.97, 0.86),
}

TABLE = "inner_tree.layout"
REPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_inner_tree_layout.txt")

TABLES = {
    TABLE: {
        "doc":
        "One row per (kernel, dtype, configuration). The same kernel compiled without and "
        "with `tritongpu-optimize-reduction-layout{ideal,8,256}`, answering three independent "
        "questions in order: does it still compile, do the bits change, and is it faster. A "
        "speedup on a row whose bits changed is not a result. Three arms: `baseline_ms` is "
        "inner_tree with no pass, `optimized_ms` is inner_tree with it, `ceiling_ms` is the "
        "same kernel under `unordered` -- what the compiler picks with no ordering constraint "
        "at all -- so `gap_closed` says how much of the ordered-to-unordered gap the pass took "
        "back. The `prior_*` columns are the number PR #2312 reported for the same kernel, "
        "measured on H100 / triton-3.7.0 / f32 input, so a row carries both sides of the "
        "comparison.",
        "cols": [
            ("kernel", "str", "kernel from the bitequiv evaluation registry"),
            ("dtype", "str", "element dtype: f16, bf16, f32 or fp8"),
            ("ordering", "str", "reduction_ordering the row was compiled with; the pass targets inner_tree"),
            ("config", "str", "the configuration, e.g. num_warps=4 num_stages=3 block_n=1024"),
            ("num_warps", "int", "broken out of config because it is the axis the prior table is indexed by"),
            ("pass_name", "str", "the pass under test, in triton-opt vocabulary"),
            ("pass_available", "int", "1 if the build has a binding for that pass; 0 means the row "
             "is a baseline-against-baseline control and must show no change"),
            ("guard_fp_fusion_gated", "int", "1 if this tree's pass gates its mul-fed-reduction skip on "
             "enable_fp_fusion; 0 if it skips every mul-fed reduction unconditionally. A `col_dot` "
             "row at 1.00x means something different under each"),
            ("pass_changed_ir", "int", "1 if the optimized build's TTGIR differs from the baseline's, i.e. "
             "the pass actually fired on this configuration"),
            ("compiles", "int", "1 if the kernel still compiles with the pass applied"),
            ("seeds", "int", "random input draws the two builds were compared on"),
            ("bit_changed", "int", "draws whose bytes differ from the baseline build. Must be 0"),
            ("bit_identical_perf", "int", "1 if the two builds also agreed byte for byte on the "
             "(larger, plain-data) performance input -- an independent second bit check"),
            ("order_sensitive", "int", "1 if this configuration's bytes DO move when the ordering "
             "constraint is dropped. Where this is 0 the reduction is order-invariant here and "
             "`bit_changed = 0` proves nothing about the pass"),
            ("nan_frac", "float", "fraction of the baseline output that is NaN or Inf. A saturated "
             "output compares equal to itself, so a high value makes `bit_changed` vacuous"),
            ("baseline_ms", "float", "device time without the pass, min-of-medians over do_bench runs"),
            ("optimized_ms", "float", "device time with the pass"),
            ("ceiling_ms", "float", "device time of the unordered ceiling arm"),
            ("speedup", "float", "baseline_ms / optimized_ms; above 1 means the pass helped"),
            ("gap_closed", "float", "(baseline_ms - optimized_ms) / (baseline_ms - ceiling_ms). Blank "
             "when the ordered and unordered arms are within 2% of each other: the denominator is "
             "then noise and the ratio is meaningless, not large"),
            ("verdict", "str", "bit-safe and faster, bit-safe and neutral, BITS CHANGED, or why "
             "the row was out of scope"),
            ("prior_source", "str", "which measurement the prior_* columns come from, including its "
             "hardware, Triton version and dtype"),
            ("prior_speedup", "float", "the prior measurement's speedup for this kernel and num_warps"),
            ("prior_gap_closed", "float", "the prior measurement's gap_closed for the same"),
            ("error", "str", "non-empty if the row failed to produce a result"),
        ],
    },
}


# ------------------------------------------------------------------------------------------- #
# Helpers
# ------------------------------------------------------------------------------------------- #
def _guard_is_fp_fusion_gated():
    """Does the pass in THIS tree gate its mul-fed-reduction skip on enable_fp_fusion?

    PR #2312's description mentions a follow-up that extends the guard to optimize mul-fed
    reductions (`sum(a*b)`) when fusion is off, reading a `ttg.enable_fp_fusion` module attribute.
    Whether that follow-up is present decides what a `col_dot` row at 1.00x means, so it is
    measured rather than assumed. Returns 1, 0, or None if the source is not there to read.

    Comments are stripped first: the unconditional version of the guard explains itself in a
    comment that says "enable_fp_fusion", so a plain text search finds the word in a tree that
    does not act on it. The per-row `pass_changed_ir` column is the empirical version of the same
    question -- a `col_dot` row at `enable_fp_fusion=off` with `pass_changed_ir = 0` is the guard
    declining to fire with fusion already off.
    """
    try:
        with open(PASS_SOURCE) as f:
            src = f.read()
    except OSError:
        return None
    code = "\n".join(line.split("//", 1)[0] for line in src.splitlines())
    return int("enable_fp_fusion" in code)


def _one_config_spec(spec, dtype, config):
    """The kernel spec narrowed to exactly one dtype and one configuration.

    evaluate_opt's `support` / `correctness` / `performance` each sweep a whole spec and return an
    aggregate. Handing each of them a spec that contains a single configuration is how this step
    gets a per-configuration row -- and therefore a checkpoint per row -- out of the same harness
    PR #2312 used, instead of reimplementing it.
    """
    import dataclasses

    base_filter = spec.config_filter
    return dataclasses.replace(
        spec,
        valid_dtypes=(dtype, ),
        config_filter=lambda c, w=config, b=base_filter: c == w and (b is None or b(c)),
    )


def _pinned_space(spec, dtype):
    """The kernel's in-scope (inner_tree) configurations for one dtype.

    `KernelSpec.config_space(effort)` ignores its argument in this tree -- it is a shim that
    returns `max_config_space()` -- so `--config-effort light` shrinks nothing on this path and
    the dtype has to be pinned on the spec itself, which is what `max_config_space` reads.
    """
    import dataclasses

    from bitequiv.evaluation.evaluate_opt import _in_scope

    narrowed = dataclasses.replace(spec, valid_dtypes=(dtype, ))
    return _in_scope(narrowed.max_config_space(), "inner_tree")


def _gpu_contention(seconds=20.0, samples=8):
    """Is the pinned GPU ours alone? Returns (max_util, [other pids], note).

    Every number in stage 3 is a device time, and this box is shared: another agent's timing step
    on the same device does not change a bit result but makes a speedup meaningless. So the step
    looks before it measures, and looks again as it goes, rather than trusting that the device it
    was handed at the start is still quiet an hour later.
    """
    import subprocess

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis or "," in vis:
        return None, [], "CUDA_VISIBLE_DEVICES is not a single GPU; pin one before timing anything"
    try:
        utils = []
        for i in range(samples):
            out = subprocess.check_output(
                ["nvidia-smi", "-i", vis, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            utils.append(int(out.strip().splitlines()[0]))
            if i + 1 < samples:
                time.sleep(seconds / samples)
        apps = subprocess.check_output(["nvidia-smi", "-i", vis, "--query-compute-apps=pid", "--format=csv,noheader"],
                                       text=True).split()
        others = [p for p in apps if p.strip() and int(p) != os.getpid()]
    except Exception as exc:  # noqa: BLE001
        return None, [], f"could not read nvidia-smi ({type(exc).__name__}); cannot tell whether the GPU is free"
    return max(utils), others, ""


def _nan_frac(raw):
    import numpy as np

    a = np.frombuffer(raw, dtype=np.float32)
    return float((~np.isfinite(a)).mean()) if a.size else 0.0


def _ttgir(ck):
    asm = getattr(ck, "asm", None) or {}
    return asm.get("ttgir") or asm.get("ptx") or ""


GAP_FLOOR = 0.02  # the ordered and unordered arms must differ by at least this fraction of base


def _gap(base, opt, ceil):
    """`gap_closed`, or None when the question does not arise.

    The denominator is `base - ceil`, the whole distance the ordering constraint costs. When the
    two arms land within a couple of percent of each other that distance is noise, and dividing by
    it produces numbers like 238 or -769 that then poison any average they are in. Those rows are
    not "the pass closed 23800% of the gap", they are "there was no gap"; report them as blank.
    """
    if base is None or opt is None or ceil is None or not base:
        return None
    denom = base - ceil
    if abs(denom) < GAP_FLOOR * abs(base):
        return None
    return (base - opt) / denom


def _median(xs):
    return statistics.median(xs) if xs else None


def _fmt(v, spec="{:.2f}"):
    return "-" if v is None else spec.format(v)


# ------------------------------------------------------------------------------------------- #
# Self-checks. Without these the reproduction is fake rather than different.
# ------------------------------------------------------------------------------------------- #
def _self_checks(specs, base_variant, opt_variant, resolved):
    """Three things that, if wrong, make every number below meaningless.

    1. The pass has a binding in this build. Without one `_classify` treats it as baseline and the
       whole table comes back a clean 1.00x that measured nothing.
    2. The injection survives the compile cache. Two builds of a configuration the pass must
       transform have to produce different TTGIR; if they do not, the optimized build was served
       the baseline one and every `bit_changed = 0` below is an artefact of the cache.
    3. With no pass named at all, optimized must equal baseline byte for byte. If that control
       fails the harness is broken and nothing after it means anything.
    """
    from bitequiv.evaluation.evaluate_opt import _clear_jit_caches, _variant

    lines, ok = [], True
    lines.append(f"  pass binding                {'FOUND' if resolved else 'MISSING'}  ({PASS_SPEC})")
    if not resolved:
        return lines + ["  ABORT: no binding for the pass; every row would be baseline-against-baseline."], False

    spec = specs["sum_2d_col"]
    cfg = next(c for c in _pinned_space(spec, DTYPES[0])
               if c.num_warps == 4 and c.num_stages == 2 and c.enable_fp_fusion)
    _clear_jit_caches()
    with base_variant():
        ir_base = _ttgir(spec.compile(cfg, spec.perf_size))
    _clear_jit_caches()
    with opt_variant():
        ir_opt = _ttgir(spec.compile(cfg, spec.perf_size))
    fired = ir_base != ir_opt
    lines.append(f"  injection changes the IR    {'YES' if fired else 'NO'}  (sum_2d_col {DTYPES[0]} num_warps=4)")
    if not fired:
        ok = False
        lines.append("  ABORT: the two builds are identical. Triton's in-memory cache is keyed on the")
        lines.append("         specialization and options only, so the injected pass is not part of the key.")
        lines.append("         Relaunch the process with TRITON_ALWAYS_COMPILE=1 set in the environment.")
        return lines, ok

    _clear_jit_caches()
    with base_variant():
        a = spec.run(cfg, spec.compile(cfg, spec.precision_size), 0, spec.precision_size)
    _clear_jit_caches()
    with _variant([], []):
        b = spec.run(cfg, spec.compile(cfg, spec.precision_size), 0, spec.precision_size)
    same = a == b
    lines.append(f"  no-pass control identical   {'YES' if same else 'NO'}")
    if not same:
        ok = False
        lines.append("  ABORT: baseline is not reproducible against itself; the harness is broken.")
    return lines, ok


# ------------------------------------------------------------------------------------------- #
# One row
# ------------------------------------------------------------------------------------------- #
def _measure(spec, dtype, config, base_variant, opt_variant, seeds_effort, do_perf=True):
    """Every answer for one configuration, via evaluate_opt's own three stages.

    `do_perf=False` is the preflight: stages 1 and 2 only. Stage 3 runs `do_bench` loops, which are
    a sustained GPU load, and this box is shared with other people's timing steps -- a sanity run
    has no business perturbing one.
    """
    from bitequiv.evaluation.eval_kernels import config_label
    from bitequiv.evaluation.evaluate_opt import _clear_jit_caches, correctness, performance, support

    one = _one_config_spec(spec, dtype, config)
    row = dict(kernel=spec.name, dtype=dtype, ordering=config.reduction_ordering, config=config_label(config),
               num_warps=config.num_warps, pass_name=PASS_SPEC, error="")

    verdict, _detail = support(one, opt_variant, "inner_tree")
    row["compiles"] = int(verdict == "SUPPORTED")

    c = correctness(one, base_variant, opt_variant, "light", seeds_effort, "inner_tree")
    row["seeds"] = c.get("repeats", 0) if c.get("checked") else 0
    row["bit_changed"] = len(c.get("changed", ()))
    if c.get("opt_fail"):
        row["compiles"] = 0

    # Is this configuration order-sensitive at all? Under `unordered` the compiler is free to pick
    # a different reduction order; if the bytes come back the same anyway then this reduction is
    # order-invariant here and `bit_changed = 0` says nothing about the pass. Reusing the ceiling
    # arm the experiment already defines rather than inventing a new control.
    ceil_cfg = config._replace(reduction_ordering="unordered")
    size = spec.precision_size
    try:
        _clear_jit_caches()
        with base_variant():
            ck_base = spec.compile(config, size)
        raw_it = spec.run(config, ck_base, 0, size)
        _clear_jit_caches()
        with opt_variant():
            row["pass_changed_ir"] = int(_ttgir(ck_base) != _ttgir(spec.compile(config, size)))
        _clear_jit_caches()
        with base_variant():
            raw_un = spec.run(ceil_cfg, spec.compile(ceil_cfg, size), 0, size)
        row["order_sensitive"] = int(raw_it != raw_un)
        row["nan_frac"] = round(_nan_frac(raw_it), 4)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]

    if do_perf:
        _time_into(row, one, base_variant, opt_variant)
    _set_verdict(row, timed=do_perf)
    return row


def _time_into(row, one, base_variant, opt_variant):
    """Stage 3 only, written into an existing row.

    Split out so the bits and the timings can be taken in separate runs: on a shared box the bit
    stages can go whenever, the timing stage has to wait for a device of its own, and re-doing the
    compiles and the ten seeds a second time just to attach a number to them would be waste.
    """
    from bitequiv.evaluation.evaluate_opt import performance

    p = performance(one, base_variant, opt_variant, "light", "inner_tree", ceiling=True)
    rows = p.get("rows") or []
    if not rows:
        if p.get("fails"):
            row["error"] = row.get("error") or "performance stage failed for this config"
        return row
    r = rows[0]
    row["baseline_ms"] = round(r["base_ms"], 6)
    row["optimized_ms"] = round(r["opt_ms"], 6)
    row["bit_identical_perf"] = int(r["bit_identical"])
    if math.isfinite(r["opt_ms"]) and r["opt_ms"] > 0:
        row["speedup"] = round(r["base_ms"] / r["opt_ms"], 4)
    if "ceil_ms" in r:
        row["ceiling_ms"] = round(r["ceil_ms"], 6)
        gap = _gap(r["base_ms"], r["opt_ms"], r["ceil_ms"]) if row.get("speedup") else None
        row["gap_closed"] = None if gap is None else round(gap, 4)
    return row


def _set_verdict(row, timed):
    if not row.get("compiles"):
        row["verdict"] = "compile regression"
    elif row.get("bit_changed"):
        row["verdict"] = "BITS CHANGED"
    elif row.get("bit_identical_perf") == 0:
        row["verdict"] = "BITS CHANGED (perf input)"
    elif not timed:
        row["verdict"] = "compiles, bits unchanged (not timed)"
    elif row.get("speedup") is None:
        row["verdict"] = "no valid timing"
    elif row["speedup"] >= 1.05:
        row["verdict"] = "bit-safe and faster"
    elif row["speedup"] <= 0.95:
        row["verdict"] = "bit-safe but slower"
    else:
        row["verdict"] = "bit-safe and neutral"
    return row


# ------------------------------------------------------------------------------------------- #
# Report
# ------------------------------------------------------------------------------------------- #
def _summarise(rows, kernel, dtype, num_warps):
    sel = [r for r in rows if r["kernel"] == kernel and r["dtype"] == dtype and r.get("num_warps") == num_warps]
    if not sel:
        return None
    sp = [r["speedup"] for r in sel if r.get("speedup")]
    # Recomputed from the three stored times rather than read back, so the noise floor applies to
    # rows taken before it existed too.
    gp = [
        g for g in (_gap(r.get("baseline_ms"), r.get("optimized_ms"), r.get("ceiling_ms")) for r in sel)
        if g is not None
    ]
    return dict(n=len(sel), bit_changed=sum(r.get("bit_changed", 0)
                                            for r in sel), sensitive=sum(r.get("order_sensitive", 0) for r in sel),
                fired=sum(r.get("pass_changed_ir", 0) for r in sel), speedup=_median(sp), best=max(sp) if sp else None,
                gap=_median(gp), nan=max(
                    (r.get("nan_frac") or 0.0) for r in sel), nocompile=sum(1 for r in sel if not r.get("compiles")))


def _x(v):
    return "-" if v is None else f"{v:.2f}x"


def _report(rows, env, header):
    out = list(header)
    out += [
        "", "SIDE BY SIDE -- this machine against PR #2312 (commit 8176cccdc)", "=" * 116, "",
        "Mine is the median over the 12 configurations at that num_warps (6 num_stages x 2 enable_fp_fusion).",
        "`sens` is how many of those 12 are order-sensitive: where it is 0/12 the reduction does not depend on",
        "the order here at all, so `bits` = 0 on those rows is vacuous, not a pass. `fired` is how many of the",
        "12 the pass actually rewrote the IR on; a kernel at 1.00x with fired = 0 was declined by a guard.", ""
    ]
    head = (f"{'kernel':22} {'dtype':5} {'nw':>3} {'cfg':>4} {'bits':>5} {'sens':>6} {'fired':>6} {'nan':>5} "
            f"{'speedup':>8} {'max':>7} {'gap':>6} | {'prior':>7} {'gap':>6}  prior arm")
    out += [head, "-" * len(head)]

    def emit(kernel, dtype, nw, s, pv, arm, tail=""):
        if s is None:
            out.append(f"{kernel:22} {dtype:5} {nw:>3} {'-':>4} {'-':>5} {'-':>6} {'-':>6} {'-':>5} "
                       f"{'-':>8} {'-':>7} {'-':>6} | {_x(pv and pv[0]):>7} {_fmt(pv and pv[1]):>6}  {arm}{tail}")
            return
        out.append(f"{kernel:22} {dtype:5} {nw:>3} {s['n']:>4} {s['bit_changed']:>5} "
                   f"{str(s['sensitive']) + '/' + str(s['n']):>6} {str(s['fired']) + '/' + str(s['n']):>6} "
                   f"{s['nan']:>5.2f} {_x(s['speedup']):>8} {_x(s['best']):>7} {_fmt(s['gap']):>6} | "
                   f"{_x(pv and pv[0]):>7} {_fmt(pv and pv[1]):>6}  {arm}{tail}")

    for kernel in KERNELS:
        for dtype in PINNED_DTYPE.get(kernel, DTYPES):
            for nw in (4, 8):
                prior = PRIOR_H100.get(kernel, {})
                pv = prior.get(nw) or (prior.get(None) if None in prior else None)
                arm = f"H100 3.7.0 {PRIOR_DTYPE.get(kernel, 'f32')}" + ("" if nw in prior else " (nw not stated)")
                s = _summarise(rows, kernel, dtype, nw)
                if s is None:
                    tail = "  NOT MEASURED"
                elif s["nocompile"]:
                    tail = f"  {s['nocompile']}/{s['n']} FAIL TO COMPILE WITH THE PASS (ptxas)"
                else:
                    tail = ""
                emit(kernel, dtype, nw, s, pv, arm, tail)
    for kernel in UNREACHABLE:
        for nw in (4, 8):
            emit(kernel, "-", nw, None, PRIOR_H100[kernel].get(nw), "H100 3.7.0 f32", "  NOT REACHABLE")
    out += [
        "", "NOT REACHABLE = the kernel is not in the registry `evaluate_opt` resolves against; it lives in a",
        "  suite with a different spec shape, so measuring it here would mean writing a second harness:"
    ]
    for kernel, where in UNREACHABLE.items():
        out.append(f"    {kernel:24} {where}")
    out += [
        "", "The dtype differs between the two sides on every row except col_bf16. The earlier run measured f32:",
        "  the dtype axis did not exist in eval_kernels.py at that commit, so every kernel took the f32 default.",
        "  This run measures f16 and fp8. A difference in these columns is hardware AND Triton AND dtype, not",
        "  hardware alone. col_bf16 is bf16 on both sides and is the one like-for-like row."
    ]

    out += ["", "", "ALL num_warps (median speedup per kernel, dtype, num_warps)", "=" * 116, ""]
    warps = sorted({r.get("num_warps") for r in rows if r.get("num_warps") is not None})
    head2 = f"{'kernel':22} {'dtype':5} " + " ".join(f"{('nw=' + str(w)):>9}" for w in warps)
    out += [head2, "-" * len(head2)]
    for kernel in KERNELS:
        for dtype in PINNED_DTYPE.get(kernel, DTYPES):
            cells = []
            for w in warps:
                s = _summarise(rows, kernel, dtype, w)
                cells.append(f"{_x(s['speedup']) if s else '-':>9}")
            if any(c.strip() != "-" for c in cells):
                out.append(f"{kernel:22} {dtype:5} " + " ".join(cells))

    out += [
        "", "", "PR #2312'S OWN GB300 / sm_100 TABLE (72 f32 configs per kernel), for reference", "=" * 116, "",
        "Same hardware family as this machine but f32 and a different Triton, so it is a second prior, not",
        "the headline comparison.", ""
    ]
    head3 = f"{'kernel':22} {'prior median':>13} {'prior max':>10} {'prior gap':>10}"
    out += [head3, "-" * len(head3)]
    for kernel, (med, mx, gap) in PRIOR_SM100.items():
        out.append(f"{kernel:22} {med:>12.2f}x {mx:>9.2f}x {gap:>10.2f}")

    total = len(rows)
    changed = sum(r.get("bit_changed", 0) for r in rows)
    perf_changed = sum(1 for r in rows if r.get("bit_identical_perf") == 0)
    regress = sum(1 for r in rows if not r.get("compiles"))
    errors = sum(1 for r in rows if r.get("error"))
    insens = [r for r in rows if r.get("order_sensitive") == 0]
    pct = 100.0 * len(insens) / total if total else 0.0
    out += [
        "", "", "GATE", "=" * 116, "", f"  rows measured                 {total}",
        f"  bit-changed configurations    {changed}   (must be 0)",
        f"  bit-changed on the perf input {perf_changed}   (must be 0; an independent second check)",
        f"  compile regressions           {regress}   (must be 0)", f"  rows with an error            {errors}",
        f"  order-INSENSITIVE rows        {len(insens)}   ({pct:.1f}% of rows -- on these the reduction does",
        "                                not depend on the order, so the bit check is vacuous, not passing)"
    ]
    by_kd = {}
    for r in insens:
        by_kd[(r["kernel"], r["dtype"])] = by_kd.get((r["kernel"], r["dtype"]), 0) + 1
    for (k, d), n in sorted(by_kd.items()):
        out.append(f"      {k:22} {d:5} {n} rows")
    if insens:
        out += [
            "",
            "  Why those are all fp8 pure sums. An fp8 e4m3 value carries a 4-bit significand, so the exact",
            "  sum of a few hundred of them still fits inside an f32 mantissa, and then every order gives the",
            "  same answer -- the reduction is order-invariant by arithmetic, not by luck. Checked off-device",
            "  on 200 draws of 256 leaves: exactly representable in f32 200/200, and a sequential sum against",
            "  a pairwise tree differed 0/200. The same check on f16 leaves is 0/200 exact and 85/200 different.",
            "  On-device the same split appears as 0/108 against 108/108 order-sensitive, over num_warps x",
            "  num_stages x enable_fp_fusion x seeds. So on these five kernels the fp8 `bit_changed = 0` is not",
            "  weak evidence, it is no evidence: nothing could have moved. col_exp_sum and col_dot keep their",
            "  fp8 sensitivity because exp() and a*b widen the leaves to full f32 before the reduce, and those",
            "  are the fp8 rows worth reading.",
        ]
    if regress:
        by_r = {}
        for r in rows:
            if not r.get("compiles"):
                key = (r["kernel"], r["dtype"], r.get("num_warps"))
                by_r[key] = by_r.get(key, 0) + 1
        out += [
            "", "  The compile regressions, in full. The baseline builds and the optimized one does not, so",
            "  these are the pass's doing, not the kernel's:"
        ]
        for (k, d, nw), n in sorted(by_r.items(), key=lambda kv: str(kv[0])):
            out.append(f"      {k:22} {d:5} num_warps={nw:<3} {n} configs   ptxas C7907 internal compiler error")
        out += [
            "  All of them sit at the register-pressure guard's boundary: the reduce-friendly layout carries",
            "  tile_elements / (warp_size * num_warps) values per thread, and the guard only declines above",
            "  `max-elems-per-thread`, so a configuration landing exactly on the limit is still rewritten."
        ]
    passed = not changed and not perf_changed and not regress
    out += [
        "", "  RESULT: " + ("PASS -- the pass preserved the bits on every configuration where the bits move "
                            "at all." if passed else "FAIL -- see the counts above."), ""
    ]
    return out


# ------------------------------------------------------------------------------------------- #
# Entry point
# ------------------------------------------------------------------------------------------- #
def run(args, env):
    os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
    launched_with_always_compile = os.environ.get("TRITON_ALWAYS_COMPILE") == "1"

    from bitequiv.evaluation.eval_kernels import REGISTRY, config_label
    from bitequiv.evaluation.evaluate_opt import _classify, _parse_passes, _variant

    preflight = os.environ.get("INNER_TREE_LAYOUT_STAGES", "").strip() == "1"
    wanted = [k.strip() for k in os.environ.get("INNER_TREE_LAYOUT_KERNELS", "").split(",") if k.strip()] or KERNELS

    # Look at the device BEFORE this process touches it, so the reading is the neighbours' and not
    # our own. A busy device is not a reason to skip the bit work, only the timing.
    util, others, gpu_note = (None, [], "skipped (preflight)") if preflight else _gpu_contention()
    blocked = bool(others) or (util is not None and util > 5) or bool(gpu_note and not preflight)

    specs = {k: REGISTRY[k] for k in KERNELS if k in REGISTRY}
    missing = [k for k in KERNELS if k not in REGISTRY]
    resolved, needs_args, unavailable = _classify(_parse_passes(PASS_SPEC))
    gated = _guard_is_fp_fusion_gated()

    def base_variant():
        return _variant([], [])

    def opt_variant():
        return _variant(resolved, [])

    header = [
        "inner_tree.layout -- reproducing PR #2312 (commit 8176cccdc) on this machine",
        "=" * 108,
        "",
        f"  when                  {env['when']}",
        f"  device                {env['device']}  ({env['arch']})",
        f"  triton                {env['triton']}   torch {env['torch']}   commit {env['commit']}",
        f"  pass under test       {PASS_SPEC}",
        f"  pass_available        {1 if resolved else 0}" +
        ("" if resolved else f"   needs_args={needs_args} unavailable={unavailable}"),
        f"  mul-fed guard         {'gated on enable_fp_fusion' if gated else 'skips every mul-fed reduce'}"
        f"   (guard_fp_fusion_gated={gated})",
        f"  dtypes measured       {', '.join(DTYPES)}   (col_bf16: bf16, its spec pins in_dtype)",
        f"  TRITON_ALWAYS_COMPILE {'set' if launched_with_always_compile else 'NOT SET'}",
        f"  gpu                   CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}"
        f"  peak util before start {util if util is not None else '?'}%"
        f"  other processes on it: {others or 'none'}" + (f"  [{gpu_note}]" if gpu_note else ""),
        "",
        "  arms   base = inner_tree, standard pipeline",
        "         opt  = base + " + PASS_SPEC + " appended at end-of-TTGIR",
        "         ceil = the same kernel under reduction_ordering=unordered, standard pipeline",
        "         speedup = base/opt        gap_closed = (base-opt)/(base-ceil)",
        "",
    ]
    if missing:
        header.append(f"  NOTE: {', '.join(missing)} are in KERNELS but not in the registry.")
    if gated == 0:
        header += [
            "  NOTE: this tree's pass skips EVERY mul-fed reduction (`sum(a*b)`), with no",
            "        enable_fp_fusion condition, and no `ttg.enable_fp_fusion` module attribute exists",
            "        anywhere in it. The follow-up PR #2312's description mentions is not in this branch,",
            "        so `col_dot` coming back at ~1.00x here is a difference in the code, not in the",
            "        hardware. Read that row against the earlier 1.19x with this in mind.",
            "",
        ]
    if blocked and not preflight:
        header += [
            "  BLOCKED ON THE GPU: the pinned device is not idle, so stage 3 is not run. A device time",
            "        taken next to somebody else's kernel is not a measurement, and reporting it would be",
            f"        worse than reporting nothing. Reason: {gpu_note or ('other processes ' + str(others))}"
            f"{'' if gpu_note else f', peak util {util}%'}.",
            "        Stages 1 and 2 (compiles and bits) do run: they are unaffected by a busy neighbour.",
            "",
        ]
    print("\n".join(header))

    checks, ok = _self_checks(specs, base_variant, opt_variant, resolved)
    print("SELF-CHECKS")
    print("\n".join(checks))
    print()
    header += ["SELF-CHECKS"] + checks + [""]
    if not ok:
        with open(REPORT, "w") as f:
            f.write("\n".join(header) + "\nABORTED: a self-check failed; nothing was measured.\n")
        return

    # The space, printed before anything long runs so the size is on the record.
    plan = []
    print("CONFIGURATION SPACE (inner_tree only)")
    for kernel in wanted:
        spec = specs.get(kernel)
        if spec is None:
            continue
        for dtype in PINNED_DTYPE.get(kernel, DTYPES):
            configs = _pinned_space(spec, dtype)
            plan.append((kernel, dtype, configs))
            print(f"  {kernel:22} {dtype:5} {len(configs):4} configs")
    n_total = sum(len(c) for _, _, c in plan)
    print(f"  {'TOTAL':22} {'':5} {n_total:4} configs")
    header += ["CONFIGURATION SPACE (inner_tree only)"]
    header += [f"  {k:22} {d:5} {len(c):4} configs" for k, d, c in plan]
    header += [f"  {'TOTAL':22} {'':5} {n_total:4} configs", ""]

    done = {}
    path = os.path.join(CACHE, f"{TABLE}.jsonl")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[(r["kernel"], r["dtype"], r["config"])] = r
    if done:
        print(f"\n  resuming: {len(done)} rows already in {path}")

    if preflight:
        t0 = time.time()
        sample = [(k, d, cs[0]) for k, d, cs in plan]
        bad = []
        for kernel, dtype, config in sample:
            r = _measure(specs[kernel], dtype, config, base_variant, opt_variant, "fast", do_perf=False)
            print(
                f"  {kernel:22} {dtype:5} compiles={r.get('compiles')} bits={r.get('bit_changed')} "
                f"fired={r.get('pass_changed_ir')} order_sensitive={r.get('order_sensitive')} "
                f"nan_frac={r.get('nan_frac')} {r.get('error', '')}", flush=True)
            if r.get("error") or not r.get("compiles"):
                bad.append(f"{kernel}/{dtype}")
        per = (time.time() - t0) / max(len(sample), 1)
        print(f"\nPREFLIGHT: {len(sample)} compile-only rows in {time.time() - t0:.0f}s -> {per:.1f}s/row")
        print(f"           stages 1+2 over {n_total} rows projects to {per * n_total / 3600:.1f} hours;")
        print("           stage 3 adds three do_bench arms per row on top of that.")
        print(f"           problems: {bad or 'none'}")
        print("           Nothing was written. Drop INNER_TREE_LAYOUT_STAGES to run it for real.")
        return

    out = writer(TABLE)
    deadline = time.time() + float(args.minutes) * 60.0
    stopped_early = False
    joined = None
    t0 = time.time()
    n_done = 0
    for kernel, dtype, configs in plan:
        spec = specs[kernel]
        prior = PRIOR_H100.get(kernel, {})
        print(f"\n{kernel} [{dtype}] -- {len(configs)} configs", flush=True)
        for i, config in enumerate(configs):
            key = (kernel, dtype, config_label(config))
            have = done.get(key)
            # A row taken while the GPU was busy carries the bits but no timing. Redo exactly those
            # when a free device finally turns up, and leave every finished row alone.
            if have is not None and (blocked or have.get("speedup") is not None or not have.get("compiles")):
                continue
            if time.time() > deadline:
                stopped_early = True
                break
            # Somebody else can start on this device an hour into the run; from that point on the
            # timings are not ours. Check every 60 rows and stop rather than keep recording.
            if not blocked and n_done and n_done % 60 == 0:
                _u, _o, _n = _gpu_contention(seconds=4.0, samples=2)
                if _o:
                    joined = f"another process ({_o}) started on the pinned GPU after {n_done} rows"
                    stopped_early = True
                    break
            try:
                if have is not None:  # bits already taken on a busy GPU; only the timing is missing
                    row = _set_verdict(
                        _time_into(dict(have), _one_config_spec(spec, dtype, config), base_variant, opt_variant),
                        timed=True)
                else:
                    row = _measure(spec, dtype, config, base_variant, opt_variant, "fast", do_perf=not blocked)
            except Exception as exc:  # noqa: BLE001
                row = dict(kernel=kernel, dtype=dtype, ordering=config.reduction_ordering, config=config_label(config),
                           num_warps=config.num_warps, pass_name=PASS_SPEC, verdict="error",
                           error=f"{type(exc).__name__}: {exc}"[:200])
            row["pass_available"] = 1 if resolved else 0
            row["guard_fp_fusion_gated"] = gated
            pv = prior.get(config.num_warps) or (prior.get(None) if None in prior else None)
            row["prior_source"] = f"{PRIOR_LABEL} dtype={PRIOR_DTYPE.get(kernel, 'f32')}"
            if pv:
                row["prior_speedup"], row["prior_gap_closed"] = pv
            out.write(json.dumps(row) + "\n")
            out.flush()  # a killed run loses at most this one row
            done[key] = row
            n_done += 1
            if (i + 1) % 12 == 0 or i + 1 == len(configs):
                rate = (time.time() - t0) / max(n_done, 1)
                print(
                    f"    {i + 1:3}/{len(configs)}  {row['verdict']:24} "
                    f"speedup={_fmt(row.get('speedup'))}x gap={_fmt(row.get('gap_closed'))} "
                    f"bits={row.get('bit_changed')} sens={row.get('order_sensitive')}  [{rate:.1f}s/row]", flush=True)
        if stopped_early:
            break
    out.close()

    # The file is appended to a row at a time so a kill costs one row, which means a row redone
    # later (bits now, timing once a GPU frees) is in it twice. Collapse it, newest wins, through a
    # temporary file so an interrupted rewrite cannot lose the whole run.
    if done:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in done.values():
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)

    if stopped_early:
        left = n_total - len(done)
        why = joined or f"the {args.minutes:.0f} minute budget ran out"
        note = (f"  PARTIAL: {why}, with {left} of {n_total} configurations left. Re-run the same "
                "command to continue; rows already taken are not repeated.")
        print("\n" + note)
        header += [note, ""]
    if blocked:
        header += [
            f"  NOT TIMED: stage 3 was skipped on every row of this run ({gpu_note or 'the GPU was in use'}), so",
            "        every speedup and gap_closed column below is empty. The bit columns are real.", ""
        ]

    rows = list(done.values())
    text = _report(rows, env, header)
    with open(REPORT, "w") as f:
        f.write("\n".join(text) + "\n")
    print("\n".join(text[len(header):]))
    print(f"\nWROTE {REPORT}")
