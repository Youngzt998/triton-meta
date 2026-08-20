"""inner_tree.layout -- the reduction layout-optimization pass: is it bit-safe, and what does it
buy?

This re-runs, on this machine, the measurement reported with the pass itself: commit `8176cccdc`,
PR #2312, "M2: reduction-layout optimization pass". Same kernels, same three arms, same
`gap_closed`. What differs from the earlier run is the machine (GB300 / sm_103 instead of H100),
the Triton (3.8.0 instead of 3.7.0), the input dtype on most rows, and one detail of the pass's
own source; each of those is printed next to the earlier number so a reader can see which axis
moved.

    arms      base = reduction_ordering=inner_tree, standard pipeline
              opt  = the same, plus `tritongpu-optimize-reduction-layout{ideal,8,256}` appended
                     at end-of-TTGIR
              ceil = the same kernel, the same configuration, `reduction_ordering=unordered`,
                     standard pipeline -- what the compiler picks with the ordering constraint
                     dropped and nothing else changed
    metric    speedup    = base_ms / opt_ms
              gap_closed = (base_ms - opt_ms) / (base_ms - ceil_ms);  1.0 = opt reached ceil

The ceiling is deliberately the *same configuration* with one flag flipped. Only two things vary
in this experiment -- is `inner_tree` on, and is the pass on -- and everything else is held
fixed. A ceiling taken as the best unordered time over the swept configurations would fold an
autotuner's freedom into the denominator and make `gap_closed` mean two things at once; that
question belongs to the `gemm.perf.*` steps, which have an arm for it.

Read `bit_changed` before any speedup. A speedup on a row whose bytes moved is not a result.

Attribution. The ordering path this pass sits on top of is not ours: the `reduction_ordering` /
`inner_tree` mechanism, and the `TRITON_STRICT_REDUCTION_ORDERING` environment variable that pins
it, were written by Nick Riasanovsky, a co-author. The pass under test is
`tritongpu-optimize-reduction-layout`, added in commit `8176cccdc` (PR #2312).

HOW TO RUN

    export PYTHONPATH=$(git rev-parse --show-toplevel)
    CUDA_VISIBLE_DEVICES=<an idle gpu> python artifact_eval/artifact.py --run inner_tree.layout \
        --minutes 600

Nothing else has to be set. `TRITON_ALWAYS_COMPILE=1` is required and this module sets it when it
is imported, which happens before `artifact.py` imports triton; setting it later would be too
late. It is required because Triton's on-disk cache key does not contain the injected pass, so
without it the pass-on build can be served the pass-off one. The in-memory cache is a separate
problem with a separate fix -- see `steps/_inner_tree_kernels.clear_caches`. Neither fix is
trusted: the step refuses to measure anything until it has watched two builds of one
configuration actually produce different TTGIR.

    INNER_TREE_LAYOUT_PREFLIGHT=1     self-checks, the kernel table, the configuration counts and
                                      a timing projection. One configuration per (kernel, dtype),
                                      no benchmarking, writes nothing. Minutes, not hours.
    INNER_TREE_LAYOUT_KERNELS=a,b     restrict the kernel list (debugging).
    INNER_TREE_LAYOUT_DTYPES=f16,f32  restrict the dtypes; a kernel keeps only the dtypes it
                                      supports, so this can empty a kernel out.
    INNER_TREE_LAYOUT_SEEDS=10        input draws the two builds are byte-compared on.
    INNER_TREE_LAYOUT_BENCH_REPS=3    `do_bench` runs per arm; the reported time is the smallest
                                      of their medians. Lower is faster and noisier.
    INNER_TREE_LAYOUT_PATIENCE_MIN=45 how long to pause, rather than stop, when another process
                                      appears on the pinned GPU mid-run. Nothing is measured while
                                      paused; the sweep gives up only if the neighbour outlasts it.
    INNER_TREE_LAYOUT_FORCE_TIMING=1  time even when the pinned GPU is not idle. Only for
                                      debugging the code path -- the numbers are not measurements.
    INNER_TREE_LAYOUT_REPORT_ONLY=1   rebuild the table from the records already in `cache/`.
                                      Launches no kernel and measures nothing; the numbers are
                                      an input on this path, only the prose is re-rendered.

`--minutes` is a resumable budget, not a sample size. Rows stream to
`cache/inner_tree.layout.jsonl` one at a time and a second invocation picks up where the first
stopped, so a short run followed by more runs gives the same table as one long run.

WHY THE RUN PAUSES

If you see a long gap in the log with `PAUSED` on either side of it, nothing hung. This box has
several people on it, and a run that meets a neighbour has two bad options and one good one:
record device times next to somebody else's kernel, which is not a measurement; or stop, which on
a ninety-minute sweep means an interrupted run needs a person to notice and restart it. So it
pauses instead -- it measures and writes nothing while another process is on the pinned device,
then continues where it left off. The report says how many times it paused and for how long, so a
gap is visible rather than silent. This is not hypothetical: the run committed here met a third
user's profiling job partway through.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from collections import namedtuple

# Set before triton is imported. `artifact.py` imports every step module during discovery, which
# happens before it imports torch or triton, so this is the last moment that still works. Without
# it Triton's on-disk cache can hand the pass-on build the pass-off binary, and the whole table
# comes back a perfect, meaningless 1.00x with 0 bits changed.
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

from ._common import CACHE, ROOT, writer  # noqa: E402

NAME = "inner_tree.layout"
ORDER = 70
DESCRIPTION = "is the reduction layout pass bit-safe, and what does it buy"
IMPLEMENTED = True

# Exactly the invocation PR #2312's description documents: strategy "ideal", min-underparallel 8,
# max-elems-per-thread 256 -- the pass Options' own defaults, spelled out so the row records them.
PASS_NAME = "tritongpu-optimize-reduction-layout"
PASS_BINDING = "add_optimize_reduction_layout"
PASS_ARGS = ("ideal", 8, 256)
PASS_SPEC = f"{PASS_NAME}{{{','.join(map(str, PASS_ARGS))}}}"
PASS_SOURCE = os.path.join(ROOT, "lib", "Dialect", "TritonGPU", "Transforms", "OptimizeReductionLayout.cpp")

# The configuration space, per (kernel, dtype). 6 x 6 x 2 = 72, the same 72 the earlier
# GB300 table swept. num_warps is the outer loop so a run that is cut short leaves whole
# num_warps groups finished rather than a ragged edge.
NUM_WARPS = (1, 2, 4, 8, 16, 32)
NUM_STAGES = (1, 2, 3, 4, 5, 6)
FP_FUSION = (True, False)
Config = namedtuple("Config", "num_warps num_stages enable_fp_fusion")

# PR #2312's H100 table, verbatim: {kernel: {num_warps: (speedup, gap_closed)}}. Measured on
# triton-3.7.0 with f32 input except `col_bf16`, whose body pinned bf16 then as it does now.
# `None` is col_bf16's one number, which the table reported without saying which num_warps.
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
PRIOR_LABEL = "H100 3.7.0"

# The same PR also carries a GB300 / sm_100 table -- 72 f32 configurations per kernel, reported as
# median / max / gap_closed. Same hardware family as this machine, so it is the closer of the two
# priors and is printed alongside; the headline comparison is still the H100 one, because that is
# the table that covers all eleven kernels.
PRIOR_SM100 = {
    "sum_3d_outer": (5.15, 5.64, 1.01),
    "col_sum_loop": (2.12, 2.40, 1.00),
    "col_exp_sum": (1.93, 2.27, 0.96),
    "sum_2d_col": (1.92, 2.26, 0.92),
    "sum_2d_col_big": (1.80, 2.15, 0.71),
    "col_bf16": (1.74, 1.97, 0.86),
    "sum_2d_axis0": (1.68, 1.97, 0.86),
}

# Why the pass declines, for the kernels it declines on. Read off the pass source; the measured
# version of the same statement is the `fired` column, which counts the configurations whose
# TTGIR the pass actually rewrote, and the report only prints a note for a kernel `fired` says
# was declined everywhere.
_CONTIGUOUS = ("reduces the CONTIGUOUS innermost axis of its tile, which the compiler already "
               "spreads across the warp, so `isWarpSynchronous` declines it -- there is no "
               "cross-warp stage left to delete")
DECLINE_NOTE = {
    "col_dot": "mul-fed reduce: the pass skips every `arith.mulf`-fed reduce so an FMA "
    "contraction cannot move the bits",
    "A_rms_norm_fwd": _CONTIGUOUS,
    "C_rms_norm_bwd_2reduce": _CONTIGUOUS,
    "D_masked_global_sum": _CONTIGUOUS,
    "E_triu_masked_rowsum": _CONTIGUOUS,
    "H_mean_permute": _CONTIGUOUS,
}

# Kernels whose bytes do not move between the two orderings for a reason that is not the fp8
# significand argument. Only printed for a (kernel, dtype) the run actually measured as
# order-insensitive, so the note explains a measurement rather than predicting one.
INSENSITIVE_NOTE = {
    "D_masked_global_sum":
    "a 1x1024 tile folded to one scalar. A single kept element leaves the "
    "compiler almost no layout to choose, so both orderings lower to the same tree here",
}

TABLE = "inner_tree.layout"
REPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_inner_tree_layout.txt")

TABLES = {
    TABLE: {
        "doc":
        "One row per (kernel, dtype, configuration). The same kernel compiled without and with "
        "`tritongpu-optimize-reduction-layout{ideal,8,256}`, answering three independent "
        "questions in order: does it still compile, do the bits change, and is it faster. A "
        "speedup on a row whose bits changed is not a result. Three arms: `baseline_ms` is "
        "inner_tree with no pass, `optimized_ms` is inner_tree with it, `ceiling_ms` is the same "
        "kernel and the same configuration under `unordered` -- one flag flipped, nothing else "
        "-- so `gap_closed` says how much of the ordered-to-unordered cost the pass took back. "
        "The `prior_*` columns are the number PR #2312 reported for the same kernel and "
        "num_warps, measured on H100 / triton-3.7.0, so a row carries both sides of the "
        "comparison; `like_for_like` says whether the two sides used the same dtype.",
        "cols": [
            ("kernel", "str", "kernel name"),
            ("suite", "str", "which source file it comes from: synthetic, inductor, or zoo"),
            ("kernel_source", "str", "file and symbol of the kernel body, so it can be read"),
            ("dtype", "str", "element dtype of the input: f16, bf16, f32 or fp8"),
            ("ordering", "str", "reduction_ordering the base and opt arms were compiled with"),
            ("config", "str", "the configuration, e.g. num_warps=4 num_stages=3 enable_fp_fusion=on"),
            ("num_warps", "int", "broken out of config because it is the axis the prior table is indexed by"),
            ("num_stages", "int", "broken out of config"),
            ("enable_fp_fusion", "int", "broken out of config; 1 = on"),
            ("pass_name", "str", "the pass under test, in triton-opt vocabulary"),
            ("pass_available", "int", "1 if the build has a binding for that pass; 0 means every row "
             "is a baseline-against-baseline control and the table measured nothing"),
            ("guard_fp_fusion_gated", "int", "1 if this tree's pass gates its mul-fed-reduction skip on "
             "enable_fp_fusion; 0 if it skips every mul-fed reduction unconditionally. A `col_dot` "
             "row at 1.00x means something different under each"),
            ("pass_changed_ir", "int", "1 if the optimized build's TTGIR differs from the baseline's, "
             "i.e. the pass actually fired on this configuration"),
            ("base_compiles", "int", "1 if the baseline arm compiled. 0 means the configuration is out "
             "of reach for reasons that have nothing to do with the pass"),
            ("compiles", "int", "1 if the kernel still compiles with the pass applied"),
            ("compile_regression", "int", "1 if the baseline compiled and the optimized build did not. "
             "This is the pass's doing and must be 0"),
            ("seeds", "int", "random input draws the two builds were byte-compared on; the compare "
             "stops at the first difference, so a changed row shows the draw that found it"),
            ("bit_changed", "int", "1 if any of those draws differed from the baseline build. Must be 0"),
            ("bit_identical_perf", "int", "1 if the two builds also agreed byte for byte on the larger, "
             "plain-data performance input -- an independent second bit check"),
            ("order_sensitive", "int", "1 if this configuration's bytes DO move when the ordering "
             "constraint is dropped. Where this is 0 the reduction is order-invariant here and "
             "`bit_changed = 0` proves nothing about the pass"),
            ("nan_frac", "float", "fraction of the baseline output that is NaN or Inf. A saturated "
             "output compares equal to itself, so a high value makes `bit_changed` vacuous"),
            ("baseline_ms", "float", "device time without the pass, min of `do_bench` medians"),
            ("optimized_ms", "float", "device time with the pass"),
            ("ceiling_ms", "float", "device time of the unordered arm, same configuration"),
            ("speedup", "float", "baseline_ms / optimized_ms; above 1 means the pass helped"),
            ("gap_closed", "float", "(baseline_ms - optimized_ms) / (baseline_ms - ceiling_ms). Blank "
             "when the ordered and unordered arms are within 2% of each other: the denominator is "
             "then noise and the ratio is meaningless, not large"),
            ("verdict", "str", "bit-safe and faster, bit-safe and neutral, BITS CHANGED, compile "
             "regression, or why the row produced no result"),
            ("prior_source", "str", "which measurement the prior_* columns come from, with its "
             "hardware, Triton version and dtype"),
            ("prior_speedup", "float", "the prior measurement's speedup for this kernel and num_warps"),
            ("prior_gap_closed", "float", "the prior measurement's gap_closed for the same"),
            ("prior_dtype", "str", "the dtype the prior measured this kernel at"),
            ("like_for_like", "int", "1 if this row and the prior used the same dtype; 0 means a "
             "difference in the columns is dtype as well as hardware and Triton version"),
            ("error", "str", "non-empty if the row failed to produce a result"),
        ],
    },
}

GAP_FLOOR = 0.02  # the ordered and unordered arms must differ by at least this fraction of base


# ------------------------------------------------------------------------------------------- #
# The pass: find it, and inject it.
# ------------------------------------------------------------------------------------------- #
def resolve_pass():
    """The `add_*` binding for the pass under test, or None if this build has no such pass.

    `pass_available` has to be a column and a hard gate. A build without the binding produces a
    completely clean table -- every arm identical, 0 bits changed, 1.00x everywhere -- that
    measured nothing at all, and there is no way to tell it apart from a real result afterwards.
    """
    from triton._C.libtriton import passes

    sub = getattr(passes, "ttgpuir", None)
    return getattr(sub, PASS_BINDING, None) if sub is not None else None


@contextlib.contextmanager
def pass_injected(add_fn, args=PASS_ARGS):
    """Append the pass at end-of-TTGIR for the duration of the block, by patching the NVIDIA
    backend's `make_ttgir`.

    Injection rather than the in-pipeline knob (`TRITON_SET_RED_ORDERING_LAYOUTS=1`) for one
    reason: it is what the measurement being reproduced used, so the two sides compare. The kernel
    source, the launch path and every other pass are untouched -- the only difference between the
    two arms is this one pass manager run.
    """
    from triton._C.libtriton import ir
    from triton.backends.nvidia.compiler import CUDABackend

    original = CUDABackend.make_ttgir  # staticmethod access -> the underlying function

    def patched(mod, metadata, opt, capability):
        mod = original(mod, metadata, opt, capability)
        pm = ir.pass_manager(mod.context)
        add_fn(pm, *args)
        try:
            pm.run(mod, "inner_tree.layout")
        except TypeError:
            pm.run(mod)
        return mod

    CUDABackend.make_ttgir = staticmethod(patched)
    try:
        yield
    finally:
        CUDABackend.make_ttgir = staticmethod(original)


@contextlib.contextmanager
def pass_absent():
    """The baseline: the standard pipeline, untouched."""
    yield


def guard_is_fp_fusion_gated():
    """Does the pass in THIS tree gate its mul-fed-reduction skip on `enable_fp_fusion`?

    PR #2312's description mentions a follow-up that extends the guard to optimize mul-fed
    reductions (`sum(a*b)`) when fusion is off, reading a `ttg.enable_fp_fusion` module attribute.
    Whether that follow-up is present decides what a `col_dot` row at 1.00x means, so it is
    measured rather than assumed. Returns 1, 0, or None if the source is not there to read.

    Comments are stripped first: the unconditional version of the guard explains itself in a
    comment that contains the words "enable_fp_fusion", so a plain text search finds the phrase in
    a tree that does not act on it.
    """
    try:
        with open(PASS_SOURCE) as f:
            src = f.read()
    except OSError:
        return None
    code = "\n".join(line.split("//", 1)[0] for line in src.splitlines())
    return int("enable_fp_fusion" in code)


# ------------------------------------------------------------------------------------------- #
# Small helpers
# ------------------------------------------------------------------------------------------- #
def config_space():
    return [Config(w, s, f) for w in NUM_WARPS for s in NUM_STAGES for f in FP_FUSION]


def config_label(c):
    return (f"num_warps={c.num_warps} num_stages={c.num_stages} "
            f"enable_fp_fusion={'on' if c.enable_fp_fusion else 'off'}")


def gap_closed(base, opt, ceil):
    """`gap_closed`, or None when the question does not arise.

    The denominator is `base - ceil`, the whole distance the ordering constraint costs on this
    configuration. When the two arms land within a couple of percent of each other that distance
    is noise, and dividing by it produces numbers like 238 or -769 that then poison any average
    they are in. Those rows are not "the pass closed 23800% of the gap", they are "there was no
    gap"; they are reported blank.
    """
    if base is None or opt is None or ceil is None or not base:
        return None
    denom = base - ceil
    if abs(denom) < GAP_FLOOR * abs(base):
        return None
    return (base - opt) / denom


def _fmt(v, spec="{:.2f}"):
    return "-" if v is None else spec.format(v)


def _x(v):
    return "-" if v is None else f"{v:.2f}x"


def gpu_contention(seconds=20.0, samples=8):
    """Is the pinned GPU ours alone? Returns (max_util, [other pids], note).

    Every number in the timing arm is a device time and this box is shared: another process on the
    same device does not change a bit result but makes a speedup meaningless. So the step looks
    before it measures, and looks again as it goes, rather than trusting that the device it was
    handed at the start is still quiet an hour later.
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


def wait_for_quiet(max_wait_s, poll_s=30.0):
    """Pause until the pinned GPU is ours again, or give up. Returns (seconds waited, pids left).

    A neighbour that arrives mid-run must not be timed next to -- but it also must not cost the
    whole run. A five-minute profiling job on a shared box would otherwise end a ninety-minute
    sweep, and the resumed run would repeat every row it had already paid for. So the sweep
    pauses: nothing is measured and nothing is written while another process is on the device,
    and it picks up where it stopped once the device is quiet again. Only if the neighbour is
    still there after `max_wait_s` does the run stop for good.
    """
    t0 = time.time()
    while True:
        _u, others, note = gpu_contention(seconds=2.0, samples=1)
        if not others and not note:
            return time.time() - t0, []
        if time.time() - t0 >= max_wait_s:
            return time.time() - t0, others or ["unknown"]
        time.sleep(poll_s)


# ------------------------------------------------------------------------------------------- #
# One row
# ------------------------------------------------------------------------------------------- #
class Harness:
    """Everything one measurement needs, gathered once: the kernel catalogue, the pass, the
    cache-clearing list and the two pipeline variants."""

    def __init__(self, kernels, add_fn, seeds, bench_reps):
        self.kernels = kernels
        self.add_fn = add_fn
        self.seeds = seeds
        self.bench_reps = bench_reps
        from . import _inner_tree_kernels as K
        self.K = K
        self.jits = K.jit_functions(kernels)

    def clear(self):
        self.K.clear_caches(self.jits)

    def base_variant(self):
        return pass_absent()

    def opt_variant(self, args=PASS_ARGS):
        return pass_injected(self.add_fn, args)

    def build(self, kernel, dtype, ordering, config, variant):
        """One compile, with the caches cleared and the pipeline variant installed. Both are
        required: the variant makes the arm different, the clear makes the difference visible."""
        self.clear()
        with variant:
            return self.K.compile_kernel(kernel, dtype, ordering, config)


def measure(h, kernel, dtype, config, do_perf):
    """Every answer for one configuration: three compiles, the bits, and the three times.

    Three compiles and not nine. The arms are compiled once each and then reused for the byte
    comparison and for the timing, so the row costs what it has to and no more. The compiled
    objects survive the cache clears that follow -- clearing the cache drops the lookup table, not
    the kernel already in hand.
    """
    K = h.K
    row = dict(kernel=kernel.name, suite=kernel.suite, kernel_source=kernel.source, dtype=dtype, ordering="inner_tree",
               config=config_label(config), num_warps=config.num_warps, num_stages=config.num_stages,
               enable_fp_fusion=int(config.enable_fp_fusion), pass_name=PASS_SPEC, seeds=0, bit_changed=0,
               base_compiles=0, compiles=0, compile_regression=0, error="")

    try:
        b_base = h.build(kernel, dtype, "inner_tree", config, h.base_variant())
        row["base_compiles"] = 1
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"baseline compile: {type(exc).__name__}: {exc}"[:220]
        row["verdict"] = "baseline does not compile (not the pass)"
        return row

    b_opt = None
    try:
        b_opt = h.build(kernel, dtype, "inner_tree", config, h.opt_variant())
        row["compiles"] = 1
        row["pass_changed_ir"] = int(b_base.ttgir != b_opt.ttgir)
    except Exception as exc:  # noqa: BLE001
        row["compile_regression"] = 1
        row["error"] = " ".join(f"optimized compile: {type(exc).__name__}: {exc}".split())[:220]

    b_ceil = None
    try:
        b_ceil = h.build(kernel, dtype, "unordered", config, h.base_variant())
    except Exception as exc:  # noqa: BLE001
        row["error"] = row["error"] or " ".join(f"ceiling compile: {type(exc).__name__}: {exc}".split())[:220]

    # Is this configuration order-sensitive at all? Under `unordered` the compiler is free to pick
    # a different reduction order; if the bytes come back the same anyway then the reduction is
    # order-invariant here and a `bit_changed = 0` says nothing about the pass. Taken before the
    # A/B and independently of whether the optimized arm built, because it is a property of the
    # kernel and the configuration, not of the pass -- a compile regression must not silently
    # report itself as order-insensitive.
    base0 = K.run_bytes(b_base, 0)
    row["nan_frac"] = round(K.nan_fraction(base0), 4)
    if b_ceil is not None:
        row["order_sensitive"] = int(K.run_bytes(b_ceil, 0) != base0)

    if b_opt is None:
        row["verdict"] = "compile regression"
        return row

    # The bits. Same draw into both arms, one draw at a time, and stop at the first difference --
    # a single differing draw is the whole answer.
    for seed in range(h.seeds):
        a = base0 if seed == 0 else K.run_bytes(b_base, seed)
        b = K.run_bytes(b_opt, seed)
        row["seeds"] = seed + 1
        if a != b:
            row["bit_changed"] = 1
            break

    if do_perf:
        time_into(h, row, b_base, b_opt, b_ceil)
    return set_verdict(row, timed=do_perf)


def time_into(h, row, b_base, b_opt, b_ceil):
    """The timing arm, written into an existing row.

    Split out so the bits and the times can be taken in separate runs: on a shared box the bit
    work can go whenever, the timing has to wait for a device of its own.
    """
    K = h.K
    try:
        base_ms, base_bytes = K.bench_ms(b_base, reps=h.bench_reps)
        opt_ms, opt_bytes = K.bench_ms(b_opt, reps=h.bench_reps)
    except Exception as exc:  # noqa: BLE001
        row["error"] = row.get("error") or f"timing: {type(exc).__name__}: {exc}"[:220]
        return row
    row["baseline_ms"] = round(base_ms, 6)
    row["optimized_ms"] = round(opt_ms, 6)
    row["bit_identical_perf"] = int(base_bytes == opt_bytes)
    if opt_ms > 0:
        row["speedup"] = round(base_ms / opt_ms, 4)
    if b_ceil is not None:
        try:
            ceil_ms, _ = K.bench_ms(b_ceil, reps=h.bench_reps)
        except Exception:  # noqa: BLE001
            return row
        row["ceiling_ms"] = round(ceil_ms, 6)
        g = gap_closed(base_ms, opt_ms, ceil_ms)
        row["gap_closed"] = None if g is None else round(g, 4)
    return row


def set_verdict(row, timed):
    if not row.get("base_compiles"):
        row["verdict"] = "baseline does not compile (not the pass)"
    elif row.get("compile_regression"):
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
# Self-checks. Without these the reproduction is fake rather than merely different.
# ------------------------------------------------------------------------------------------- #
def self_checks(h, add_fn):
    """Four things that, if wrong, make every number below meaningless.

    1. The pass has a binding in this build. Without one every arm is the baseline and the whole
       table is a clean 1.00x that measured nothing.
    2. The injection survives both compile caches. Two builds of a configuration the pass must
       transform have to produce different TTGIR; if they do not, the optimized build was served
       the baseline one and every `bit_changed = 0` below is an artefact of a cache.
    3. With the pass injected at `strategy=off` -- the pass's own documented no-op -- the TTGIR
       must come back byte-identical to the baseline. This separates "the pass changed the IR"
       from "running an extra pass manager changed the IR", which would make check 2 vacuous.
    4. Baseline against baseline, on the GPU, must be byte-identical. If a kernel is not
       reproducible against itself, nothing measured afterwards means anything.
    """
    lines, ok = [], True
    lines.append(f"  pass binding                  {'FOUND' if add_fn else 'MISSING'}  ({PASS_SPEC})")
    if not add_fn:
        lines.append("  ABORT: no binding for the pass in this build; every row would be baseline against")
        lines.append("         baseline, which reports a perfect result and measures nothing.")
        return lines, False

    always = os.environ.get("TRITON_ALWAYS_COMPILE")
    lines.append(f"  TRITON_ALWAYS_COMPILE         {always or 'NOT SET'}")

    # A probe has to be a configuration the pass certainly transforms, so try the strided-axis
    # kernels first and fall through the rest. A candidate that fails to build, or that the pass
    # declines, is a bad probe rather than a failed check -- the one row on this tree that fails
    # to build with the pass is a real result reported later, and it must not abort the run.
    preferred = ("sum_2d_col", "sum_2d_axis0", "sum_3d_outer", "I_bias_grad_dim0", "J_epilogue_colsum_dim0")
    ordered = ([k for n in preferred
                for k in h.kernels if k.name == n] + [k for k in h.kernels if k.name not in preferred])
    probe = None
    tried = []
    for kernel in ordered:
        for dtype in kernel.dtypes:
            for cfg in (Config(4, 2, True), Config(8, 2, True), Config(2, 2, True)):
                try:
                    b_base = h.build(kernel, dtype, "inner_tree", cfg, h.base_variant())
                    b_opt = h.build(kernel, dtype, "inner_tree", cfg, h.opt_variant())
                except Exception as exc:  # noqa: BLE001
                    tried.append(f"{kernel.name}/{dtype}/nw{cfg.num_warps}: {type(exc).__name__}")
                    continue
                if b_base.ttgir != b_opt.ttgir:
                    probe = (kernel, dtype, cfg, b_base)
                    break
                tried.append(f"{kernel.name}/{dtype}/nw{cfg.num_warps}: pass declined")
            if probe:
                break
        if probe:
            break

    if probe is None:
        lines.append("  injection changes the IR      NO   (no probe anywhere in the kernel list)")
        lines.append("  ABORT: no configuration produced different TTGIR with the pass injected. Either the")
        lines.append("         pass fires on nothing here, or a compile cache is serving the baseline build")
        lines.append("         to both arms. Triton's caches are keyed on the specialization and the launch")
        lines.append("         options only, so an injected pass is part of neither key; relaunch with")
        lines.append("         TRITON_ALWAYS_COMPILE=1 set before the process starts. Candidates tried:")
        for t in tried[:8]:
            lines.append(f"           {t}")
        return lines, False

    kernel, dtype, cfg, b_base = probe
    where = f"({kernel.name} {dtype} num_warps={cfg.num_warps})"
    lines.append(f"  injection changes the IR      YES   {where}")

    b_off = h.build(kernel, dtype, "inner_tree", cfg, h.opt_variant(("off", ) + PASS_ARGS[1:]))
    off_same = b_off.ttgir == b_base.ttgir
    lines.append(f"  strategy=off equals baseline  {'YES' if off_same else 'NO'}"
                 "   (so the check above is the pass, not the extra pass manager)")
    if not off_same:
        lines.append("  ABORT: injecting the pass at its own documented no-op setting changed the IR, so the")
        lines.append("         difference measured above is not attributable to the transformation.")
        return lines, False

    same = h.K.run_bytes(b_base, 0) == h.K.run_bytes(b_base, 0)
    lines.append(f"  baseline reproduces itself    {'YES' if same else 'NO'}   {where}")
    if not same:
        ok = False
        lines.append("  ABORT: the baseline is not byte-reproducible against itself; the harness is broken.")
    return lines, ok


def determinism_sweep(h, plan):
    """Run every kernel's baseline twice on one configuration and require the same bytes.

    Cheap, and it catches the failure mode a byte comparison is most exposed to: an output buffer
    the kernel does not fully write, whose leftover contents differ between two launches. That
    would read as BITS CHANGED on a kernel that is perfectly bit-safe.
    """
    lines, bad = [], []
    for kernel, dtype, _configs in plan:
        same, err = None, ""
        for cfg in (Config(4, 2, True), Config(8, 2, True), Config(2, 2, True)):
            try:
                b = h.build(kernel, dtype, "inner_tree", cfg, h.base_variant())
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                continue
            same = h.K.run_bytes(b, 0) == h.K.run_bytes(b, 0)
            break
        if same is None:
            lines.append(f"      {kernel.name:24} {dtype:5} NO BASELINE BUILD AT ALL  {err}"[:118])
            bad.append(kernel.name)
        elif not same:
            lines.append(f"      {kernel.name:24} {dtype:5} NOT REPRODUCIBLE")
            bad.append(kernel.name)
    head = (f"  every kernel reproduces itself  {'YES' if not bad else 'NO'}"
            f"   ({len(plan) - len(bad)}/{len(plan)} (kernel, dtype) pairs)")
    return [head] + lines, not bad


# ------------------------------------------------------------------------------------------- #
# Report
# ------------------------------------------------------------------------------------------- #
def summarise(rows, kernel, dtype, num_warps):
    sel = [r for r in rows if r["kernel"] == kernel and r["dtype"] == dtype and r.get("num_warps") == num_warps]
    if not sel:
        return None
    from . import _inner_tree_kernels as K
    sp = [r["speedup"] for r in sel if r.get("speedup")]
    # Recomputed from the three stored times rather than read back, so the noise floor applies to
    # rows taken before it existed too.
    gp = [
        g for g in (gap_closed(r.get("baseline_ms"), r.get("optimized_ms"), r.get("ceiling_ms")) for r in sel)
        if g is not None
    ]
    # The sensitivity denominator counts only the rows where the question was actually asked: a
    # row whose ceiling arm never built has no answer, and reporting it as 0 would read as
    # "order-invariant", which is the opposite of "not measured".
    sens = [r for r in sel if r.get("order_sensitive") is not None]
    return dict(n=len(sel), bit_changed=sum(r.get("bit_changed", 0)
                                            for r in sel), sensitive=sum(r.get("order_sensitive", 0) for r in sens),
                n_sens=len(sens), fired=sum(r.get("pass_changed_ir", 0) for r in sel), speedup=K.median(sp),
                best=max(sp) if sp else None, gap=K.median(gp), nan=max((r.get("nan_frac") or 0.0) for r in sel),
                nocompile=sum(r.get("compile_regression", 0) for r in sel), nobase=sum(1 for r in sel
                                                                                       if not r.get("base_compiles")))


def report(rows, kernels, header, probe):
    out = list(header)

    # ---- the side by side ----------------------------------------------------------------- #
    out += [
        "", "SIDE BY SIDE -- this machine against PR #2312 (commit 8176cccdc)", "=" * 118, "",
        "Mine is the median over the 12 configurations at that num_warps (6 num_stages x 2 enable_fp_fusion).",
        "`sens` is how many of those 12 are order-sensitive: where it is 0/12 the reduction does not depend on",
        "the order here at all, so `bits` = 0 on those rows is vacuous, not a pass. `fired` is how many of the",
        "12 the pass actually rewrote the IR on; a kernel at 1.00x with fired = 0 was declined by a guard, and",
        "that is a result, not a gap. `L` marks a row measured at the same dtype the prior used -- the only",
        "rows where a difference is hardware and Triton alone.", ""
    ]
    head = (f"{'kernel':24} {'dtype':5} {'nw':>3} {'cfg':>4} {'bits':>5} {'sens':>6} {'fired':>6} {'nan':>5} "
            f"{'speedup':>8} {'max':>7} {'gap':>6} | {'prior':>7} {'gap':>6} L  prior arm")
    out += [head, "-" * len(head)]

    for kernel in kernels:
        for dtype in kernel.dtypes:
            prior = PRIOR_H100.get(kernel.name, {})
            like = "L" if prior and dtype == kernel.prior_dtype else " "
            for nw in (4, 8):
                pv = prior.get(nw) or (prior.get(None) if None in prior else None)
                if prior:
                    arm = f"{PRIOR_LABEL} {kernel.prior_dtype}" + ("" if nw in prior else " (nw not stated)")
                else:
                    arm = "not in the prior table"
                s = summarise(rows, kernel.name, dtype, nw)
                tail = ""
                if s is None:
                    tail = "  NOT MEASURED"
                elif s["nocompile"]:
                    tail = f"  {s['nocompile']}/{s['n']} FAIL TO COMPILE WITH THE PASS"
                elif s["nobase"]:
                    tail = f"  {s['nobase']}/{s['n']} baseline does not compile"
                if s is None:
                    out.append(f"{kernel.name:24} {dtype:5} {nw:>3} {'-':>4} {'-':>5} {'-':>6} {'-':>6} {'-':>5} "
                               f"{'-':>8} {'-':>7} {'-':>6} | {_x(pv and pv[0]):>7} {_fmt(pv and pv[1]):>6} {like}  "
                               f"{arm}{tail}")
                else:
                    out.append(f"{kernel.name:24} {dtype:5} {nw:>3} {s['n']:>4} {s['bit_changed']:>5} "
                               f"{str(s['sensitive']) + '/' + str(s['n_sens']):>6} "
                               f"{str(s['fired']) + '/' + str(s['n']):>6} {s['nan']:>5.2f} {_x(s['speedup']):>8} "
                               f"{_x(s['best']):>7} {_fmt(s['gap']):>6} | {_x(pv and pv[0]):>7} "
                               f"{_fmt(pv and pv[1]):>6} {like}  {arm}{tail}")

    like_names = [k.name for k in kernels if PRIOR_H100.get(k.name) and k.prior_dtype in k.dtypes]
    no_prior = [k.name for k in kernels if not PRIOR_H100.get(k.name)]
    out += [
        "",
        "Every kernel PR #2312 measured is in the table above; there is no `NOT REACHABLE` row any more.",
        f"  No prior number ({len(no_prior)} kernels, that table did not carry them): {', '.join(no_prior) or 'none'}.",
        "",
        "The dtype differs between the two sides on every row without an `L`. The earlier run measured f32:",
        "  the dtype axis did not exist in eval_kernels.py at that commit, so every synthetic kernel took the",
        "  f32 default. This run measures the synthetic kernels at f16 and fp8, which is what this project",
        "  reports. A difference on an unmarked row is hardware AND Triton AND dtype, not hardware alone.",
        f"  Like-for-like ({len(like_names)} kernels, prior number and same dtype): {', '.join(like_names) or 'none'}.",
    ]

    # ---- the full num_warps sweep --------------------------------------------------------- #
    warps = sorted({r.get("num_warps") for r in rows if r.get("num_warps") is not None})
    for title, field, fmt in (("median speedup", "speedup", _x), ("configurations the pass fired on", "fired", None),
                              ("order-sensitive configurations", "sensitive", None)):
        out += ["", "", f"ALL num_warps -- {title}", "=" * 118, ""]
        head2 = f"{'kernel':24} {'dtype':5} " + " ".join(f"{('nw=' + str(w)):>9}" for w in warps)
        out += [head2, "-" * len(head2)]
        for kernel in kernels:
            for dtype in kernel.dtypes:
                cells, any_cell = [], False
                for w in warps:
                    s = summarise(rows, kernel.name, dtype, w)
                    if s is None:
                        cells.append(f"{'-':>9}")
                        continue
                    any_cell = True
                    if fmt is not None:
                        cells.append(f"{fmt(s[field]):>9}")
                    else:
                        denom = s["n_sens"] if field == "sensitive" else s["n"]
                        cells.append(f"{str(s[field]) + '/' + str(denom):>9}")
                if any_cell:
                    out.append(f"{kernel.name:24} {dtype:5} " + " ".join(cells))

    # ---- the second prior ----------------------------------------------------------------- #
    out += [
        "", "", "PR #2312'S OWN GB300 / sm_100 TABLE (72 f32 configurations per kernel), for reference", "=" * 118, "",
        "Same hardware family as this machine but f32 and a different Triton, and it covers only seven of",
        "the kernels, so it is a second prior rather than the headline comparison.", ""
    ]
    head3 = f"{'kernel':24} {'prior median':>13} {'prior max':>10} {'prior gap':>10}"
    out += [head3, "-" * len(head3)]
    for kernel, (med, mx, gap) in PRIOR_SM100.items():
        out.append(f"{kernel:24} {med:>12.2f}x {mx:>9.2f}x {gap:>10.2f}")

    # ---- the gate ------------------------------------------------------------------------- #
    total = len(rows)
    changed = sum(r.get("bit_changed", 0) for r in rows)
    perf_changed = sum(1 for r in rows if r.get("bit_identical_perf") == 0)
    regress = sum(r.get("compile_regression", 0) for r in rows)
    nobase = sum(1 for r in rows if not r.get("base_compiles"))
    errors = sum(1 for r in rows if r.get("error"))
    insens = [r for r in rows if r.get("order_sensitive") == 0]
    pct = 100.0 * len(insens) / total if total else 0.0
    out += [
        "", "", "GATE", "=" * 118, "", f"  rows measured                 {total}",
        f"  bit-changed configurations    {changed}   (must be 0)",
        f"  bit-changed on the perf input {perf_changed}   (must be 0; an independent second check)",
        f"  compile regressions           {regress}   (must be 0: baseline built, optimized did not)",
        f"  baseline did not compile      {nobase}   (not the pass's doing; excluded from the gate)",
        f"  rows carrying an error        {errors}   (the compile regressions above carry theirs)",
        f"  order-INSENSITIVE rows        {len(insens)}   ({pct:.1f}% of rows -- on these the reduction does",
        "                                not depend on the order, so the bit check is vacuous, not passing)"
    ]
    by_kd = {}
    for r in insens:
        by_kd[(r["kernel"], r["dtype"])] = by_kd.get((r["kernel"], r["dtype"]), 0) + 1
    for (k, d), n in sorted(by_kd.items()):
        out.append(f"      {k:24} {d:5} {n} rows")
    if insens and probe:
        fp8 = probe.get("fp8")
        f16 = probe.get("f16")
        out += [
            "",
            "  Why the fp8 pure sums are among those. An fp8 e4m3 value carries a four-bit significand, so",
            "  the exact sum of a few hundred of them still fits inside an f32 mantissa, and then every order",
            "  gives the same answer -- the reduction is order-invariant by arithmetic, not by luck. Measured",
            "  off-device on this run, on the same draw the kernels use:",
        ]
        if fp8:
            out.append(f"      fp8   {fp8[0]}/{fp8[2]} draws sum exactly in f32; sequential vs pairwise tree "
                       f"differed on {fp8[1]}/{fp8[2]}")
        if f16:
            out.append(f"      f16   {f16[0]}/{f16[2]} draws sum exactly in f32; sequential vs pairwise tree "
                       f"differed on {f16[1]}/{f16[2]}")
        out += [
            "  So on those kernels an fp8 `bit_changed = 0` is not weak evidence, it is no evidence: nothing",
            "  could have moved. `col_exp_sum` and `col_dot` keep their fp8 sensitivity because exp() and a*b",
            "  widen the leaves to f32 before the reduce, and those are the fp8 rows worth reading.",
        ]
    other_insens = sorted({(k, d) for (k, d) in by_kd if d != "fp8"})
    if other_insens:
        out += ["", "  Order-insensitive and NOT fp8, so the argument above does not cover them:"]
        for k, d in other_insens:
            out.append(f"      {k:24} {d:5} {INSENSITIVE_NOTE.get(k, 'no note recorded')}")
        out += [
            "  What is measured on these is only that the two orderings produced the same bytes on every",
            "  configuration; the reason offered is an explanation, not a second measurement. Either way the",
            "  bit check has nothing to catch there and the gate does not count them as a pass.",
        ]
    sat = {}
    for r in rows:
        f = r.get("nan_frac") or 0.0
        if f > 0.01:
            key = (r["kernel"], r["dtype"])
            sat[key] = max(sat.get(key, 0.0), f)
    if sat:
        out += [
            "", "  Partly saturated outputs. A NaN or an Inf compares equal to itself, so on the saturated",
            "  part of these outputs the bit check cannot see a difference. The rest of the output still",
            "  can, and these rows are order-sensitive, so the check is weakened rather than vacuous:"
        ]
        for (k, d), f in sorted(sat.items()):
            out.append(f"      {k:24} {d:5} up to {100 * f:.1f}% of the output is NaN or Inf")

    if regress:
        by_r = {}
        for r in rows:
            if r.get("compile_regression"):
                key = (r["kernel"], r["dtype"], r.get("num_warps"))
                by_r[key] = by_r.get(key, 0) + 1
        out += [
            "", "  The compile regressions, in full. The baseline builds and the optimized one does not, so",
            "  these are the pass's doing, not the kernel's:"
        ]
        for (k, d, nw), n in sorted(by_r.items(), key=lambda kv: str(kv[0])):
            first = next(
                (r.get("error", "")
                 for r in rows
                 if r.get("compile_regression") and (r["kernel"], r["dtype"], r.get("num_warps")) == (k, d, nw)), "")
            out.append(f"      {k:24} {d:5} num_warps={nw:<3} {n} configurations")
            if first:
                out.append(f"          {' '.join(first.split())[:104]}")
        out += [
            "  These sit on the register-pressure guard's boundary: the reduce-friendly layout carries",
            "  tileElems / (warpSize * num_warps) values per thread, and the guard declines only ABOVE",
            "  `max-elems-per-thread`, so a configuration landing exactly on the limit is still rewritten."
        ]
    declined = [(k, d)
                for k in {r["kernel"]
                          for r in rows}
                for d in {r["dtype"]
                          for r in rows
                          if r["kernel"] == k}
                if not any(r.get("pass_changed_ir") for r in rows if r["kernel"] == k and r["dtype"] == d)]
    if declined:
        out += [
            "", "  Kernels the pass declined on every configuration. This is the pass working, not failing:",
            "  it is only allowed to touch an inner_tree reduce whose axis is under-parallelized across the",
            "  warp, and it must leave a mul-fed reduce alone whatever the shape."
        ]
        fallback = ("one of the two shape guards: the axis is already spread across the warp, or the "
                    "tile carries more than max-elems-per-thread values per thread at this num_warps")
        for k, d in sorted(declined):
            out.append(f"      {k:24} {d:5} {DECLINE_NOTE.get(k, fallback)}")
    passed = not changed and not perf_changed and not regress
    out += [
        "", "  RESULT: " + ("PASS -- the pass preserved the bits on every configuration where the bits move at "
                            "all." if passed else "FAIL -- see the counts above."), ""
    ]
    return out


# ------------------------------------------------------------------------------------------- #
# Entry point
# ------------------------------------------------------------------------------------------- #
def run(args, env):
    from . import _inner_tree_kernels as K

    if os.environ.get("INNER_TREE_LAYOUT_REPORT_ONLY", "").strip() == "1":
        _report_only(K)
        return

    preflight = os.environ.get("INNER_TREE_LAYOUT_PREFLIGHT", "").strip() == "1"
    want_kernels = [k.strip() for k in os.environ.get("INNER_TREE_LAYOUT_KERNELS", "").split(",") if k.strip()]
    want_dtypes = [d.strip() for d in os.environ.get("INNER_TREE_LAYOUT_DTYPES", "").split(",") if d.strip()]
    seeds = int(os.environ.get("INNER_TREE_LAYOUT_SEEDS", "10"))
    bench_reps = int(os.environ.get("INNER_TREE_LAYOUT_BENCH_REPS", "3"))
    force_timing = os.environ.get("INNER_TREE_LAYOUT_FORCE_TIMING", "").strip() == "1"
    patience_s = float(os.environ.get("INNER_TREE_LAYOUT_PATIENCE_MIN", "45")) * 60.0

    kernels = K.catalogue()
    if want_kernels:
        kernels = [k for k in kernels if k.name in want_kernels]
    if want_dtypes:
        kernels = [k for k in kernels if any(d in want_dtypes for d in k.dtypes)]

    # Look at the device BEFORE this process touches it, so the reading is the neighbours' and not
    # our own. A busy device is not a reason to skip the bit work, only the timing.
    util, others, gpu_note = (None, [], "skipped (preflight)") if preflight else gpu_contention()
    blocked = (bool(others) or (util is not None and util > 5) or bool(gpu_note)) and not preflight
    if force_timing:
        blocked = False

    add_fn = resolve_pass()
    gated = guard_is_fp_fusion_gated()
    h = Harness(kernels, add_fn, seeds, bench_reps)

    header = [
        "inner_tree.layout -- reproducing PR #2312 (commit 8176cccdc) on this machine",
        "=" * 118,
        "",
        f"  when                  {env['when']}",
        f"  device                {env['device']}  ({env['arch']})",
        f"  triton                {env['triton']}   torch {env['torch']}   commit {env['commit']}",
        f"  pass under test       {PASS_SPEC}",
        f"  pass_available        {1 if add_fn else 0}",
        f"  mul-fed guard         {'gated on enable_fp_fusion' if gated else 'skips every mul-fed reduce'}"
        f"   (guard_fp_fusion_gated={gated})",
        f"  seeds per row         {seeds}   bench reps {bench_reps} (min of that many do_bench medians)",
        f"  TRITON_ALWAYS_COMPILE {os.environ.get('TRITON_ALWAYS_COMPILE') or 'NOT SET'}",
        f"  gpu                   CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}"
        f"  peak util before start {util if util is not None else '?'}%"
        f"  other processes on it: {others or 'none'}" + (f"  [{gpu_note}]" if gpu_note else ""),
        "",
        "  arms   base = inner_tree, standard pipeline",
        "         opt  = base + " + PASS_SPEC + " appended at end-of-TTGIR",
        "         ceil = the SAME configuration with reduction_ordering=unordered, standard pipeline",
        "         speedup = base/opt        gap_closed = (base-opt)/(base-ceil)",
        "",
        "  The ceiling flips one flag and holds everything else fixed, because the experiment has exactly",
        "  two variables: is inner_tree on, and is the pass on. A best-over-configurations ceiling would",
        "  answer a different question -- what an unconstrained autotuner finds -- and that one belongs to",
        "  the gemm.perf steps.",
        "",
        "  sizes  synthetic kernels    64 programs for the bit check, 2048 for the timing (their own sizes)",
        "         inductor and zoo     the grid the generated shape implies, the same for both arms",
        "",
    ]
    if gated == 0:
        header += [
            "  NOTE: this tree's pass skips EVERY mul-fed reduction (`sum(a*b)`), with no enable_fp_fusion",
            "        condition, and no `ttg.enable_fp_fusion` module attribute exists anywhere in it. The",
            "        follow-up PR #2312's description mentions is not in this branch, and the prior's col_dot",
            "        row is labelled `fusion off`, which only means something with that follow-up present. So",
            "        col_dot coming back at ~1.00x here is a difference in the code, not in the hardware.",
            "",
        ]
    if blocked:
        header += [
            "  BLOCKED ON THE GPU: the pinned device is not idle, so the timing arm is not run. A device time",
            "        taken next to somebody else's kernel is not a measurement, and reporting it would be worse",
            f"        than reporting nothing. Reason: {gpu_note or ('other processes ' + str(others))}"
            f"{'' if gpu_note else f', peak util {util}%'}.",
            "        The compiles and the bit comparisons do run: a busy neighbour does not change them.",
            "",
        ]
    print("\n".join(header))

    checks, ok = self_checks(h, add_fn)
    print("SELF-CHECKS")
    print("\n".join(checks), flush=True)
    header += ["SELF-CHECKS"] + checks
    if not ok:
        with open(REPORT, "w") as f:
            f.write("\n".join(header) + "\nABORTED: a self-check failed; nothing was measured.\n")
        return

    # ---- the plan, printed before anything long runs so the size is on the record ----------- #
    plan = []
    for kernel in kernels:
        for dtype in kernel.dtypes:
            if want_dtypes and dtype not in want_dtypes:
                continue
            plan.append((kernel, dtype, config_space()))
    n_total = sum(len(c) for _, _, c in plan)

    det, det_ok = determinism_sweep(h, plan)
    print("\n".join(det) + "\n", flush=True)
    header += det + [""]
    if not det_ok:
        with open(REPORT, "w") as f:
            f.write("\n".join(header) + "\nABORTED: a kernel is not byte-reproducible against itself.\n")
        return

    catalogue_lines = ["THE KERNELS", "=" * 118, ""]
    chead = f"{'kernel':24} {'suite':10} {'dtypes':10} {'cfg':>4}  what it is"
    catalogue_lines += [chead, "-" * len(chead)]
    for kernel, dtype, configs in plan:
        catalogue_lines.append(f"{kernel.name:24} {kernel.suite:10} {dtype:10} {len(configs):>4}  {kernel.what}")
    catalogue_lines += ["", "Kernel bodies are imported, not retyped, so they can be diffed against their source:"]
    for kernel in kernels:
        catalogue_lines.append(f"    {kernel.name:24} {kernel.source}")
    catalogue_lines += ["", "NOT MEASURED, and why:"]
    for name, where, why in K.OUT_OF_SUITE:
        catalogue_lines.append(f"    {name:26} {where}")
        catalogue_lines.append(f"        {why}")
    catalogue_lines += ["", f"TOTAL {len(plan)} (kernel, dtype) pairs x 72 configurations = {n_total} rows", ""]
    print("\n".join(catalogue_lines), flush=True)
    header += catalogue_lines

    # ---- resume ----------------------------------------------------------------------------- #
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
        # Provenance, not chatter. A resumed run's table is assembled from more than one
        # invocation, and a reader is entitled to know that before comparing timings across rows.
        resumed = [
            f"  RESUMED: {len(done)} of the rows below were measured by an earlier invocation of the same",
            "        command and were not re-taken. Each was timed on a device this step had checked was",
            "        quiet, or it carries no timing at all.", ""
        ]
        print("\n".join(resumed), flush=True)
        header += resumed

    if preflight:
        t0 = time.time()
        for kernel, dtype, configs in plan:
            K.reset_launches()
            r = measure(h, kernel, dtype, Config(num_warps=4, num_stages=2, enable_fp_fusion=True), do_perf=False)
            print(
                f"  {kernel.name:24} {dtype:5} base_compiles={r.get('base_compiles')} "
                f"compiles={r.get('compiles')} bits={r.get('bit_changed')} fired={r.get('pass_changed_ir')} "
                f"order_sensitive={r.get('order_sensitive')} nan={r.get('nan_frac')} {r.get('error', '')}"[:150],
                flush=True)
        per = (time.time() - t0) / max(len(plan), 1)
        print(f"\nPREFLIGHT: {len(plan)} compile-and-bits rows in {time.time() - t0:.0f}s -> {per:.1f}s/row")
        print(f"           the same work over {n_total} rows projects to {per * n_total / 3600:.1f} hours;")
        print(f"           the timing arm adds three do_bench arms ({bench_reps} runs each) per row on top.")
        print("           Nothing was written. Drop INNER_TREE_LAYOUT_PREFLIGHT to run it for real.")
        return

    # ---- the sweep --------------------------------------------------------------------------- #
    out = writer(TABLE)
    deadline = time.time() + float(args.minutes) * 60.0
    stopped_early, joined, n_done = False, None, 0
    paused_s, pauses = 0.0, 0
    t0 = time.time()
    for kernel, dtype, configs in plan:
        K.reset_launches()
        prior = PRIOR_H100.get(kernel.name, {})
        print(f"\n{kernel.name} [{dtype}] -- {len(configs)} configurations", flush=True)
        for i, config in enumerate(configs):
            key = (kernel.name, dtype, config_label(config))
            have = done.get(key)
            # A row taken while the GPU was busy carries the bits but no timing. Redo exactly
            # those when a free device turns up, and leave every finished row alone.
            if have is not None and (blocked or have.get("speedup") is not None or not have.get("base_compiles")
                                     or have.get("compile_regression")):
                continue
            if time.time() > deadline:
                stopped_early = True
                break
            # Somebody else can start on this device an hour into the run; from that point on the
            # timings are not ours. Check every 24 rows, then pause rather than record next to
            # them, and give up only if they are still there after the patience window.
            if not blocked and not force_timing and n_done and n_done % 24 == 0:
                _u, _o, _n = gpu_contention(seconds=4.0, samples=2)
                if _o:
                    print(
                        f"    PAUSED after {n_done} rows: another process {_o} is on the pinned GPU. "
                        f"Waiting up to {patience_s / 60:.0f} min for it to leave; nothing is measured "
                        "while waiting.", flush=True)
                    waited, still = wait_for_quiet(patience_s)
                    paused_s += waited
                    pauses += 1
                    print(f"    {'RESUMED' if not still else 'GIVING UP'} after {waited / 60:.1f} min", flush=True)
                    if still:
                        joined = (f"another process ({still}) was on the pinned GPU after {n_done} rows and was "
                                  f"still there {patience_s / 60:.0f} minutes later")
                        stopped_early = True
                        break
            try:
                row = measure(h, kernel, dtype, config, do_perf=not blocked)
            except Exception as exc:  # noqa: BLE001
                row = dict(kernel=kernel.name, suite=kernel.suite, kernel_source=kernel.source, dtype=dtype,
                           ordering="inner_tree", config=config_label(config), num_warps=config.num_warps,
                           num_stages=config.num_stages, enable_fp_fusion=int(config.enable_fp_fusion),
                           pass_name=PASS_SPEC, verdict="error", error=f"{type(exc).__name__}: {exc}"[:220])
            row["pass_available"] = 1 if add_fn else 0
            row["guard_fp_fusion_gated"] = gated
            row["prior_source"] = f"PR #2312 {PRIOR_LABEL} dtype={kernel.prior_dtype}" if prior else "not in the prior"
            row["prior_dtype"] = kernel.prior_dtype if prior else ""
            row["like_for_like"] = int(bool(prior) and dtype == kernel.prior_dtype)
            pv = prior.get(config.num_warps) or (prior.get(None) if None in prior else None)
            if pv:
                row["prior_speedup"], row["prior_gap_closed"] = pv
            out.write(json.dumps(row) + "\n")
            out.flush()  # a killed run loses at most this one row
            done[key] = row
            n_done += 1
            if (i + 1) % 12 == 0 or i + 1 == len(configs):
                rate = (time.time() - t0 - paused_s) / max(n_done, 1)
                print(
                    f"    {i + 1:3}/{len(configs)}  {row['verdict']:34} "
                    f"speedup={_fmt(row.get('speedup'))}x gap={_fmt(row.get('gap_closed'))} "
                    f"bits={row.get('bit_changed')} sens={row.get('order_sensitive')} "
                    f"fired={row.get('pass_changed_ir')}  [{rate:.1f}s/row]", flush=True)
        if stopped_early:
            break
    out.close()
    K.reset_launches()

    # The file is appended to a row at a time so a kill costs one row, which means a row redone
    # later (bits now, timing once a GPU frees) is in it twice. Collapse it, newest wins, through a
    # temporary file so an interrupted rewrite cannot lose the whole run.
    if done:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in done.values():
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)

    if pauses:
        note = (f"  PAUSED {pauses} time(s) for {paused_s / 60:.1f} minutes in total, because another process was "
                "on the pinned GPU. Nothing was measured or written while paused, so no row below was timed "
                "next to a neighbour.")
        print("\n" + note)
        header += [note, ""]
    if stopped_early:
        left = n_total - len(done)
        why = joined or f"the {args.minutes:.0f} minute budget ran out"
        note = (f"  PARTIAL: {why}, with {left} of {n_total} configurations left. Re-run the same command to "
                "continue; rows already taken are not repeated.")
        print("\n" + note)
        header += [note, ""]
    untimed = sum(1 for r in done.values()
                  if r.get("speedup") is None and r.get("base_compiles") and not r.get("compile_regression"))
    if untimed:
        header += [
            f"  NOT TIMED: {untimed} rows carry bits but no time ({gpu_note or 'the GPU was in use'}), so their",
            "        speedup and gap_closed columns are empty. The bit columns are real. Re-running on a free",
            "        device fills in exactly those rows and leaves the finished ones alone.", ""
        ]

    _write_report(done.values(), kernels, header, K)


def _write_report(rows, kernels, header, K):
    """Render the table and keep the header, so it can be rendered again without a GPU."""
    rows = list(rows)
    probe = K.fp8_order_invariance_probe() if any(r.get("order_sensitive") == 0 for r in rows) else None
    text = report(rows, kernels, header, probe)
    with open(REPORT, "w") as f:
        f.write("\n".join(text) + "\n")
    with open(os.path.join(CACHE, f"{TABLE}.header.txt"), "w") as f:
        f.write("\n".join(header) + "\n")
    print("\n".join(text[len(header):]))
    print(f"\nWROTE {REPORT}")


def _report_only(K):
    """Rebuild the table from the records already in `cache/`, launching no kernel.

    The same idea as `artifact.py --export`: the numbers are an input here, not something this
    path can change. Only the prose around them is re-rendered, which is what makes it safe to
    improve an explanation after a two-hour run without paying for the run again. The header is
    the one the run itself wrote, kept beside the records, so the machine, the self-checks and
    the GPU state reported are the ones the numbers were taken under and not today's.
    """
    path = os.path.join(CACHE, f"{TABLE}.jsonl")
    head = os.path.join(CACHE, f"{TABLE}.header.txt")
    if not os.path.exists(path) or not os.path.exists(head):
        print(f"report-only needs both {path} and {head}; run the step once first.")
        return
    rows = [json.loads(x) for x in open(path) if x.strip()]
    header = open(head).read().rstrip("\n").split("\n")
    print(f"rebuilding the table from {len(rows)} records in {path}; nothing is measured.")
    _write_report(rows, K.catalogue(), header, K)
