"""inner_tree.bitmatch -- when the reduction order is pinned, do the bits actually stop moving?

    with reduction_ordering=inner_tree, changing the layout knobs does not change a single
    output byte; with unordered, on the same kernels and the same inputs, it does.

Both halves are the claim. A cell whose `inner_tree` arm is invariant proves nothing on its own:
if the matching `unordered` arm does not differ either, the sweep is blind on that kernel and the
row is no evidence in either direction. `sensitive` is the column that says which it is, and the
headline count at the bottom of the report is taken over the cells that have it.

THE DESIGN
----------
A **cell** is a fixed `(kernel, dtype, enable_fp_fusion, ordering)`. Inside a cell the layout axes
are swept, every point is compiled and launched on the same input draws, and the outputs are
grouped by their exact bytes. The cell is invariant when that group count is 1.

    swept       num_warps, num_stages          the mode has to make these irrelevant
    cell axes   reduction_ordering,            genuinely bit-relevant; sweeping them would
                enable_fp_fusion, dtype        manufacture a failure that is not the mode's

`num_warps` and `num_stages` are swept JOINTLY, all 6 x 6 of them, not one at a time. A mode that
survives each axis alone and breaks on the pair is exactly what a one-axis-at-a-time sweep cannot
see.

`enable_fp_fusion` is a cell axis and not a swept one because it decides FMA contraction below
TTGIR. `inner_tree` fixes the shape of the add tree and says nothing about whether a multiply
folds into an add, so on a mul-fed kernel such as `col_dot` the two settings genuinely give
different bits and that is not a failure of the ordering.

`block_n` is deliberately NOT swept, and should not be added later. On a chunked reduction --
`col_sum_loop`, `G_plain_sum_looped` -- `inner_tree` fixes the order within one `tl.sum`, not the
accumulation ACROSS loop iterations, so the chunk width stays bit-relevant by design. Sweeping it
would produce a failing row that says nothing about whether the ordering switch works.

TWO THINGS THAT WOULD SILENTLY FAKE A PASS
------------------------------------------
**An fp8 pure sum cannot test this at all.** An e4m3 value carries a four-bit significand, so the
exact sum of a few hundred of them still fits inside an f32 mantissa and every association order
gives the same answer -- order-invariant by arithmetic, not because the mode did anything. Those
rows are no evidence, not weak evidence. The step measures it rather than asserting it
(`fp8_order_invariance_probe`) and labels the rows through `sensitive`; it never counts them.
`exp()` and `a*b` widen the leaves to f32 first, so `col_exp_sum` and `col_dot` keep their fp8
sensitivity -- which is why this is decided per kernel, from the measurement, and not per dtype.

**Narrow inputs hide regrouping.** Every byte comparison here runs on the wide draw
(`_inner_tree_kernels.adv_nd`): exponents spread across the dtype's usable range with alternating
signs, so the sum cancels. On tame unit-scale data almost any regrouping rounds to the same bits
and the check passes things it should not.

CONTROLS
--------
`col_max` is a `tl.max` reduction. Max is order-invariant as an operation, so the cell must come
back invariant in BOTH arms for reasons that have nothing to do with the ordering switch. It is
labelled a control (`control = 1`), it is what a working harness looks like on an unmovable
kernel, and it is never counted toward the claim.

TWO TRAPS FROM THE NEIGHBOURING STEP
------------------------------------
Triton's in-memory kernel cache is keyed on `str(specialization) + str(options)` only
(`compute_cache_key`, `jit.py`). Every axis this step sweeps is inside that key, so unlike
`inner_tree.layout` the cache should not confuse two of these builds -- but "should" is how a
table comes back a meaningless, perfect 1.00 invariant everywhere. So the cache is cleared before
every build, `TRITON_ALWAYS_COMPILE=1` is set at module import (before `artifact.py` imports
triton), and the step refuses to measure until it has WATCHED two builds that must differ actually
produce different TTGIR -- once for a layout knob and once for the ordering itself.

`TRITON_STRICT_REDUCTION_ORDERING` does nothing for a kernel that takes `reduction_ordering` as a
constexpr, which is all of these. The ordering is passed explicitly on every build; the env var is
recorded on every row so a reader knows it was not what did the work, and never relied on.

Attribution. The ordering mechanism this step evaluates is not ours. `reduction_ordering` /
`inner_tree` and the `TRITON_STRICT_REDUCTION_ORDERING` environment variable were written by Nick
Riasanovsky, a co-author. What is ours is the measurement: asking whether the guarantee holds bit
for bit across the configurations an autotuner would actually try.

HOW TO RUN

    export PYTHONPATH=$(git rev-parse --show-toplevel)
    CUDA_VISIBLE_DEVICES=<a gpu> python artifact_eval/artifact.py --run inner_tree.bitmatch \
        --minutes 600

Nothing else has to be set. Nothing here is timed, so a neighbour on the same GPU does not change
a single number -- only device memory and wall clock are shared.

    INNER_TREE_BITMATCH_PREFLIGHT=1      self-checks, the cell table and the counts; measures one
                                         cell pair, writes nothing. Minutes, not hours.
    INNER_TREE_BITMATCH_KERNELS=a,b      restrict the kernel list.
    INNER_TREE_BITMATCH_DTYPES=f16,f32   restrict the dtypes.
    INNER_TREE_BITMATCH_SEEDS=5          input draws every configuration is compared on.
    INNER_TREE_BITMATCH_WARPS=1,2,4      restrict the num_warps sweep.
    INNER_TREE_BITMATCH_STAGES=1,2       restrict the num_stages sweep.

`--minutes` is a resumable budget, not a sample size. Rows stream to
`cache/inner_tree.bitmatch.jsonl` one cell at a time and a second invocation picks up where the
first stopped, so a short run repeated gives the same table as one long run. A row taken with a
different seed count or a different swept space is redone rather than reused.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import namedtuple

# Set before triton is imported. `artifact.py` imports every step module during discovery, which
# happens before it imports torch or triton, so this is the last moment that still works.
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

from ._common import CACHE, writer  # noqa: E402

NAME = "inner_tree.bitmatch"
ORDER = 60
DESCRIPTION = "does the enforced reduction order hold, bit for bit"
IMPLEMENTED = True

# The swept axes: what an autotuner moves and what the mode has to make irrelevant. Same values as
# `inner_tree.layout` sweeps, so a kernel that fails to build at num_warps=32 there fails here too
# and the two steps' `n_failed` columns mean the same thing.
NUM_WARPS = (1, 2, 4, 8, 16, 32)
NUM_STAGES = (1, 2, 3, 4, 5, 6)
SWEPT_AXES = ("num_warps", "num_stages")

# The cell axes: bit-relevant on purpose, held fixed inside a cell.
ORDERINGS = ("inner_tree", "unordered")
FP_FUSION = (True, False)

Config = namedtuple("Config", "num_warps num_stages enable_fp_fusion")

TABLE = "inner_tree.bitmatch"
REPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_inner_tree_bitmatch.txt")

TABLES = {
    TABLE: {
        "doc":
        "One row per cell, where a cell is a fixed (kernel, dtype, enable_fp_fusion, ordering). "
        "Inside a cell the layout axes num_warps and num_stages are swept jointly, every point is "
        "compiled and launched on the same input draws, and the outputs are grouped by their "
        "exact bytes; `n_bit_classes` is the number of groups and 1 means the cell is invariant. "
        "Three columns carry the argument. `n_bit_classes` is the result. `sensitive` says "
        "whether the matching `unordered` cell actually returned more than one byte pattern -- a "
        "row with 0 there proves nothing in either direction, because the sweep never moved the "
        "bits on that kernel at all. `split_axes` names the axis whose value differs between two "
        "configurations that disagreed, which is the finding; the count on its own is not. "
        "`n_configs`, `n_ran` and `n_failed` are there so a cell that silently lost most of its "
        "space cannot read as clean, and `cache_defeat_verified` is the analogue of "
        "`pass_available` in the sibling step: 0 means the run never proved two builds that must "
        "differ do differ, and then every invariant row is suspect.",
        "cols": [
            ("kernel", "str", "kernel name; the bodies come from bitequiv/evaluation, imported not retyped"),
            ("suite", "str", "which source file it comes from: synthetic, inductor, zoo or control"),
            ("kernel_source", "str", "file and symbol of the kernel body, so it can be read"),
            ("dtype", "str", "element dtype: f16, bf16, f32 or fp8. A cell axis, not swept"),
            ("enable_fp_fusion", "int", "1 = on. A cell axis, not swept: it decides FMA contraction "
             "below TTGIR, which inner_tree says nothing about, so on a mul-fed reduce the two "
             "settings genuinely differ"),
            ("ordering", "str", "reduction_ordering under test: inner_tree or unordered. A cell axis"),
            ("control", "int", "1 if this kernel is an order-invariant control (a max reduction). It "
             "must come back invariant in BOTH arms for reasons unrelated to the ordering, and it "
             "is never counted toward the claim"),
            ("axes", "str", "the configuration axes swept inside the cell"),
            ("axis_values", "str", "the values each swept axis took, so a row carries the space it covered"),
            ("n_configs", "int", "configurations in the sweep, i.e. points in the cell"),
            ("n_ran", "int", "of those, how many compiled and launched"),
            ("n_failed", "int", "configurations that failed to compile or launch"),
            ("seeds", "int", "random input draws each configuration was run on; the outputs of all of "
             "them together are what a configuration is grouped by"),
            ("n_bit_classes", "int", "distinct byte outputs across the configurations that ran. 1 = invariant"),
            ("largest_class", "int", "size of the biggest group of configurations returning the same bytes"),
            ("class_sizes", "str", "the group sizes, largest first"),
            ("invariant", "int", "1 if n_bit_classes is 1 and at least two configurations ran"),
            ("sensitive", "int", "1 if the matching `unordered` cell -- same kernel, dtype and "
             "enable_fp_fusion -- returned more than one byte pattern. 0 means the sweep is blind "
             "here and this row is no evidence. Blank if that cell was not measured"),
            ("partner_classes", "int", "n_bit_classes of that matching unordered cell"),
            ("split_axes", "str", "the axes whose value changes between two configurations that "
             "disagreed; empty when invariant. This is the finding, not n_bit_classes on its own"),
            ("split_example", "str", "two configurations that disagreed, spelled out"),
            ("nan_frac", "float", "fraction of the output that is NaN or Inf. A saturated output "
             "compares equal to itself, so a high value weakens an invariant verdict"),
            ("strict_env", "int", "1 if TRITON_STRICT_REDUCTION_ORDERING was set before triton was "
             "imported. Recorded, never relied on: it does nothing for a kernel that takes "
             "reduction_ordering as a constexpr, which is all of these"),
            ("ordering_passed_as", "str", "how the ordering reached the kernel; always the explicit "
             "constexpr argument, never the environment variable"),
            ("always_compile", "int", "1 if TRITON_ALWAYS_COMPILE was set, defeating the on-disk cache"),
            ("cache_defeat_verified", "int", "1 if this run watched two builds that must differ produce "
             "different TTGIR, for a layout knob and for the ordering. 0 invalidates every invariant row"),
            ("verdict", "str", "invariant, NOT INVARIANT, no evidence, control, or why the cell "
             "produced no result"),
            ("error", "str", "non-empty if the cell failed to produce a result; the first failure if "
             "some configurations failed and others ran"),
        ],
    },
}


# ------------------------------------------------------------------------------------------- #
# The space
# ------------------------------------------------------------------------------------------- #
def config_space(enable_fp_fusion, warps=NUM_WARPS, stages=NUM_STAGES):
    """The joint sweep. num_warps is the outer loop so a cell cut short leaves whole num_warps
    groups finished rather than a ragged edge."""
    return [Config(w, s, enable_fp_fusion) for w in warps for s in stages]


def config_label(c):
    return f"num_warps={c.num_warps} num_stages={c.num_stages}"


def axis_values(warps, stages):
    return f"num_warps={','.join(map(str, warps))} num_stages={','.join(map(str, stages))}"


def control_kernels(K):
    """The order-invariant controls, built here so the sibling step's catalogue is not edited.

    `col_max` reduces with `tl.max`, which has no order to fix: the answer is the same whatever the
    layout does, in both arms, and `_k_col_max` does not even take the ordering into its body. A
    harness that reported this one as varying would be broken, and a harness that cannot report it
    as invariant cannot be believed when it says the same of a sum.
    """
    from bitequiv.evaluation import eval_kernels as ek

    src = "bitequiv/evaluation/eval_kernels.py"
    build = _control_build(K, dict(M=256, C=32), 256 * 32, 32)
    what = ("[M=256,C=32] column MAX over axis 0: order-invariant as an operation, so it must come back "
            "invariant in BOTH arms -- the harness's own control")
    return [
        K.Kernel(name="col_max", suite="control", source=f"{src} :: _k_col_max", what=what, jit=ek._k_col_max,
                 build=build, dtypes=("f16", "f32"), prior_dtype="f32", tags=("control", "order-invariant"))
    ]


def _control_build(K, consts, tile_numel, out_numel):
    """The `eval_kernels` launch shape: one input buffer, one f32 output, tile dims as constexprs,
    one program per tile. Same sizes the sibling step uses for the synthetic kernels."""

    def build(torch, dtype_name, seed, wide, size):
        grid = 64 if size == "precision" else 2048
        dt = K.torch_dtype(torch, dtype_name)
        draw = K.adv_nd if wide else K.plain_nd
        ins = [draw(torch, grid * tile_numel, seed, dt)]
        out = torch.zeros(grid * out_numel, device=K.DEVICE, dtype=torch.float32)
        return K.Launch(args=[*ins, out], outs=[out], consts=dict(consts), grid=grid)

    return build


# ------------------------------------------------------------------------------------------- #
# Grouping configurations by the bytes they returned
# ------------------------------------------------------------------------------------------- #
def output_digest(chunks):
    """One key per configuration, over the outputs of every seed at once.

    Two configurations are in the same class only if they agreed on every draw, so a difference on
    any single draw splits them. sha256 of the exact bytes: the comparison is byte equality, the
    hash is only there so a cell does not hold 36 output buffers at once.
    """
    h = hashlib.sha256()
    for c in chunks:
        h.update(len(c).to_bytes(8, "little"))
        h.update(c)
    return h.hexdigest()


def classify(per_config):
    """Group the configurations by output bytes and say which axis split them.

    `per_config` maps (num_warps, num_stages) -> digest. Returns the class count, the class sizes,
    the axes responsible and one worked example.

    The axis is found from pairs that differ in exactly ONE swept axis: those are the pairs where
    the answer is unambiguous. On the full grid that is enough -- if two points disagree, walking
    from one to the other one axis at a time has to cross a step where the class changes, and that
    step differs in one axis. A cell whose grid is ragged (some configurations failed to compile)
    can have no such pair, and then the example pair's differing axes are reported instead, marked
    as joint, rather than reporting no axis at all.
    """
    digests = sorted(set(per_config.values()))
    sizes = sorted((sum(1 for d in per_config.values() if d == x) for x in digests), reverse=True)
    if len(digests) <= 1:
        return len(digests), sizes, "", ""
    axes, example = set(), ""
    items = sorted(per_config.items())
    for (w1, s1), d1 in items:
        for (w2, s2), d2 in items:
            if d1 == d2 or (w1, s1) >= (w2, s2):
                continue
            differing = [a for a, x, y in (("num_warps", w1, w2), ("num_stages", s1, s2)) if x != y]
            if len(differing) == 1:
                axes.add(differing[0])
                if not example:
                    example = f"num_warps={w1} num_stages={s1} vs num_warps={w2} num_stages={s2}"
    if not axes:
        for (w1, s1), d1 in items:
            for (w2, s2), d2 in items:
                if d1 == d2 or (w1, s1) >= (w2, s2):
                    continue
                example = f"num_warps={w1} num_stages={s1} vs num_warps={w2} num_stages={s2} (jointly)"
                return len(digests), sizes, "num_warps num_stages (jointly only)", example
    return len(digests), sizes, " ".join(sorted(axes)), example


# ------------------------------------------------------------------------------------------- #
# One cell
# ------------------------------------------------------------------------------------------- #
class Harness:
    """Everything a cell needs, gathered once: the kernel catalogue, the cache-clearing list and
    the seed count."""

    def __init__(self, kernels, seeds, warps, stages):
        from . import _inner_tree_kernels as K
        self.K = K
        self.kernels = kernels
        self.seeds = seeds
        self.warps = warps
        self.stages = stages
        self.jits = K.jit_functions(kernels)
        self.cache_defeat_verified = 0

    def build(self, kernel, dtype, ordering, config):
        """One compile with the in-memory cache dropped first.

        The clear is cheap and it is the difference between measuring 36 kernels and measuring one
        kernel 36 times. Every axis swept here is inside Triton's cache key, so in principle the
        lookup would be correct anyway -- `self_checks` is what turns that principle into something
        this run watched happen.
        """
        self.K.clear_caches(self.jits)
        return self.K.compile_kernel(kernel, dtype, ordering, config)


def measure_cell(h, kernel, dtype, enable_fp_fusion, ordering):
    """Sweep one cell and group its configurations by output bytes."""
    K = h.K
    configs = config_space(enable_fp_fusion, h.warps, h.stages)
    row = dict(kernel=kernel.name, suite=kernel.suite, kernel_source=kernel.source, dtype=dtype,
               enable_fp_fusion=int(enable_fp_fusion), ordering=ordering, control=int("control" in (kernel.tags or ())),
               axes=" ".join(SWEPT_AXES), axis_values=axis_values(h.warps, h.stages), n_configs=len(configs), n_ran=0,
               n_failed=0, seeds=h.seeds, strict_env=int(bool(os.environ.get("TRITON_STRICT_REDUCTION_ORDERING"))),
               ordering_passed_as="explicit constexpr argument",
               always_compile=int(bool(os.environ.get("TRITON_ALWAYS_COMPILE"))),
               cache_defeat_verified=h.cache_defeat_verified, error="")

    per_config, first_bytes = {}, None
    for cfg in configs:
        try:
            build = h.build(kernel, dtype, ordering, cfg)
            chunks = [K.run_bytes(build, seed) for seed in range(h.seeds)]
        except Exception as exc:  # noqa: BLE001
            row["n_failed"] += 1
            if not row["error"]:
                row["error"] = " ".join(f"{config_label(cfg)}: {type(exc).__name__}: {exc}".split())[:220]
            continue
        row["n_ran"] += 1
        per_config[(cfg.num_warps, cfg.num_stages)] = output_digest(chunks)
        if first_bytes is None:
            first_bytes = chunks[0]

    if first_bytes is not None:
        row["nan_frac"] = round(K.nan_fraction(first_bytes), 4)
    n_classes, sizes, axes, example = classify(per_config)
    row["n_bit_classes"] = n_classes
    row["largest_class"] = sizes[0] if sizes else 0
    row["class_sizes"] = " ".join(map(str, sizes))
    row["invariant"] = int(n_classes == 1 and row["n_ran"] >= 2)
    row["split_axes"] = axes
    row["split_example"] = example
    return row


def link_pairs(rows):
    """Fill in `sensitive`, `partner_classes` and `verdict`.

    A cell's evidence is a property of the PAIR, not of the cell: an invariant `inner_tree` cell
    means something only next to an `unordered` cell that actually moved. So the two arms are
    linked after the fact, which also means a run cut short between the two arms leaves an honest
    blank rather than a 0 that would read as "the sweep was blind here".
    """
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["kernel"], r["dtype"], r["enable_fp_fusion"]), {})[r["ordering"]] = r
    for arms in by_pair.values():
        un = arms.get("unordered")
        for r in arms.values():
            if un is not None and un.get("n_ran", 0) >= 2:
                r["partner_classes"] = un["n_bit_classes"]
                r["sensitive"] = int(un["n_bit_classes"] > 1)
            else:
                r["partner_classes"] = None
                r["sensitive"] = None
            r["verdict"] = verdict_of(r)
    return rows


def verdict_of(r):
    if r.get("n_ran", 0) == 0:
        return "no configuration compiled"
    if r.get("n_ran", 0) < 2:
        return "only one configuration ran -- nothing to compare"
    if r.get("control"):
        return "control: invariant, as it must be" if r["invariant"] else "CONTROL NOT INVARIANT -- harness suspect"
    if not r["invariant"]:
        return "differs, as expected" if r["ordering"] == "unordered" else "NOT INVARIANT"
    if r.get("sensitive") is None:
        return "invariant; unordered arm not measured"
    if not r["sensitive"]:
        return "no evidence: unordered did not differ"
    return "invariant" if r["ordering"] == "inner_tree" else "invariant (blind pair)"


# ------------------------------------------------------------------------------------------- #
# Self-checks. Without these an invariant table is not a result.
# ------------------------------------------------------------------------------------------- #
def self_checks(h):
    """Three things that, if wrong, make every invariant row below a false pass.

    1. A swept axis reaches the compiler. Two builds of one kernel differing only in num_warps
       have to produce different TTGIR. If they do not, the second build was served the first one
       out of the in-memory cache and the whole table is one kernel measured 36 times.
    2. The ordering reaches the compiler. `inner_tree` and `unordered` on the same configuration
       have to produce different TTGIR somewhere in the kernel list. If the ordering argument were
       being dropped, both arms would be the same build and BOTH would read invariant -- the exact
       shape of a clean, meaningless pass.
    3. A build is byte-reproducible against itself, on every kernel. A kernel that does not agree
       with its own previous launch would report as several bit classes for reasons that have
       nothing to do with the layout.
    """
    lines, ok = [], True
    lines.append(f"  TRITON_ALWAYS_COMPILE            {os.environ.get('TRITON_ALWAYS_COMPILE') or 'NOT SET'}")
    strict = os.environ.get("TRITON_STRICT_REDUCTION_ORDERING")
    lines.append(f"  TRITON_STRICT_REDUCTION_ORDERING {strict or 'NOT SET'}   (not relied on: it does nothing for a "
                 "kernel that takes")
    lines.append("                                   reduction_ordering as a constexpr, which is all of these, so the "
                 "ordering")
    lines.append("                                   is passed explicitly on every build instead)")

    warp_probe = order_probe = None
    tried = []
    for kernel in h.kernels:
        for dtype in kernel.dtypes:
            base = Config(4, 2, True)
            try:
                b4 = h.build(kernel, dtype, "unordered", base)
                b8 = h.build(kernel, dtype, "unordered", base._replace(num_warps=8))
                b_it = h.build(kernel, dtype, "inner_tree", base)
            except Exception as exc:  # noqa: BLE001
                tried.append(f"{kernel.name}/{dtype}: {type(exc).__name__}")
                continue
            if warp_probe is None and b4.ttgir != b8.ttgir:
                warp_probe = f"({kernel.name} {dtype} num_warps 4 vs 8)"
            if order_probe is None and b4.ttgir != b_it.ttgir:
                order_probe = f"({kernel.name} {dtype} unordered vs inner_tree at num_warps=4)"
            if warp_probe and order_probe:
                break
        if warp_probe and order_probe:
            break

    lines.append(f"  a layout knob changes the IR     {'YES  ' + warp_probe if warp_probe else 'NO'}")
    lines.append(f"  the ordering changes the IR      {'YES  ' + order_probe if order_probe else 'NO'}")
    if not (warp_probe and order_probe):
        ok = False
        lines.append("  ABORT: two builds that must differ came back identical. Either an argument is not")
        lines.append("         reaching the compiler, or a compile cache is serving one build to both arms.")
        lines.append("         Triton's in-memory cache is keyed on the specialization and the launch options")
        lines.append("         only; relaunch with TRITON_ALWAYS_COMPILE=1 set before the process starts.")
        for t in tried[:8]:
            lines.append(f"           {t}")
    h.cache_defeat_verified = int(bool(warp_probe and order_probe))
    return lines, ok


def determinism_sweep(h, plan):
    """Every (kernel, dtype) has to reproduce its own bytes. Cheap, and it catches the one failure
    a byte comparison is most exposed to: an output buffer the kernel does not fully write, whose
    leftover contents differ between two launches. That would read as several bit classes on a
    kernel that never moved."""
    lines, bad, seen = [], [], set()
    for kernel, dtype, _fpf in plan:
        if (kernel.name, dtype) in seen:
            continue
        seen.add((kernel.name, dtype))
        same, err = None, ""
        for cfg in (Config(4, 2, True), Config(8, 2, True), Config(1, 2, True)):
            try:
                b = h.build(kernel, dtype, "inner_tree", cfg)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                continue
            same = h.K.run_bytes(b, 0) == h.K.run_bytes(b, 0)
            break
        if same is None:
            lines.append(f"      {kernel.name:24} {dtype:5} NO BUILD AT ALL  {err}"[:118])
            bad.append(kernel.name)
        elif not same:
            lines.append(f"      {kernel.name:24} {dtype:5} NOT REPRODUCIBLE")
            bad.append(kernel.name)
    head = (f"  every kernel reproduces itself   {'YES' if not bad else 'NO'}"
            f"   ({len(seen) - len(bad)}/{len(seen)} (kernel, dtype) pairs)")
    return [head] + lines, not bad


# ------------------------------------------------------------------------------------------- #
# Report
# ------------------------------------------------------------------------------------------- #
def _cell(r, key, blank="-"):
    v = r.get(key) if r else None
    return blank if v is None else str(v)


def report(rows, header, probe):
    out = list(header)
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["kernel"], r["dtype"], r["enable_fp_fusion"]), {})[r["ordering"]] = r

    n_points = max((r.get("n_configs") or 0) for r in rows) if rows else 0
    out += [
        "", "THE PAIRS -- one line per (kernel, dtype, enable_fp_fusion), both arms side by side", "=" * 118, "",
        f"`cls` is the number of distinct byte outputs the {n_points} configurations produced; 1 means the layout",
        f"knobs did not move a single byte. `ran` is how many of the {n_points} compiled and launched -- a cell that",
        "lost most of its space is not clean, it is empty. The argument is the pair: `inner_tree cls = 1` says",
        "something only when `unordered cls > 1` on the same kernel, the same dtype and the same inputs. Where it",
        "does not, the row is marked `no evidence` and is not counted. The CSV carries one row per cell, not pair.", ""
    ]
    head = (f"{'kernel':24} {'dtype':5} {'fpf':>4} | {'it_cls':>6} {'it_ran':>7} | {'un_cls':>6} {'un_ran':>7} | "
            f"{'sens':>4} | {'verdict (inner_tree arm)':38} split")
    out += [head, "-" * len(head)]
    for key in sorted(by_pair):
        arms = by_pair[key]
        it, un = arms.get("inner_tree"), arms.get("unordered")
        kernel, dtype, fpf = key
        sens = _cell(it, "sensitive")
        line = (f"{kernel:24} {dtype:5} {('on' if fpf else 'off'):>4} | "
                f"{_cell(it, 'n_bit_classes'):>6} "
                f"{(str(_cell(it, 'n_ran')) + '/' + str(_cell(it, 'n_configs'))):>7} | "
                f"{_cell(un, 'n_bit_classes'):>6} "
                f"{(str(_cell(un, 'n_ran')) + '/' + str(_cell(un, 'n_configs'))):>7} | "
                f"{sens:>4} | {(it or {}).get('verdict', 'not measured'):38} "
                f"{(un or {}).get('split_axes', '')}")
        out.append(line)

    # ---- where the unordered arm split, and on which axis ----------------------------------- #
    splits = [(k, r) for k, arms in sorted(by_pair.items()) for o, r in sorted(arms.items()) if r.get("split_axes")]
    out += [
        "", "", "WHICH AXIS MOVES THE BITS WHEN THE ORDER IS NOT PINNED", "=" * 118, "",
        "Every cell that produced more than one byte pattern, with the axis responsible and a pair of",
        "configurations that disagreed. An `inner_tree` row here is a refutation of the claim; an `unordered`",
        "row here is the control working -- it is what makes the matching inner_tree row mean anything.", ""
    ]
    if not splits:
        out += [
            "  none: no cell in this run produced more than one byte pattern. That is not a pass -- it means",
            "  the sweep never moved the bits anywhere, so it could not have caught them moving."
        ]
    else:
        head2 = f"{'kernel':24} {'dtype':5} {'fpf':>4} {'ordering':11} {'cls':>4} {'sizes':>12}  {'axis':26} example"
        out += [head2, "-" * len(head2)]
        for (kernel, dtype, fpf), r in splits:
            out.append(f"{kernel:24} {dtype:5} {('on' if fpf else 'off'):>4} {r['ordering']:11} "
                       f"{r['n_bit_classes']:>4} {r['class_sizes'][:12]:>12}  {r['split_axes']:26} "
                       f"{r['split_example']}")

    # ---- the gate ---------------------------------------------------------------------------- #
    real = [r for r in rows if not r.get("control")]
    controls = [r for r in rows if r.get("control")]
    it_rows = [r for r in real if r["ordering"] == "inner_tree"]
    with_evidence = [r for r in it_rows if r.get("sensitive") == 1]
    broken = [r for r in with_evidence if not r.get("invariant")]
    blind = [r for r in it_rows if r.get("sensitive") == 0]
    unpaired = [r for r in it_rows if r.get("sensitive") is None]
    thin = [r for r in real if r.get("n_ran", 0) < 2]
    bad_controls = [r for r in controls if not r.get("invariant") and r.get("n_ran", 0) >= 2]
    verified = all(r.get("cache_defeat_verified") for r in rows) if rows else False

    out += [
        "",
        "",
        "GATE",
        "=" * 118,
        "",
        f"  cells measured                     {len(rows)}   ({len(real)} real, {len(controls)} control)",
        f"  cache defeat verified on every row {int(verified)}   (must be 1: two builds that must differ were "
        "watched differing)",
        f"  inner_tree cells                   {len(it_rows)}",
        f"  of those, with a sensitive partner {len(with_evidence)}   (the matching unordered cell returned more "
        "than one byte pattern)",
        f"  of those, NOT invariant            {len(broken)}   (must be 0; this is the claim)",
        f"  inner_tree cells with no evidence  {len(blind)}   (the unordered arm did not differ either, so these "
        "prove nothing",
        "                                     in either direction and are not counted)",
        f"  inner_tree cells with no partner   {len(unpaired)}   (the unordered arm was not measured -- a partial run)",
        f"  cells where fewer than 2 ran       {len(thin)}   (nothing to compare inside them)",
        f"  controls not invariant             {len(bad_controls)}   (must be 0: a max reduction is order-invariant "
        "by operation,",
        "                                     so a control that moves means the harness is broken, not the mode)",
    ]

    if blind and probe:
        fp8, f16 = probe.get("fp8"), probe.get("f16")
        out += [
            "",
            "  Why the fp8 pure sums are among the cells with no evidence. An fp8 e4m3 value carries a four-bit",
            "  significand, so the exact sum of a few hundred of them still fits inside an f32 mantissa, and then",
            "  every order gives the same answer -- order-invariant by arithmetic, not because the ordering switch",
            "  did anything. Measured off-device on this run, on the same draw the kernels use:",
        ]
        if fp8:
            out.append(f"      fp8   {fp8[0]}/{fp8[2]} draws sum exactly in f32; sequential vs pairwise tree "
                       f"differed on {fp8[1]}/{fp8[2]}")
        if f16:
            out.append(f"      f16   {f16[0]}/{f16[2]} draws sum exactly in f32; sequential vs pairwise tree "
                       f"differed on {f16[1]}/{f16[2]}")
        out += [
            "  So an fp8 pure-sum cell coming back with one bit class is not weak evidence, it is none. `exp()` and",
            "  `a*b` widen the leaves to f32 before the reduce, so `col_exp_sum` and `col_dot` keep their fp8",
            "  sensitivity; whether a kernel keeps it is decided per kernel from the measurement above, not per",
            "  dtype from this paragraph.",
        ]
    if blind:
        out += ["", "  The cells with no evidence, in full:"]
        for r in sorted(blind, key=lambda x: (x["kernel"], x["dtype"], x["enable_fp_fusion"])):
            out.append(f"      {r['kernel']:24} {r['dtype']:5} fp_fusion={'on' if r['enable_fp_fusion'] else 'off':3} "
                       f"  unordered returned {r.get('partner_classes')} bit class over {r.get('n_ran')} "
                       "configurations")
    sat = {}
    for r in rows:
        f = r.get("nan_frac") or 0.0
        if f > 0.01:
            sat[(r["kernel"], r["dtype"])] = max(sat.get((r["kernel"], r["dtype"]), 0.0), f)
    if sat:
        out += [
            "", "  Partly saturated outputs. A NaN or an Inf compares equal to itself, so on the saturated part",
            "  of these outputs no regrouping could show. These cells are still order-sensitive, so the check is",
            "  weakened rather than vacuous:"
        ]
        for (k, d), f in sorted(sat.items()):
            out.append(f"      {k:24} {d:5} up to {100 * f:.1f}% of the output is NaN or Inf")

    # ---- the headline ------------------------------------------------------------------------ #
    out += [
        "",
        "",
        "HEADLINE",
        "=" * 118,
        "",
        f"  {len(rows)} cells, of which {len(with_evidence)} were inner_tree cells whose matching unordered cell "
        f"actually differed;",
        f"  {len(broken)} of those {len(with_evidence)} were not invariant.",
        "",
        "  The space, in full, because this is a universal claim and a count without one is worth nothing:",
        f"      swept jointly   {rows[0]['axis_values'] if rows else '(nothing)'}",
        "      cell axes       reduction_ordering x enable_fp_fusion x dtype, held fixed inside a cell",
        f"      draws per point {rows[0]['seeds'] if rows else 0}, wide exponent range, alternating signs",
        f"      configurations  {sum(r.get('n_ran', 0) for r in rows)} compiled and launched, "
        f"{sum(r.get('n_failed', 0) for r in rows)} failed to build, "
        f"{sum(r.get('n_configs', 0) for r in rows)} attempted",
        "",
        "  What this does NOT say. It says that on these kernels, at these dtypes, over this num_warps x",
        "  num_stages grid, no inner_tree cell with a sensitive partner returned two different byte patterns.",
        "  It does not say the guarantee holds everywhere. Read the space above before generalising, and read",
        f"  the {len(blind)} cells with no evidence as untested rather than as passing.",
    ]
    hard_fail = bool(broken) or bool(bad_controls) or not verified
    if hard_fail:
        out += ["", "  RESULT: FAIL -- see the counts above."]
    else:
        ctl = (f"{len(controls)} control cells came back invariant"
               if controls else "NO control cell was reached in this run")
        out += [
            "", "  RESULT: PASS -- every inner_tree cell that could have moved did not, "
            f"{ctl},", "          and the run watched two builds that must differ actually differ."
        ]
    out.append("")
    return out


# ------------------------------------------------------------------------------------------- #
# Entry point
# ------------------------------------------------------------------------------------------- #
def _int_list(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return tuple(int(x) for x in raw.split(",") if x.strip())


def run(args, env):
    from . import _inner_tree_kernels as K

    preflight = os.environ.get("INNER_TREE_BITMATCH_PREFLIGHT", "").strip() == "1"
    want_kernels = [k.strip() for k in os.environ.get("INNER_TREE_BITMATCH_KERNELS", "").split(",") if k.strip()]
    want_dtypes = [d.strip() for d in os.environ.get("INNER_TREE_BITMATCH_DTYPES", "").split(",") if d.strip()]
    seeds = int(os.environ.get("INNER_TREE_BITMATCH_SEEDS", "5"))
    warps = _int_list("INNER_TREE_BITMATCH_WARPS", NUM_WARPS)
    stages = _int_list("INNER_TREE_BITMATCH_STAGES", NUM_STAGES)

    kernels = K.catalogue() + control_kernels(K)
    if want_kernels:
        kernels = [k for k in kernels if k.name in want_kernels]
    if want_dtypes:
        kernels = [k for k in kernels if any(d in want_dtypes for d in k.dtypes)]

    h = Harness(kernels, seeds, warps, stages)

    plan = []
    for kernel in kernels:
        for dtype in kernel.dtypes:
            if want_dtypes and dtype not in want_dtypes:
                continue
            for fpf in FP_FUSION:
                plan.append((kernel, dtype, fpf))
    n_cells = 2 * len(plan)
    n_configs = n_cells * len(warps) * len(stages)

    header = [
        "inner_tree.bitmatch -- does pinning the reduction order actually pin the bits?",
        "=" * 118,
        "",
        f"  when                  {env['when']}",
        f"  device                {env['device']}  ({env['arch']})",
        f"  triton                {env['triton']}   torch {env['torch']}   commit {env['commit']}",
        f"  gpu                   CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}"
        "   (nothing here is timed, so a neighbour on the same device changes no number)",
        "",
        "  claim   with reduction_ordering=inner_tree, changing the layout knobs does not change a single",
        "          output byte; with unordered, on the same kernels and the same inputs, it does.",
        "",
        "  cell    a fixed (kernel, dtype, enable_fp_fusion, ordering). Inside it the layout axes are swept,",
        "          every point runs on the same draws, and the outputs are grouped by their exact bytes.",
        "          The cell is invariant when that group count is 1.",
        "",
        f"  swept   {' '.join(SWEPT_AXES)} -- jointly, all {len(warps)} x {len(stages)} of them, because a mode that",
        "          survives each axis alone and breaks on the pair is what a one-axis sweep misses",
        f"          {axis_values(warps, stages)}",
        "  fixed   reduction_ordering, enable_fp_fusion, dtype. All three are genuinely bit-relevant, so",
        "          sweeping them would manufacture a failure that says nothing about the ordering. block_n is",
        "          deliberately absent for the same reason: on a chunked reduction inner_tree fixes the order",
        "          inside one tl.sum, not the accumulation across loop iterations.",
        "",
        f"  seeds   {seeds} draws per configuration, wide exponent range and alternating signs so the sum cancels;",
        "          on tame data almost any regrouping rounds to the same bits and the check passes everything",
        "",
        f"  space   {len(plan)} (kernel, dtype, enable_fp_fusion) triples -> {n_cells} cells "
        f"-> {n_configs} compile-and-launch points",
        "",
    ]
    print("\n".join(header), flush=True)

    checks, ok = self_checks(h)
    print("SELF-CHECKS")
    print("\n".join(checks), flush=True)
    header += ["SELF-CHECKS"] + checks
    if not ok:
        with open(REPORT, "w") as f:
            f.write("\n".join(header) + "\nABORTED: a self-check failed; nothing was measured.\n")
        return

    det, det_ok = determinism_sweep(h, plan)
    print("\n".join(det) + "\n", flush=True)
    header += det + [""]
    if not det_ok:
        with open(REPORT, "w") as f:
            f.write("\n".join(header) + "\nABORTED: a kernel is not byte-reproducible against itself.\n")
        return

    cat = ["THE CELLS", "=" * 118, ""]
    chead = f"{'kernel':24} {'suite':10} {'dtype':6} {'cells':>5} {'points':>7}  what it is"
    cat += [chead, "-" * len(chead)]
    for kernel, dtype, fpf in plan:
        if fpf:  # one line per (kernel, dtype); the two fp_fusion settings are on it
            cat.append(f"{kernel.name:24} {kernel.suite:10} {dtype:6} {4:>5} {4 * len(warps) * len(stages):>7}  "
                       f"{kernel.what}")
    cat += ["", "Kernel bodies are imported, not retyped, so they can be diffed against their source:"]
    for kernel in kernels:
        cat.append(f"    {kernel.name:24} {kernel.source}")
    cat += ["", "NOT MEASURED, and why:"]
    for name, where, why in K.OUT_OF_SUITE:
        cat.append(f"    {name:26} {where}")
        cat.append(f"        {why}")
    cat += ["", f"TOTAL {n_cells} cells x {len(warps) * len(stages)} configurations = {n_configs} points", ""]
    print("\n".join(cat), flush=True)
    header += cat

    # ---- resume ------------------------------------------------------------------------------ #
    done = {}
    path = os.path.join(CACHE, f"{TABLE}.jsonl")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # A row taken with a different seed count or over a different swept space answers a
            # different question; redo it rather than mixing the two in one table.
            if r.get("seeds") == seeds and r.get("axis_values") == axis_values(warps, stages):
                done[(r["kernel"], r["dtype"], r["enable_fp_fusion"], r["ordering"])] = r
    if done:
        print(f"  resuming: {len(done)} cells already in {path}\n", flush=True)

    if preflight:
        t0 = time.time()
        kernel, dtype, fpf = plan[0]
        for ordering in ORDERINGS:
            r = measure_cell(h, kernel, dtype, fpf, ordering)
            print(
                f"  {kernel.name:24} {dtype:5} fpf={int(fpf)} {ordering:11} classes={r['n_bit_classes']} "
                f"ran={r['n_ran']}/{r['n_configs']} split={r['split_axes']!r} {r['error'][:40]}", flush=True)
        per = (time.time() - t0) / 2.0
        print(f"\nPREFLIGHT: 2 cells in {time.time() - t0:.0f}s -> {per:.0f}s/cell")
        print(f"           the same work over {n_cells} cells projects to {per * n_cells / 3600:.1f} hours.")
        print("           Nothing was written. Drop INNER_TREE_BITMATCH_PREFLIGHT to run it for real.")
        return

    # ---- the sweep ---------------------------------------------------------------------------- #
    out = writer(TABLE)
    deadline = time.time() + float(args.minutes) * 60.0
    stopped_early, n_done, t0 = False, 0, time.time()
    # The input draws depend on (kernel, dtype, seed) and NOT on the configuration or the ordering,
    # so they are built once and every point in every cell of that pair sees the same bytes -- which
    # is required, not merely cheaper: two arms compared on different inputs compare nothing. They
    # are released only when the sweep leaves the pair.
    held = None
    for kernel, dtype, fpf in plan:
        if held != (kernel.name, dtype):
            K.reset_launches()
            held = (kernel.name, dtype)
        if all((kernel.name, dtype, int(fpf), o) in done for o in ORDERINGS):
            continue
        print(
            f"\n{kernel.name} [{dtype}] fp_fusion={'on' if fpf else 'off'} -- "
            f"{len(warps) * len(stages)} configurations per arm", flush=True)
        for ordering in ORDERINGS:
            key = (kernel.name, dtype, int(fpf), ordering)
            if key in done:
                continue
            if time.time() > deadline:
                stopped_early = True
                break
            try:
                row = measure_cell(h, kernel, dtype, fpf, ordering)
            except Exception as exc:  # noqa: BLE001
                row = dict(kernel=kernel.name, suite=kernel.suite, kernel_source=kernel.source, dtype=dtype,
                           enable_fp_fusion=int(fpf), ordering=ordering, control=int("control" in (kernel.tags or ())),
                           axes=" ".join(SWEPT_AXES), axis_values=axis_values(warps, stages), seeds=seeds,
                           n_configs=len(warps) * len(stages), n_ran=0, n_failed=0, verdict="error",
                           cache_defeat_verified=h.cache_defeat_verified, error=f"{type(exc).__name__}: {exc}"[:220])
            out.write(json.dumps(row) + "\n")
            out.flush()  # a killed run loses at most this one cell
            done[key] = row
            n_done += 1
            rate = (time.time() - t0) / max(n_done, 1)
            print(
                f"    {ordering:11} classes={row.get('n_bit_classes')} "
                f"ran={row.get('n_ran')}/{row.get('n_configs')} largest={row.get('largest_class')} "
                f"split={row.get('split_axes') or '-'}  [{rate:.0f}s/cell]", flush=True)
        if stopped_early:
            break
    out.close()
    K.reset_launches()

    rows = link_pairs(list(done.values()))
    # The file is appended to one cell at a time so a kill costs one cell; rewrite it collapsed and
    # linked, through a temporary file so an interrupted rewrite cannot lose the whole run.
    if rows:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)

    if stopped_early:
        left = n_cells - len(done)
        note = (f"  PARTIAL: the {args.minutes:.0f} minute budget ran out, with {left} of {n_cells} cells left. "
                "Re-run the same command to continue; cells already taken are not repeated.")
        print("\n" + note)
        header += [note, ""]

    probe = K.fp8_order_invariance_probe() if any(r.get("sensitive") == 0 for r in rows) else None
    text = report(rows, header, probe)
    with open(REPORT, "w") as f:
        f.write("\n".join(text) + "\n")
    print("\n".join(text[len(header):]))
    print(f"\nWROTE {REPORT}")
