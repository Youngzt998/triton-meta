"""checker.precision -- is the static equivalence checker sound, and how much tuning freedom does
it recover?

PLACEHOLDER. This mirrors the `precision` stage that `bitequiv/evaluation/evaluate.py` already
runs; the work is to drive it from here and record its numbers as a table, not to design a new
experiment.
"""
from __future__ import annotations

from ._common import print_design

NAME = "checker.precision"
ORDER = 80
DESCRIPTION = "is the checker sound, and how much of the search space does it recover"
IMPLEMENTED = False

# Both checkers are evaluated by the same framework; only these two arguments change. Written
# once here and imported by the other checker.* steps so the four tables agree on what a
# (checker, artifact) pair is called.
CHECKERS = (
    ("bitequiv.ptx_reduction:ptx_reduction_descriptor", "ptx"),
    ("bitequiv.ttgir_reduction:ttgir_reduction_descriptor", "ttgir"),
)

CHECKER_COLS = [
    ("checker", "str", "checker under test, as module:function"),
    ("artifact", "str", "which compiled IR the checker was fed: ptx, ttgir or amdgcn"),
    ("kernel", "str", "kernel spec from bitequiv/evaluation/eval_kernels.py, named <kernel>_<dtype>"),
    ("dtype", "str", "element dtype: f16, bf16, f32 or fp8"),
]

TABLES = {
    "checker.precision": {
        "doc":
        "One row per (checker, artifact, kernel). Compile the kernel's configuration "
        "space, let the checker group the configurations, and independently fuzz every "
        "configuration to get the empirical grouping. Two numbers matter and they point "
        "in opposite directions: over-merges is the soundness violation and must be 0; "
        "over-splits is tuning freedom the checker gave up to stay safe.",
        "cols":
        CHECKER_COLS + [
            ("effort", "str", "light, mid or heavy: how much of the configuration space was run "
             "and how large the input was"),
            ("seeds", "int", "random input draws per configuration in the fuzzer"),
            ("attempted", "int", "configurations in the space at this effort"),
            ("ok", "int", "of those, how many compiled and launched"),
            ("fails", "int", "configurations that failed to compile or launch"),
            ("checker_n", "int", "groups the checker produced"),
            ("checker_max", "int", "largest checker group; this is the search space it recovers"),
            ("empirical_n", "int", "groups the fuzzer produced"),
            ("empirical_max", "int", "largest empirical group; this is the recovery ceiling"),
            ("over_merges", "int", "pairs the checker called equal that the fuzzer separated. "
             "The soundness violation. MUST be 0"),
            ("over_splits", "int", "pairs the fuzzer merged that the checker separated. Safe, "
             "but it is recovery left on the table"),
            ("refines", "int", "1 if every checker group sits inside one empirical group, which "
             "is the formal soundness relation"),
            ("straddle", "int", "checker groups spanning more than one empirical group; 0 when refines is 1"),
            ("largest_spans", "str", "which configuration axes vary inside the largest checker "
             "group. This is what the recovered freedom is made of"),
            ("error", "str", "non-empty if the row failed to produce a result"),
        ],
    },
}


def run(args, env):
    print_design(
        NAME,
        claim="""
        The static checker is sound on the space it was measured over -- it never calls two
        configurations equal that actually return different bytes -- and it recovers a useful
        part of the tuning freedom rather than splitting every configuration into its own group.
        """,
        method="""
        Do not design this. `bitequiv/evaluation/evaluate.py --stages precision` already is it,
        and `bitequiv/evaluation/README.md` explains what it measures. Per kernel it builds the
        configuration space, compiles each configuration, asks the checker to group them from the
        IR alone (no launch), then independently launches every configuration on N random seeds
        and groups them by the output bytes. `equivalence_fuzzer.py` does the partition and the
        soundness arithmetic and is standalone, so the numbers do not depend on the checker being
        correct.

        Run it for both checkers, because they are peers and not a chain: the PTX checker
        (`bitequiv/ptx_reduction.py`, `--artifact ptx`) reconstructs the floating-point reduction
        tree from the PTX, and the TTGIR checker (`bitequiv/ttgir_reduction.py`,
        `--artifact ttgir`) reads the association order out of the parsed MLIR with the
        compiler's own layout machinery.

        Run at `--effort mid` for the table, `light` for a smoke run. Record the effort on the
        row, because the numbers are not comparable across efforts.
        """,
        cost="Tens of minutes at `light`, hours at `mid` over the whole registry. It compiles and "
        "launches every configuration N times.",
        see="Per row: the two group counts, the two largest groups, the over-merge and over-split "
        "counts, and which axes vary inside the biggest checker group.",
        judge="""
        `over_merges` = 0 is the gate. Anything else means the checker certified two
        configurations as identical and the fuzzer found an input where they were not, which is
        a soundness bug and not a tuning trade-off.

        The TTGIR checker is EXPECTED to over-merge and that is not a defect being hidden. TTGIR
        fixes the association order but is blind to FMA contraction -- whether a multiply feeding
        a reduction fuses into one rounded `fma` is decided below TTGIR and gated by
        `enable_fp_fusion` -- so on the multiply-fed kernels (`dot`, `welford`) it merges
        configurations whose bits differ. That is precisely the gap the PTX checker closes, and
        the two rows side by side are the finding. Run the TTGIR rows with `--allow-unsound` so
        the report is still produced.

        Then read `checker_max` against `empirical_max`. Equal means the checker recovered
        everything there was to recover; far below means it is safe but conservative, and
        `largest_spans` says which axes it did manage to merge across.

        What a clean run can conclude. "0 over-merges over N configurations on these kernels with
        these seeds." A fuzzer can only ever refute equivalence, never prove it, so more seeds is
        stronger evidence and never certainty.
        """,
        notes="""
        The PTX checker's own docstring is careful about this and the artifact should be too: it
        is sound only RELATIVE to the result-DAG it reconstructs from the PTX, and `ptxas` can
        still reorder below PTX. It is a cheap static pre-filter, not a proof. The fuzzer is the
        ground truth here, and `checker.regpressure` is the step that pushes on the PTX-to-SASS
        gap directly.
        """,
    )
