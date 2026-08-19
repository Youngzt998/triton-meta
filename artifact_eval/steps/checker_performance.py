"""checker.performance -- what does an autotuner give up when it is only allowed to pick inside
one checker-certified set?

PLACEHOLDER. Mirrors the `performance` stage of `bitequiv/evaluation/evaluate.py`.
"""
from __future__ import annotations

from ._common import print_design
from .checker_precision import CHECKER_COLS

NAME = "checker.performance"
ORDER = 90
DESCRIPTION = "the speed an autotuner gives up to stay inside one checker set"
IMPLEMENTED = False

TABLES = {
    "checker.performance": {
        "doc":
        "One row per (checker, artifact, kernel). Benchmark every configuration, take the "
        "global fastest as the ceiling an ordinary equivalence-blind autotuner would "
        "reach, then look inside the largest checker-certified set: fastest against "
        "slowest member is the freedom the checker hands the autotuner, and best member "
        "against the ceiling is what demanding identical bits cost.",
        "cols":
        CHECKER_COLS + [
            ("effort", "str", "light, mid or heavy: configuration subset and input size"),
            ("ok", "int", "configurations that compiled, launched and were timed"),
            ("fails", "int", "configurations that failed"),
            ("size", "str", "input size the row was benchmarked at"),
            ("ceiling_ms", "float", "fastest configuration anywhere in the space, no numerics "
             "requirement. This is the equivalence-blind autotuner"),
            ("ceiling_cfg", "str", "the configuration that won it"),
            ("slowest_ms", "float", "slowest configuration in the space, for scale"),
            ("checker_set_size", "int", "members of the largest checker-certified set"),
            ("checker_byte_identical", "int", "1 if every member of that set really did return "
             "the same bytes. If 0 the rest of the row is meaningless"),
            ("checker_fast_ms", "float", "fastest member of that set"),
            ("checker_slow_ms", "float", "slowest member of that set"),
            ("tuning_freedom", "float", "checker_slow_ms / checker_fast_ms; how much there was to "
             "gain by tuning inside the set at all"),
            ("cost_of_bits", "float", "checker_fast_ms / ceiling_ms; above 1 is what the bit "
             "requirement cost against the unconstrained autotuner"),
            ("empirical_set_size", "int", "members of the largest set the fuzzer merged; the ceiling "
             "on how big a certified set could be"),
            ("empirical_fast_ms", "float", "fastest member of that empirical set"),
            ("empirical_slow_ms", "float", "slowest member of that empirical set"),
            ("error", "str", "non-empty if the row failed to produce a result"),
        ],
    },
}


def run(args, env):
    print_design(
        NAME,
        claim="""
        Restricting an autotuner to one checker-certified set still leaves it real choices, and
        the fastest configuration it can then pick is close to the fastest configuration
        available with no numerics requirement at all. In short: identical bits are cheap, and
        the checker is not so conservative that it tunes away the tuning.
        """,
        method="""
        `bitequiv/evaluation/evaluate.py --stages performance` already does this. It benchmarks
        every configuration in the space, takes the global minimum as the ceiling, then takes the
        largest checker-certified set and -- after verifying every member really is byte-identical
        -- reports fastest against slowest inside it, and best-in-set against the ceiling.

        Two ratios come out and they must not be confused. `tuning_freedom` is how much the
        autotuner can still win inside the set. `cost_of_bits` is what the requirement cost
        against the unconstrained ceiling. Only the second is the price of the constraint.

        Run it for both checkers, exactly as in `checker.precision`, and record the effort: the
        input size changes with it and the times are not comparable across efforts.
        """,
        cost="Hours over the whole registry at `mid`; it times every configuration.",
        see="Per row: the ceiling, the certified set size, the fastest and slowest member, and "
        "the two ratios.",
        judge="""
        Check `checker_byte_identical` = 1 first. It is the row's own self-check: if the members
        of a certified set did not in fact return the same bytes, the set is not certified and
        nothing else on the row means anything.

        `cost_of_bits` is the claim. `tuning_freedom` says whether the claim is interesting: a
        certified set of one configuration has a cost of exactly 1 and means nothing, because
        there was no choice to make.

        `checker_set_size` against `empirical_set_size` is the honest measure of how much of the
        available freedom the checker actually handed over.
        """,
        notes="""
        This step and `gemm.perf.*` measure different things and the names invite mixing them up.
        `gemm.perf.*` asks what it costs to match cuBLAS's bytes on a GEMM. This asks what it
        costs to stay inside one equivalence class of a reduction kernel, which is a constraint
        of a different kind on a different search space.
        """,
    )
