"""checker.regpressure -- does the checker's verdict survive ptxas?

PLACEHOLDER. Mirrors the `regpressure` stage of `bitequiv/evaluation/evaluate.py`. The checker
reads PTX, but PTX is not what runs; ptxas turns it into SASS and can allocate registers, spill
and reorder underneath. This step is the one that pushes on that gap on purpose.
"""
from __future__ import annotations

from ._common import print_design
from .checker_precision import CHECKER_COLS

NAME = "checker.regpressure"
ORDER = 100
DESCRIPTION = "does the checker's verdict survive ptxas and register spilling"
IMPLEMENTED = False

TABLES = {
    "checker.regpressure": {
        "doc":
        "One row per (checker, artifact, kernel). Take the full configuration space and "
        "compile every configuration again under several `maxnreg` caps low enough to "
        "make ptxas spill. A `.maxnreg` directive leaves the PTX body identical, so the "
        "checker puts the capped and uncapped builds in the SAME group; the fuzzer then "
        "says whether they really do return the same bytes. One checker group still "
        "equal to one bit group means the verdict held across the PTX-to-SASS step.",
        "cols":
        CHECKER_COLS + [
            ("caps", "str", "the maxnreg caps applied, including the uncapped baseline"),
            ("n_configs", "int", "base configurations before the caps multiply them"),
            ("attempted", "int", "base configurations times caps"),
            ("ok", "int", "of those, how many compiled and launched"),
            ("fails", "int", "members that failed to compile or launch"),
            ("seeds", "int", "random input draws per member in the fuzzer"),
            ("checker_n", "int", "groups the checker produced"),
            ("checker_max", "int", "largest checker group"),
            ("empirical_n", "int", "groups the fuzzer produced"),
            ("empirical_max", "int", "largest empirical group"),
            ("over_merges", "int", "pairs the checker called equal that the fuzzer separated. MUST be 0"),
            ("refines", "int", "1 if every checker group sits inside one empirical group"),
            ("n_spilled", "int", "members whose ptxas report shows a non-zero spill. If this is 0 "
             "the caps were not tight enough and the step tested nothing"),
            ("reg_min", "int", "lowest register count across all members"),
            ("reg_max", "int", "highest register count across all members"),
            ("spill_min", "int", "lowest spill count across all members"),
            ("spill_max", "int", "highest spill count across all members"),
            ("largest_maxnreg", "str", "the caps present inside the largest checker group"),
            ("largest_spilled", "int", "members of the largest checker group that spilled"),
            ("largest_spans", "str", "configuration axes that vary inside the largest checker group"),
            ("error", "str", "non-empty if the row failed to produce a result"),
        ],
    },
}


def run(args, env):
    print_design(
        NAME,
        claim="""
        The checker's equivalence verdict is not an artefact of reading PTX. Configurations it
        certifies as identical stay byte-identical after ptxas has allocated registers and
        spilled, so the verdict survives the one step the checker cannot see.
        """,
        method="""
        `bitequiv/evaluation/evaluate.py --stages regpressure` already does this, and it always
        runs at `mid` size. Take the full diverse configuration space -- num_warps, num_stages,
        `enable_fp_fusion`, block size -- and compile every configuration again under one or more
        `maxnreg` caps (`--maxnreg-sweep`, default 16), plus an uncapped baseline.

        The trick is that a `.maxnreg` directive does not change the PTX reduction body at all,
        so the checker's descriptor is unchanged and the capped and uncapped builds land in the
        same group by construction. Only ptxas behaves differently. Fuzzing every
        `(configuration, cap)` member then asks the question directly: is one checker group still
        one bit group once ptxas has had its say?
        """,
        cost="Each cap multiplies the member count, so keep the sweep to one or two caps. Longer "
        "than `checker.precision` on the same kernels.",
        see="Per row: the group counts, the over-merge count, how many members actually spilled, "
        "and the register and spill ranges the row covered.",
        judge="""
        `over_merges` = 0 is the gate, same as `checker.precision`.

        Read `n_spilled` before believing a clean result. If no member spilled, the caps were too
        loose and the step exercised nothing; a pass then says only that the caps did not bite.
        `reg_range` and `spill_range` say how wide a register regime the row really covered.

        `largest_maxnreg` should contain more than one value. If the largest checker group is all
        one cap, the checker never merged across register regimes and the PTX-to-SASS question
        was not put to it.

        What a clean run can conclude, precisely. "Over this configuration space and these caps,
        on this GPU and this ptxas, no certified pair diverged." The PTX-to-SASS gap is a
        residual that this narrows with evidence; it does not close it. Reading SASS directly
        would be the stronger check.
        """,
    )
