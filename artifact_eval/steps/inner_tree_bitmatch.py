"""inner_tree.bitmatch -- when the reduction order is pinned, do the bits actually stop moving?

PLACEHOLDER.

Attribution. The ordering mechanism this step evaluates is not ours. `reduction_ordering` /
`inner_tree` and the `TRITON_STRICT_REDUCTION_ORDERING` environment variable
were written by Nick Riasanovsky, a co-author. What is ours is the measurement:
asking whether the guarantee holds bit for bit across the configurations an autotuner would
actually try.
"""
from __future__ import annotations

from ._common import print_design

NAME = "inner_tree.bitmatch"
ORDER = 60
DESCRIPTION = "does the enforced reduction order hold, bit for bit"
IMPLEMENTED = False

TABLES = {
    "inner_tree.bitmatch": {
        "doc":
        "One row per (kernel, dtype, ordering). Sweep the configuration axes that change "
        "the reduction layout and group the configurations by the bytes they return. "
        "Under `unordered` the group count is the number of distinct reduction orders the "
        "layout produced; under `inner_tree` it should be 1, because the order is pinned "
        "and the layout is no longer allowed to decide it.",
        "cols": [
            ("kernel", "str", "kernel from the bitequiv evaluation registry, e.g. sum, dot, softmax"),
            ("dtype", "str", "element dtype: f16, bf16, f32 or fp8"),
            ("ordering", "str", "reduction_ordering under test: unordered or inner_tree"),
            ("strict_env", "int", "1 if TRITON_STRICT_REDUCTION_ORDERING was set before triton was imported"),
            ("axes", "str", "space-separated configuration axes swept, e.g. num_warps num_stages block_n"),
            ("n_configs", "int", "configurations in the sweep"),
            ("n_ran", "int", "of those, how many compiled and launched"),
            ("n_failed", "int", "configurations that failed to compile or launch"),
            ("seeds", "int", "random input draws each configuration was run on"),
            ("n_bit_classes", "int", "distinct byte outputs across the configurations that ran"),
            ("largest_class", "int", "size of the biggest group of configurations returning the same bytes"),
            ("invariant", "int", "1 if n_bit_classes is 1, i.e. every configuration agreed"),
            ("split_axes", "str", "the axes whose value changes between two configurations that "
             "disagreed; empty when invariant. This is the finding, not n_bit_classes on its own"),
            ("error", "str", "non-empty if the row failed to produce a result"),
        ],
    },
}


def run(args, env):
    print_design(
        NAME,
        claim="""
        With `reduction_ordering=inner_tree` the reduction order is fixed by the request, not by
        the layout, so configurations that change the layout -- num_warps above all -- return
        byte-identical output. With `unordered` they do not. That is what makes the ordering
        switch a real guarantee rather than a hint.
        """,
        method="""
        Do not build a new harness. `bitequiv/evaluation/` already does this: `eval_kernels.py`
        is the kernel registry and declares each kernel's configuration space,
        `equivalence_fuzzer.py` is the empirical oracle (run a configuration on N random seeds,
        key it by the tuple of output digests, group), and `evaluate.py --stages precision`
        already compiles a space, fuzzes it and partitions it. This step calls that machinery and
        writes the partition sizes into a table; it does not reimplement the fuzzer.

        Per kernel and dtype, run the sweep twice: once with `reduction_ordering=unordered` and
        once with `inner_tree`, holding every other axis the same. Sweep the axes that move the
        layout -- num_warps first, then num_stages and the block size. Count the empirical bit
        classes each time.

        GOTCHA. `TRITON_STRICT_REDUCTION_ORDERING` is read when triton is imported, so it has to
        be set in the environment before the process starts. Setting it from Python after
        `import triton` silently does nothing and the run looks like a clean pass. Record whether
        it was set on every row (`strict_env`) so a row cannot be read without knowing.

        Cover the non-commutative case on purpose. A commutative combine (sum, max) can come out
        invariant for reasons that have nothing to do with the ordering switch; a custom
        non-commutative `combine_fn` is where the guarantee is actually load-bearing.
        """,
        cost="Minutes per (kernel, dtype, ordering) at `light` effort; the full space is longer. "
        "It compiles and launches every configuration in the sweep.",
        see="Per row: the configuration count, the number of distinct byte outputs, and the "
        "biggest agreeing group. Side by side, `unordered` should show several classes where "
        "`inner_tree` shows one.",
        judge="""
        `invariant` = 1 on every `inner_tree` row is the claim. One `inner_tree` row with
        `invariant` = 0 refutes it, and `split_axes` names the axis that broke it -- that is the
        useful output, not the count.

        The `unordered` rows are the control and they matter as much. If `unordered` is also
        invariant on some kernel, then that kernel's bits never depended on the layout in the
        first place and it proves nothing about the ordering switch. A sweep made only of such
        kernels would report a clean pass while testing nothing.

        Note what this can and cannot conclude. A pass over these kernels and these axes says
        "on this sweep, no configuration disagreed". It does not say the guarantee holds
        everywhere. Read `axes` and `n_configs` before generalising.
        """,
        notes="""
        Prior finding, narrow on purpose. An earlier audit on NVIDIA covered roughly 150
        configurations and found `inner_tree` bit-invariant for every commutative reduction it
        tried. The corner it did not reach was a non-commutative custom `combine_fn`. So the
        commutative side is a re-run for the artifact; the non-commutative side is the part that
        is genuinely open.
        """,
    )
