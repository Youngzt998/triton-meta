"""checker.korder -- which GEMM knobs move the bits, decided by the bits themselves.

PLACEHOLDER. Mirrors the `korder` stage of `bitequiv/evaluation/evaluate.py`. This is the one
checker step whose verdict does NOT come from the checker: it compiles two configurations that
differ in exactly one axis and compares the output bytes. That is the ground truth a checker has
to agree with, and it is what says which axes a checker is allowed to merge across.
"""
from __future__ import annotations

from ._common import print_design

NAME = "checker.korder"
ORDER = 110
DESCRIPTION = "which GEMM knobs move the bits -- tensor-core version, split-K, 2-CTA"
IMPLEMENTED = False

TABLES = {
    "checker.korder": {
        "doc":
        "One row per (kernel, dtype, question). Two configurations matched on every axis "
        "but one, compiled and run on the same input, compared byte for byte. The answer "
        "is a fact about the hardware and the lowering, not about the checker; a checker "
        "may merge across an axis only where the verdict here is BIT-IDENTICAL.",
        "cols": [
            ("kernel", "str", "GEMM kernel spec from bitequiv/evaluation/eval_kernels.py"),
            ("dtype", "str", "element dtype: f16, bf16, f32 or fp8"),
            ("question", "str", "which axis is being varied: mma_version, k_split or num_ctas"),
            ("axis", "str", "the configuration axis, e.g. gemm_block_m, gemm_num_splits, num_ctas"),
            ("value_a", "str", "value on the first arm"),
            ("value_b", "str", "value on the second arm"),
            ("lowered_a", "str", "what the first arm actually lowered to, read from its asm: "
             "v5(tcgen05.mma), v2(mma.sync), wgmma(Hopper) or FMA"),
            ("lowered_b", "str", "the same for the second arm. If the two are equal the row asked "
             "nothing, whatever the configuration said"),
            ("verdict", "str", "BIT-IDENTICAL or DIFFER"),
            ("bit_free", "int", "1 if the axis turned out not to move the bits on this row, so a "
             "checker may merge across it here"),
            ("note", "str", "why the row was skipped, or what qualifies it"),
        ],
    },
}


def run(args, env):
    print_design(
        NAME,
        claim="""
        Some GEMM knobs are free for a checker to merge across and some are not, and which is
        which is decided by the bits rather than by reasoning about the hardware. Concretely: on
        Blackwell the MMAv5 and MMAv2 paths return identical bytes for f16, bf16 and f32 while
        fp8 does not; and splitting the K reduction changes the summation order, so it moves the
        bits and no checker may merge across it.
        """,
        method="""
        `bitequiv/evaluation/evaluate.py --stages korder` already does this and always runs at
        `mid` size. For each question it uses `_pair_on` to find two configurations from the
        kernel's own space that differ in exactly one axis and are matched on every other, then
        compiles and runs both on the same input and compares the raw bytes.

            mma_version   `gemm_block_m` 32 against 64. 32 lowers to MMAv2 `mma.sync`, 64 to
                          MMAv5 `tcgen05`. Read what each arm actually lowered to out of its asm
                          rather than trusting the knob.
            k_split       `gemm_num_splits` 1 against 2 on `gemm_kgroup` and `gemm_splitk`.
            num_ctas      a real two-CTA MMA needs the TLX cluster path, so on a bare matmul this
                          knob is expected to be bit-free. Say that on the row instead of
                          reporting a pass that came from the feature not being exercised.

        Run it per dtype. dtype is a first-class axis for a GEMM here and is never collapsed --
        the fp8 exception is the entire point of the mma_version question.
        """,
        cost="Minutes. It is two compiles and two launches per row, not a sweep.",
        see="Per row: what each arm lowered to, and BIT-IDENTICAL or DIFFER.",
        judge="""
        Read `lowered_a` and `lowered_b` before the verdict. If they are the same string the two
        arms took the same path and BIT-IDENTICAL is trivially true and says nothing. This is the
        most likely way for the row to look like a pass while having asked no question.

        A DIFFER verdict is not a failure. `k_split` is EXPECTED to differ: splitting K changes
        the summation order, which changes the bits. The result would be alarming the other way
        round.

        What one row can conclude. A DIFFER on one shape proves the axis moves the bits -- one
        case is enough for that. BIT-IDENTICAL on one shape does NOT prove the axis is bit-free;
        it is one point in a space, and a wrong recipe can survive a large majority of shapes. So
        `bit_free` on a single row is a hypothesis. Treat it as a fact only after a sweep, and say
        in the report how many shapes it covered.
        """,
        notes="""
        `evaluate.py` also has a `cublas` stage, and it is a stub there on purpose: comparing a
        Triton GEMM against a cuBLAS reference is `gemm.bitmatch` in this artifact, at a far
        larger scale than that stage was ever going to reach. There is deliberately no
        `checker.cublas` step.
        """,
    )
