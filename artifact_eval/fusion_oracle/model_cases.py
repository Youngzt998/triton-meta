"""Turn `fusion_moe/models_2026.py`'s 419 real-model cases into cases `three_way.py` can run.

`fusion_moe` measures OUR fused GEMM against cuBLAS plus a separate epilogue kernel. This file
points the same case list at the other question: what does byte-exactness cost against the kernel
`torch.compile` ITSELF produces? So the shape and the epilogue come from `models_2026.py`
unchanged, and only the arms differ.

    python model_cases.py                       # what is reachable, and what is not
    python model_cases.py --out cases_moe.txt   # then: three_way.py --cases-file cases_moe.txt

Reachability
------------
A case is reachable when Inductor folds its epilogue into the mm template, because arm 3 is arm
2's own kernel with the epilogue swapped -- with no arm 2 there is nothing to swap. Every
pointwise epilogue in the table qualifies; `argmax` does not, and `ARGMAX_VERDICT` below says
why, at length, because leaving 56 cases out is a claim that needs its reason recorded next to
the code that leaves them out.

The scalars
-----------
`alpha/rank` for LoRA and the fp8 static scale are the values `fusion_moe/run_moe.py` uses, read
from it rather than restated, so the two runs cannot drift apart. `swiglu_limit` comes from each
model's own config.json via `models_2026.py`.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE, HERE, os.path.join(AE, "fusion_moe")):
    if p not in sys.path:
        sys.path.insert(0, p)

from models_2026 import GROUPS, build  # noqa: E402

# `fusion_moe/run_moe.py` sets these; importing it would drag in torch and bitequiv, so they are
# read out of its source instead of copied, which still fails loudly if it renames them.
_RUN_MOE = os.path.join(AE, "fusion_moe", "run_moe.py")


def _const(name):
    for ln in open(_RUN_MOE):
        if ln.startswith(name + " ="):
            return float(ln.split("=")[1].split("#")[0])
    raise RuntimeError(f"{name} is gone from run_moe.py")


LORA_ALPHA = _const("LORA_ALPHA")
FP8_STATIC_SCALE = _const("FP8_STATIC_SCALE")

# ------------------------------------------------------------------------------------------- #

ARGMAX_VERDICT = """\
argmax is NOT reachable in this experiment, and the reason is structural rather than numeric.

1. There is no arm 2. Inductor's mm template fuses a POINTWISE epilogue into `store_output`; a
   reduction along N is not one. Compiled at three real lm_head shapes with
   `max_autotune_gemm_backends=TRITON`, `torch.compile(lambda a, b: torch.argmax(a @ b, dim=1))`
   gave `triton_tem_fused_mm_0` plus a separate `triton_red_fused_argmax_1` at 8x32000x4096 and
   256x129280x4096, and at 1x201088x2880 it did not even pick the mm template
   (`triton_red_fused_mm_0` plus `triton_red_fused_argmax_mm_1`). Arm 3 is defined as arm 2's own
   kernel with the epilogue text replaced; with no fused arm 2 there is no kernel to patch.

2. Making one anyway would not be the same comparison. A row's argmax needs the whole row, and a
   template program owns one BLOCK_M x BLOCK_N tile, so the N axis is already split across
   programs. A fused version has to write one (value, index) pair per row per column tile and
   merge them in a second kernel -- which is exactly what `fusion_moe/fused_moe.py::fused_argmax`
   does. That changes the output shape, the store, the kernel count and the launch. The whole
   basis of arm 2 versus arm 3 is that ONLY the epilogue text differs, and none of that is
   epilogue text.

3. The bit reference would be a rule we chose, not torch. Over a 128k-wide fp16 vocabulary a tie
   for the maximum is the common case, and nothing documents which index torch returns then. So
   both arms of `fusion_moe` agree to a rule they picked -- largest value, and among equal
   largest values the lowest column index -- and check `torch_argmax_agrees` on the side. On 576
   tie-saturated rows (9 shapes x 64 rows, every row with at least two maxima) torch's CUDA
   argmax did return the lowest index every time, so the rule is a good guess at what torch does;
   576 rows is not a proof that it always does, and torch promises nothing. Matching a rule of
   our own is a different claim from matching torch eager, which is what every other row here
   means.

So the 56 lm_head cases are left out with this reason recorded, rather than filled with a number
measured against something else. `fusion_moe/run_moe.py::one_argmax` is where the fused argmax
question IS asked, against its own reference and with its own tie rule stated."""

UNREACHABLE = {"argmax": "a reduction along N; see ARGMAX_VERDICT"}


def params_of(case):
    """The scalars that get baked into the generated kernel for this case."""
    if case.epi == "lora":
        return {"scale": LORA_ALPHA / case.K}  # K is the LoRA rank
    if case.epi == "fp8q":
        return {"scale": FP8_STATIC_SCALE}
    if case.epi == "swiglu_cw":
        return {"lim": case.lim}
    return {}


def token(case):
    p = params_of(case)
    return ",".join([str(case.M), str(case.N), str(case.K), case.epi] + [f"{k}={v!r}" for k, v in sorted(p.items())])


def reachable_cases(groups=("moe", "lora", "lmhead", "attn")):
    """(reachable, skipped) as lists of `models_2026.Case`."""
    cases = build(list(groups))
    ok = [c for c in cases if c.epi not in UNREACHABLE]
    no = [c for c in cases if c.epi in UNREACHABLE]
    return ok, no


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", default=",".join(GROUPS))
    p.add_argument("--out", default="", help="write the deduplicated case tokens here")
    args = p.parse_args()

    ok, no = reachable_cases(tuple(g.strip() for g in args.groups.split(",")))
    toks = []
    seen = set()
    for c in ok:
        t = token(c)
        if t not in seen:
            seen.add(t)
            toks.append(t)

    print(f"{len(ok) + len(no)} cases in the model table\n")
    print(f"{'epilogue':12s} {'cases':>6} {'distinct shapes':>16}  reachable")
    per = Counter(c.epi for c in ok + no)
    distinct = Counter()
    for c in ok:
        distinct[c.epi] += 0
    for t in toks:
        distinct[t.split(",")[3]] += 1
    for epi in sorted(per, key=lambda e: -per[e]):
        why = UNREACHABLE.get(epi)
        print(f"{epi:12s} {per[epi]:>6} {distinct.get(epi, 0):>16}  {'no -- ' + why if why else 'yes'}")
    print(f"\n{len(ok)} of {len(ok) + len(no)} reachable, {len(toks)} distinct (shape, epilogue, scalars) "
          f"to compile and measure")
    if no:
        print("\n" + ARGMAX_VERDICT)
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(toks) + "\n")
        print(f"\nwrote {len(toks)} case tokens to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
