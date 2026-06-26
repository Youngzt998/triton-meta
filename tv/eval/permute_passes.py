#!/usr/bin/env python
"""
TTIR optimization-permutation campaign for the tv translation validator.

Idea
----
Take an UNOPTIMIZED kernel TTIR (a realistic block size, as a real GPU program
would use) and apply every ordered permutation of 1, 2, and 3 optimization
passes from a curated TTIR pass set. Each pass must preserve semantics, so every
optimized result MUST validate EQUIVALENT against the unoptimized reference.

Two goals:
  1. Stress the validator on realistic-size IR under many pass orders.
  2. Hunt for a real Triton miscompile: a NOT-EQUIVALENT result means some pass
     (combination) changed observable behavior — a potential compiler bug.

The campaign keeps validating until it finds an inequality (NEQ) or exhausts all
permutations. A handful (~`--max-keep`) of *distinct* optimized variants are
written next to the standard with a comment header naming the passes that were
turned on, so the routine `compile-options` gate (run_eval.py) covers them.

Usage:
    python tv/eval/permute_passes.py            # full campaign, keep ~10 variants
    python tv/eval/permute_passes.py --max-keep 0   # hunt only, write nothing
"""

import argparse
import hashlib
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

# Curated TTIR-level passes that are valid for an elementwise kernel (no loops,
# no dot, no block pointers). These are real passes from Triton's TTIR pipeline
# plus standard MLIR cleanups; each is semantics-preserving by contract.
PASSES = [
    "--canonicalize",
    "--cse",
    "--triton-combine",
    "--triton-reorder-broadcast",
    "--sccp",
    "--symbol-dce",
    "--loop-invariant-code-motion",
]


def run_opt(topt, baseline, passes):
    """Run triton-opt with `passes`; return optimized IR text or None on error."""
    proc = subprocess.run(
        [str(topt), str(baseline), *passes],
        capture_output=True, text=True,
    )
    return (proc.stdout if proc.returncode == 0 else None), proc.stderr


def strip_comments(ir):
    """Normalize IR for dedup: drop // comment lines and blank lines."""
    return "\n".join(
        ln for ln in ir.splitlines()
        if ln.strip() and not ln.lstrip().startswith("//")
    )


def header_for(combo):
    return (
        "// add_kernel TTIR — optimizations turned ON over the unoptimized standard:\n"
        f"//   triton-opt standard.ttir {' '.join(combo)}\n"
        "// Expected vs standard.ttir: EQUIVALENT (each pass must preserve semantics).\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline",
                    default=str(HERE / "compile-options/add_kernel/standard.ttir"),
                    help="unoptimized reference TTIR")
    ap.add_argument("--out", default=str(HERE / "compile-options/add_kernel"),
                    help="folder to write kept distinct variants (gated by run_eval)")
    ap.add_argument("--max-keep", type=int, default=10,
                    help="how many distinct EQUIV variants to keep on disk")
    ap.add_argument("--max-len", type=int, default=3,
                    help="max number of passes per permutation")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    topt = common.find_triton_opt()
    tv = common.find_triton_tv()
    baseline = Path(args.baseline)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not baseline.is_file():
        sys.exit(f"baseline not found: {baseline}")

    print(f"triton-opt: {topt}")
    print(f"triton-tv : {tv}")
    print(f"baseline  : {baseline}  "
          f"(size {next((l for l in baseline.read_text().splitlines() if 'tensor<' in l), '?')!r})")
    print(f"passes    : {[p.lstrip('-') for p in PASSES]}\n")

    # Seed the dedup set with outputs already on disk (standard + existing variants)
    # so kept files are genuinely new transformations.
    seen = set()
    for f in sorted(out.glob("*.ttir")):
        seen.add(hashlib.sha1(strip_comments(f.read_text()).encode()).hexdigest())

    tested = equiv = neq = err = kept = 0
    bug = None

    for k in range(1, args.max_len + 1):
        for combo in itertools.permutations(PASSES, k):
            ir, stderr = run_opt(topt, baseline, list(combo))
            tested += 1
            label = " ".join(p.lstrip("-") for p in combo)
            if ir is None:
                err += 1
                print(f"[OPT-ERR] ({k}) {label}: {stderr.strip().splitlines()[-1] if stderr.strip() else 'triton-opt failed'}")
                continue

            with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=True) as tf:
                tf.write(ir)
                tf.flush()
                res = common.run_validation(tv, baseline, tf.name, args.timeout)

            if res.verdict == "EQUIV":
                equiv += 1
            elif res.verdict == "NEQ":
                neq += 1
            else:
                err += 1
            print(f"[{res.verdict:<7}] ({k}) {label}  solver={res.solver_s:.3f}s")

            if res.verdict == "NEQ":
                # Potential Triton miscompile — save it and stop.
                bugfile = out / "BUG_neq.ttir"
                bugfile.write_text(header_for(combo) + ir)
                bug = (combo, res, bugfile)
                break

            # Keep a few distinct semantics-preserving variants for the gate.
            h = hashlib.sha1(strip_comments(ir).encode()).hexdigest()
            if (res.verdict == "EQUIV" and h not in seen
                    and kept < args.max_keep):
                seen.add(h)
                name = f"perm{kept:02d}_" + "_".join(
                    p.lstrip("-").replace("triton-", "")[:8] for p in combo) + ".ttir"
                (out / name).write_text(header_for(combo) + ir)
                kept += 1
                print(f"          kept -> {name}")
        if bug:
            break

    print("\n==== campaign summary ====")
    print(f"permutations tested : {tested}")
    print(f"  EQUIVALENT        : {equiv}")
    print(f"  NOT EQUIVALENT    : {neq}")
    print(f"  errors/unknown    : {err}")
    print(f"distinct variants kept: {kept}  (in {out})")

    if bug:
        combo, res, bugfile = bug
        print("\n!!!! POTENTIAL TRITON BUG — pass order produced a NON-equivalent program:")
        print(f"     passes : {' '.join(combo)}")
        print(f"     saved  : {bugfile}")
        print("     (re-check soundness: confirm the validator is right before filing.)")
        sys.exit(1)

    print("\nNo inequality found — all permutations validate EQUIVALENT.")
    sys.exit(0)


if __name__ == "__main__":
    main()
