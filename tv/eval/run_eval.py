#!/usr/bin/env python
"""
Driver for the tv translation-validator evaluation suite.

Three kinds of evaluation, one per folder:

  pairs/            curated before/after pairs, each tagged EQUIV or NEQ.
                    The validator's verdict must match the tag.   (gate)

  inequal/          pairs that are genuinely NOT equivalent. The validator must
                    report NEQ for all — a soundness check that it catches real
                    differences.                                  (gate)

  compile-options/  kernels compiled under different options. The unoptimized
                    'standard.ttir' is the correct reference; every other
                    variant in the folder must validate EQUIV against it. (gate)

  solver-cost/      runs the queries and records solver/interpret timing to
                    results.csv.                                  (reporting)

Subcommands:
    pairs            run the curated pairs gate
    inequal          run the inequality-detection gate
    compile-options  run the unopt-vs-variant gate
    solver-cost      time the queries, write results.csv
    all              pairs + inequal + compile-options (gates), then solver-cost

Exit code is non-zero if any gating case fails.
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

PAIRS_DIR = HERE / "pairs"
INEQUAL_DIR = HERE / "inequal"
COMPILE_DIR = HERE / "compile-options"
SOLVER_DIR = HERE / "solver-cost"


def _read_manifest(manifest_dir=PAIRS_DIR):
    """Yield (src_path, tgt_path, expected, note) from <manifest_dir>/cases.tsv.

    Paths in the manifest are relative to manifest_dir.
    """
    manifest = manifest_dir / "cases.tsv"
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        cols = [c.strip() for c in cols if c.strip() != ""]
        if len(cols) < 3:
            print(f"  [warn] bad manifest line: {line!r}")
            continue
        src, tgt, expected = cols[0], cols[1], cols[2].upper()
        note = cols[3] if len(cols) > 3 else ""
        yield manifest_dir / src, manifest_dir / tgt, expected, note


def _collect_compile_queries():
    """Yield (kernel, standard_path, variant_path) for every variant."""
    if not COMPILE_DIR.is_dir():
        return
    for kernel_dir in sorted(p for p in COMPILE_DIR.iterdir() if p.is_dir()):
        if kernel_dir.name == "kernels":   # helper scripts, not a kernel folder
            continue
        standard = kernel_dir / "standard.ttir"
        if not standard.is_file():
            print(f"  [warn] {kernel_dir.name}: no standard.ttir, skipping")
            continue
        for variant in sorted(kernel_dir.glob("*.ttir")):
            if variant.name == "standard.ttir":
                continue
            yield kernel_dir.name, standard, variant


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def run_pairs(binary, timeout, manifest_dir=PAIRS_DIR, label="pairs"):
    print(f"== {label} ==")
    passed = failed = 0
    for src, tgt, expected, note in _read_manifest(manifest_dir):
        if not src.is_file() or not tgt.is_file():
            print(f"[FAIL] {src.name} vs {tgt.name}: missing file")
            failed += 1
            continue
        res = common.run_validation(binary, src, tgt, timeout)
        ok = res.verdict == expected
        tag = "PASS" if ok else "FAIL"
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        extra = f" ({note})" if note else ""
        print(f"[{tag}] {src.name} vs {tgt.name}: "
              f"expected {expected}, got {res.verdict}{extra}")
    print(f"-- {label}: {passed} passed, {failed} failed --\n")
    return failed == 0


def run_compile_options(binary, timeout):
    print("== compile-options ==")
    queries = list(_collect_compile_queries())
    if not queries:
        print("  (no kernels generated yet; run compile-options/generate.py)\n")
        return True
    passed = failed = 0
    for kernel, standard, variant in queries:
        res = common.run_validation(binary, standard, variant, timeout)
        ok = res.verdict == "EQUIV"   # a pass must preserve semantics
        tag = "PASS" if ok else "FAIL"
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"[{tag}] {kernel}/{variant.name} vs standard: "
              f"expected EQUIV, got {res.verdict}")
    print(f"-- compile-options: {passed} passed, {failed} failed --\n")
    return failed == 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def run_solver_cost(binary, timeout):
    print("== solver-cost ==")
    SOLVER_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SOLVER_DIR / "results.csv"

    rows = []
    # Every curated pair...
    for src, tgt, expected, _note in _read_manifest():
        if src.is_file() and tgt.is_file():
            rows.append(("pairs", src, tgt))
    # ...plus every compile-options variant-vs-standard query.
    for kernel, standard, variant in _collect_compile_queries():
        rows.append((f"compile-options/{kernel}", standard, variant))

    records = []
    print(f"{'case':<28} {'verdict':<8} {'solver_s':>10}")
    for case, src, tgt in rows:
        res = common.run_validation(binary, src, tgt, timeout)
        records.append({
            "case": case,
            "src": src.name,
            "tgt": tgt.name,
            "fp_mode": "Abstract",   # only mode implemented today
            "verdict": res.verdict,
            "interp_s": res.interp_s,
            "solver_s": res.solver_s,
        })
        print(f"{case:<28} {res.verdict:<8} {res.solver_s:>10.4f}")

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case", "src", "tgt", "fp_mode", "verdict", "interp_s", "solver_s"])
        w.writeheader()
        w.writerows(records)
    print(f"-- wrote {len(records)} rows to {out_csv} --\n")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="tv evaluation suite driver")
    ap.add_argument("mode", choices=["pairs", "inequal", "compile-options",
                                     "solver-cost", "all"])
    ap.add_argument("--timeout", type=int, default=120,
                    help="per-validation timeout in seconds (default 120)")
    args = ap.parse_args()

    binary = common.find_triton_tv()
    print(f"Using triton-tv: {binary}\n")

    ok = True
    if args.mode in ("pairs", "all"):
        ok &= run_pairs(binary, args.timeout)
    if args.mode in ("inequal", "all"):
        ok &= run_pairs(binary, args.timeout, INEQUAL_DIR, "inequal")
    if args.mode in ("compile-options", "all"):
        ok &= run_compile_options(binary, args.timeout)
    if args.mode in ("solver-cost", "all"):
        run_solver_cost(binary, args.timeout)  # reporting only, never gates

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
