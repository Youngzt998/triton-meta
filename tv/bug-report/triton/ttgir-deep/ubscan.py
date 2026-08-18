"""Check every kernel in this line's corpus for the write-write race class.

Background (from line ttgir-broad): a kernel that stores to an address computed
from *loaded* data has a write-write race whenever two lanes compute the same
address.  The determinism pre-screen cannot see it -- one binary has a fixed
write order, so three runs agree -- but a different binary may order the writes
differently, so the race shows up as a cross-compilation bit difference and
looks exactly like a miscompile.  A data race is undefined behaviour, so those
are not findings.

This scan is stricter than `ttgir-broad/fz/irscan.py` in two ways:

* that one walks back from the FIRST ssa operand of the store only.  For
  `tt.descriptor_store %desc[%x, %y], %val` the first operand is the
  descriptor, so a coordinate computed from loaded data would be missed.  This
  one walks the descriptor *and* every coordinate.
* it also covers the lowered TMA store `ttng.async_tma_copy_local_to_global`
  and the TMA gather/scatter ops, which only appear after TTGIR lowering.

Both scans are run and both results are printed, so the two can be compared.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, "/home/youngzt/fuzz/ttgir-deep")

_DEF = re.compile(r"^\s*(%[0-9a-zA-Z_]+)\s*(?::\d+)?\s*=\s*"
                  r"([a-z_]+(?:\.[a-z_0-9]+)+)(.*)$")
_USE = re.compile(r"%[0-9a-zA-Z_]+")

# every op that writes to global memory, and which of its operands are the
# address side.  "all-before-value" means: the descriptor plus the coordinates.
STORE_OPS = {
    "tt.store": "first",
    "tt.descriptor_store": "desc+coords",
    "tt.descriptor_scatter": "desc+coords",
    "ttng.async_tma_copy_local_to_global": "desc+coords",
    "ttng.async_tma_scatter": "desc+coords",
    "tt.atomic_rmw": "first",
    "tt.atomic_cas": "first",
}
LOAD_OPS = ("tt.load", "tt.descriptor_load", "tt.descriptor_gather",
            "ttng.async_tma_copy_global_to_local", "ttng.async_tma_gather",
            "tt.atomic_rmw", "tt.atomic_cas", "ttg.local_load")

_STORE_LINE = re.compile(
    r"^\s*(?:%[0-9a-zA-Z_:, ]+=\s*)?(" + "|".join(
        re.escape(k) for k in STORE_OPS) + r")\b(.*)$")


def build_defs(ir_text: str) -> Dict[str, Tuple[str, List[str]]]:
    out: Dict[str, Tuple[str, List[str]]] = {}
    for ln in ir_text.splitlines():
        m = _DEF.match(ln)
        if not m:
            continue
        rhs = m.group(3).split(" : ")[0]
        out[m.group(1)] = (m.group(2), _USE.findall(rhs))
    return out


def address_operands(op: str, rest: str) -> List[str]:
    """The SSA values that decide WHERE the store lands."""
    rest = rest.split(" : ")[0]
    if STORE_OPS[op] == "first":
        vs = _USE.findall(rest)
        return vs[:1]
    # descriptor form:  %desc[%c0, %c1] %src   /   %desc[%c0, %c1], %val
    m = re.match(r"\s*(%[0-9a-zA-Z_]+)\s*\[([^\]]*)\]", rest)
    if not m:
        return _USE.findall(rest)[:1]
    return [m.group(1)] + _USE.findall(m.group(2))


def scan_strict(ir_text: str) -> Dict[str, object]:
    defs = build_defs(ir_text)
    stores = 0
    scatter: List[str] = []
    atomics: List[str] = []
    for ln in ir_text.splitlines():
        m = _STORE_LINE.match(ln)
        if not m:
            continue
        op, rest = m.group(1), m.group(2)
        stores += 1
        if op.startswith("tt.atomic"):
            atomics.append(ln.strip()[:90])
        work = list(address_operands(op, rest))
        seen: Set[str] = set()
        while work:
            v = work.pop()
            if v in seen or v not in defs:
                continue
            seen.add(v)
            o, ops = defs[v]
            if o in LOAD_OPS:
                scatter.append(f"{op}: address depends on {o} ({v})")
                work = []
                break
            work.extend(ops)
    return {"n_store": stores, "scatter": scatter, "atomics": atomics,
            "ub": bool(scatter or atomics)}


def main() -> int:
    import torch
    from harness import pipeline as P
    from harness import runner as R
    from harness.kernels import all_cases

    try:
        sys.path.insert(0, "/home/youngzt/fuzz/ttgir-broad")
        from fz import irscan as bscan          # line B's scan, read-only
    except Exception as e:                      # it may move; do not depend on it
        bscan = None
        print(f"(line B scan not importable: {e})")

    tgt = P.parse_target("cuda:90")
    cases = all_cases()
    per_family: Dict[str, List] = {}
    for c in cases:
        per_family.setdefault(c.family, []).append(c)

    print(f"corpus: {len(cases)} cases, {len(per_family)} families\n")
    bad = []

    # 1. TTIR for EVERY case (cheap, CPU only) -- provenance of a store address
    #    cannot change between configs, but check them all rather than assume.
    print("== pass 1: TTIR of all %d cases ==" % len(cases))
    n_ub = 0
    for c in cases:
        try:
            ttir = P.make_ttir_text(c.fn, c.signature, c.constexprs, tgt, c.opts)
        except Exception as e:
            print(f"  {c.key}: ttir error {type(e).__name__}")
            continue
        s = scan_strict(ttir)
        if s["ub"]:
            n_ub += 1
            bad.append((c.key, "ttir", s))
    print(f"  cases whose TTIR has a data-dependent store address or an atomic:"
          f" {n_ub}\n")

    # 2. full-pipeline TTGIR for one representative per family (this is the
    #    program that actually runs, with TMA stores lowered)
    print("== pass 2: full-pipeline TTGIR, one case per family ==")
    R.ensure_allocator()
    rows = []
    for fam, lst in sorted(per_family.items()):
        c = lst[0]
        ctx = R.CaseCtx(c, "cuda:90", Path("/tmp/ubscan_work"))
        ttgir = ctx.ttgir([])
        s = scan_strict(ttgir)
        b = bscan.scan(ttgir) if bscan else {}
        rows.append((fam, c.key, s, b))
        if s["ub"]:
            bad.append((c.key, "ttgir", s))
        print(f"  {fam:16s} stores={s['n_store']:3d}  "
              f"scatter={'YES ' + str(s['scatter']) if s['scatter'] else 'no'}  "
              f"atomics={'YES' if s['atomics'] else 'no'}"
              + (f"   [line-B scan: scatter={b.get('scatter_store')}, "
                 f"atomic={b.get('has_atomic')}]" if b else ""))

    print()
    if bad:
        print("UB-RACE CANDIDATES FOUND:")
        for k, lvl, s in bad:
            print(f"  {k} ({lvl}): {s['scatter']} {s['atomics']}")
        verdict = "DIRTY"
    else:
        print("No store address in this corpus is derived from loaded data, and "
              "no kernel uses an atomic.")
        verdict = "CLEAN"
    print(f"\nVERDICT: {verdict}")
    Path("/home/youngzt/fuzz/ttgir-deep/artifacts/ubscan.json").write_text(
        json.dumps({"verdict": verdict, "n_cases": len(cases),
                    "families": {f: {"n_store": s["n_store"],
                                     "scatter": s["scatter"],
                                     "atomics": s["atomics"],
                                     "line_b": b}
                                 for f, _k, s, b in rows},
                    "bad": [(k, l, s) for k, l, s in bad]}, indent=2, default=str))
    return 0 if verdict == "CLEAN" else 1


if __name__ == "__main__":
    sys.exit(main())
