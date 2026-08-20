"""Stock pipeline only, all-zero inputs: a gemv/rr output must be all zero.

Any nonzero (or NaN) output is the compiler leaking uninitialised registers.
Sweeps the thread count, which is what decides whether the warp-specialised
consumer's fragment index can be folded to a constant.
"""
import itertools, json, sys
import torch
from tlfz import core, gen, steps as S

S.install()
fam = sys.argv[1] if len(sys.argv) > 1 else "gemv"
rows = []
for th, bM, bK in itertools.product([32, 64, 128, 256], [64, 128], [64, 256]):
    spec = {"fam": fam, "p": {"M": 511, "K": 1025, "bM": bM, "bK": bK, "th": th,
                              "st": 3, "dt": "float32"}, "pc": {}}
    try:
        k = core.load_item({"kind": "gen", "spec": spec, "kid": gen.spec_id(spec)})
        kern = core.compile_variant(k, {})
        src = kern.get_kernel_source()
        args = core.make_inputs(k, "zeros", 7)
        core.run(kern, args)
        bad = 0
        for a in args:
            if torch.is_tensor(a) and a.is_floating_point():
                bad += int((a != 0).sum()) + int(torch.isnan(a).sum())
        ws = "__launch_bounds__(%d" % 0
        ws = "yes" if ("tma_load" in src or "mbarrier" in src) else "no"
        rows.append((th, bM, bK, ws, bad))
        print(f"th={th:4d} bM={bM:4d} bK={bK:4d}  warp-spec={ws:3s}  nonzero-or-nan outputs = {bad}")
    except Exception as e:
        print(f"th={th:4d} bM={bM:4d} bK={bK:4d}  SKIP {type(e).__name__}: {str(e)[:70]}")
print("bad points:", [r for r in rows if r[4]])
