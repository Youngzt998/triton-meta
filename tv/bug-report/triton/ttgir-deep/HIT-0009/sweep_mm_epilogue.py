import sys, itertools
from pathlib import Path
import torch
sys.path.insert(0, "/home/youngzt/fuzz/ttgir-deep")
from harness import runner as R
from harness.kernels import all_cases
R.ensure_allocator()
cases = [c for c in all_cases() if c.family == "mm_epilogue"]
print(f"{'cfg':58s} {'dist':10s} {'ndiff':>8s} {'maxabs':>10s}")
rows = 0
bad = 0
for c in cases:
    ctx = R.CaseCtx(c, "cuda:90", Path("/tmp/sw"))
    try:
        ref = ctx.build(ctx.ttgir(["hopper-warpspec"]), "ref")
        cand = ctx.build(ctx.ttgir([]), "cand")
    except Exception as e:
        print(f"{c.key:58s} build error {e}"[:110]); continue
    ws = ctx.ttgir([]).count("ttg.warp_specialize")
    for dist in ("randn", "intvalued", "zeros", "wide"):
        g1 = torch.Generator(device="cuda").manual_seed(11)
        g2 = torch.Generator(device="cuda").manual_seed(11)
        a, b = c.make_inputs(g1, dist), c.make_inputs(g2, dist)
        ref.run(a, c.grid); cand.run(b, c.grid); torch.cuda.synchronize()
        d = R.first_diff(a["c_ptr"], b["c_ptr"])
        rows += 1
        if d:
            bad += 1
        print(f"{c.key:58s} {dist:10s} {d.get('count',0):8d} {d.get('max_abs_diff',0.0):10.4g}"
              f"  ws_in_ir={ws}")
print(f"\n{bad} of {rows} (config, distribution) pairs differ")
