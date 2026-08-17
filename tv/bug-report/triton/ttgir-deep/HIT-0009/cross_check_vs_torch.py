import sys, json
from pathlib import Path
import torch
sys.path.insert(0, "/home/youngzt/fuzz/ttgir-deep")
from harness import runner as R
from harness.kernels import all_cases
torch.backends.cuda.matmul.allow_tf32 = False
R.ensure_allocator()
cfg = dict(dt="bf16", M=512, N=512, K=256, BM=128, BN=128, BK=64, NS=2, WS=1, W=4)
c = next(x for x in all_cases() if x.family == "mm_epilogue"
         and all(str(x.cfg.get(k)) == str(v) for k, v in cfg.items()))
ctx = R.CaseCtx(c, "cuda:90", Path("/tmp/xchk"))
ref = ctx.build(ctx.ttgir(["hopper-warpspec"]), "ref")
cand = ctx.build(ctx.ttgir([]), "cand")
outs = {}
for name, v in (("ref", ref), ("cand", cand)):
    g = torch.Generator(device="cuda").manual_seed(1262280619)
    inp = c.make_inputs(g, "intvalued")
    v.run(inp, c.grid); torch.cuda.synchronize()
    outs[name] = inp["c_ptr"].clone()
    src = inp
g = torch.Generator(device="cuda").manual_seed(1262280619)
inp = c.make_inputs(g, "intvalued")
A, B, D = inp["a_ptr"], inp["b_ptr"], inp["d_ptr"]
acc = (A.float() @ B.float().T)                    # exact: integers, |sum| <= 16384
print("acc integral?", bool((acc == acc.round()).all().item()), "max|acc|", acc.abs().max().item())
x = acc + D[:, None]
x = x * torch.sigmoid(x)
# the kernel takes the max over the 128-wide tile, not the whole row
BM, BN = 128, 128
xr = x.view(512 // BM, BM, 512 // BN, BN).permute(0, 2, 1, 3)   # [tile_m, tile_n, BM, BN]
rmax = xr.amax(dim=3, keepdim=True)
y = (xr - rmax).permute(0, 2, 1, 3).reshape(512, 512).to(torch.bfloat16)
for name in ("ref", "cand"):
    o = outs[name]
    same = torch.equal(o.view(torch.int16), y.view(torch.int16))
    d = (o.float() - y.float()).abs()
    print(f"{name:5s} vs torch fp32 model: bitequal={same}  ndiff={(o.view(torch.int16)!=y.view(torch.int16)).sum().item()}  maxabs={d.max().item()}")
print("ref vs cand ndiff:", (outs['ref'].view(torch.int16)!=outs['cand'].view(torch.int16)).sum().item())
