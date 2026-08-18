import sys
import torch
import triton

sys.path.insert(0, "/home/youngzt/fuzz/ttgir-broad")
from fz import compile_utils as cu  # noqa: E402

TGT = cu.parse_target("cuda:90")
k = triton.compile(sys.argv[1], target=TGT)
a = torch.ones(1 << 20, device="cuda", dtype=torch.float16)
b = torch.ones(1 << 20, device="cuda", dtype=torch.float16)
o = torch.full((1 << 20,), float("nan"), device="cuda", dtype=torch.float16)
k[(1, 1, 1)](a, b, o, 64, 512)
torch.cuda.synchronize()
w = (~torch.isnan(o)).nonzero().flatten()
print("written", w.numel(), "max", int(w.max()), "OOB(>=32768)",
      int((w >= 32768).sum()))
