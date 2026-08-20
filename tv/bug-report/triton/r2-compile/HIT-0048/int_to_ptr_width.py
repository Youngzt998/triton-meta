# Minimal hand-written repro for r2-compile/HIT-0048.
# `tl.int_to_ptr` (built by Semantic.cast, semantic.py:897-899) requires an i64
# operand (TT_I64Like, TritonOps.td:46). The frontend does not widen or reject a
# narrower integer, so it emits invalid IR.
import sys, triton, triton.language as tl

@triton.jit
def k(handles, out):
    p = tl.load(handles + tl.program_id(0)).to(tl.pointer_type(tl.float32))
    tl.store(out + tl.program_id(0), tl.load(p))

for ity in ("*i64", "*i32", "*i8", "*u8", "*i16"):
    src = triton.compiler.ASTSource(fn=k, signature={"handles": ity, "out": "*fp32"}, constexprs={})
    try:
        triton.compile(src, target=triton.backends.compiler.GPUTarget("cuda", 80, 32))
        print(f"{ity:6s} -> OK")
    except Exception as e:
        msg = str(e).strip().splitlines()
        print(f"{ity:6s} -> {type(e).__name__}: {msg[0] if msg else ''}")
