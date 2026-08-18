"""Minimal repro: TileLang emits invalid CUDA for a vectorized signed-integer
floordiv / floormod inside T.Parallel.

    source /home/youngzt/fuzz-tilelang/env.sh
    python repro_floordiv.py

Expected: nvcc fails with
    error: expression must have bool type (or be convertible to bool)
on a line of the form  `int4 v__N = __M ? __P : __Q;`  where `__M` is a 4-wide
vector of per-lane comparison results, not a scalar bool.
"""
import tilelang
import tilelang.language as T
import tvm
from tilelang.backend.target import determine_target


@T.prim_func
def main(A: T.Tensor((64, 64), "int32"), C: T.Tensor((64, 64), "int32")):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        As = T.alloc_shared((64, 64), "int32")
        Cl = T.alloc_fragment((64, 64), "int32")
        T.copy(A[0, 0], As)
        for i, j in T.Parallel(64, 64):
            Cl[i, j] = As[i, j] // 2
        T.copy(Cl, C[0, 0])


if __name__ == "__main__":
    tgt = tvm.target.Target(determine_target("auto"))
    with tvm.transform.PassContext(opt_level=3, config={}), tgt:
        src = tilelang.lower(main, target=tgt).kernel_source
    print("generated CUDA, the offending line:")
    for n, l in enumerate(src.splitlines(), 1):
        if "?" in l and "int4" in l:
            print(f"  {n}: {l.strip()}")
    print("the condition it uses is a 4-wide vector, built element by element:")
    for l in src.splitlines():
        if l.strip().startswith("__4."):
            print("  " + l.strip())
    print("\nnvcc:")
    try:
        tilelang.compile(main, out_idx=None)
        print("  UNEXPECTED: compiled")
    except Exception as e:
        for l in str(e).splitlines():
            if "error:" in l:
                print("  " + l.strip())
    print("\nsame kernel with the vectorizer off:")
    tilelang.compile(main, out_idx=None, pass_configs={"tirx.disable_vectorize": True})
    print("  compiled OK")
