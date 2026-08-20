import tilelang
import tilelang.language as T


@T.prim_func
def main(A: T.Tensor((511, 1025), "float32"), X: T.Tensor((1025,), "float32"), Y: T.Tensor((511,), "float32")):
    with T.Kernel(T.ceildiv(511, 64), threads=32) as bx:
        As = T.alloc_shared((64, 256), "float32")
        Xs = T.alloc_shared((256,), "float32")
        p = T.alloc_fragment((64, 256), "float32")
        acc = T.alloc_fragment((64,), "float32")
        T.fill(acc, 0)
        for ko in T.Pipelined(T.ceildiv(1025, 256), num_stages=3):
            T.copy(A[bx * 64, ko * 256], As)
            T.copy(X[ko * 256], Xs)
            for i, j in T.Parallel(64, 256):
                p[i, j] = T.cast(As[i, j], "float32") * T.cast(Xs[j], "float32")
            T.reduce_sum(p, acc, dim=1, clear=False)
        T.copy(acc, Y[bx * 64])
