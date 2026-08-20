import tilelang
import tilelang.language as T


@T.prim_func
def main(A: T.Tensor((64, 64), "float16"), X: T.Tensor((64,), "float16"), Y: T.Tensor((64,), "float32")):
    with T.Kernel(T.ceildiv(64, 64), threads=32) as bx:
        As = T.alloc_shared((64, 32), "float16")
        Xs = T.alloc_shared((32,), "float16")
        p = T.alloc_fragment((64, 32), "float32")
        acc = T.alloc_fragment((64,), "float32")
        T.fill(acc, 0)
        for ko in T.Pipelined(T.ceildiv(64, 32), num_stages=2):
            T.copy(A[bx * 64, ko * 32], As)
            T.copy(X[ko * 32], Xs)
            for i, j in T.Parallel(64, 32):
                p[i, j] = T.cast(As[i, j], "float32") * T.cast(Xs[j], "float32")
            T.reduce_sum(p, acc, dim=1, clear=False)
        T.copy(acc, Y[bx * 64])
