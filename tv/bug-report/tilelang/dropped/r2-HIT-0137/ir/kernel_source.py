import tilelang
import tilelang.language as T


@T.prim_func
def main(A: T.Tensor((2037, 224), "bfloat16"), X: T.Tensor((224,), "bfloat16"), Y: T.Tensor((2037,), "float32")):
    with T.Kernel(T.ceildiv(2037, 64), threads=32) as bx:
        As = T.alloc_shared((64, 64), "bfloat16")
        Xs = T.alloc_shared((64,), "bfloat16")
        p = T.alloc_fragment((64, 64), "float32")
        acc = T.alloc_fragment((64,), "float32")
        T.fill(acc, 0)
        for ko in T.Pipelined(T.ceildiv(224, 64), num_stages=5):
            T.copy(A[bx * 64, ko * 64], As)
            T.copy(X[ko * 64], Xs)
            for i, j in T.Parallel(64, 64):
                p[i, j] = T.cast(As[i, j], "float32") * T.cast(Xs[j], "float32")
            T.reduce_sum(p, acc, dim=1, clear=False)
        T.copy(acc, Y[bx * 64])
