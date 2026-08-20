# from tvm.script import tirx as T

@T.prim_func
def main(A_handle: T.handle, X_handle: T.handle, Y_handle: T.handle):
    A = T.match_buffer(A_handle, (2037, 224), "bfloat16", strides=(224, 1))
    X = T.match_buffer(X_handle, (224,), "bfloat16", strides=(1,))
    Y = T.match_buffer(Y_handle, (2037,), strides=(1,))
    # with T.sblock("root"):
    for bx in T.thread_binding(32, thread="blockIdx.x"):
        for tx in T.thread_binding(32, thread="threadIdx.x"):
            for ty in T.thread_binding(1, thread="threadIdx.y"):
                for tz in T.thread_binding(1, thread="threadIdx.z"):
                    with T.sblock("tilelang_root"):
                        T.reads()
                        T.writes()
                        As = T.sblock_alloc_buffer((64, 64), "bfloat16", scope="shared.dyn")
                        Xs = T.sblock_alloc_buffer((64,), "bfloat16", scope="shared.dyn")
                        p = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
                        acc = T.sblock_alloc_buffer((64,), scope="local.fragment")
                        T.fill(T.region(acc[0], 2, 64), 0)
                        for ko in T.serial(4, annotations={"num_stages": 5}):
                            T.copy(T.region(A[bx * 64, ko * 64], 1, 64, 64), T.region(As[0, 0], 2, 64, 64))
                            T.copy(T.region(X[ko * 64], 1, 64), T.region(Xs[0], 2, 64))
                            for i in T.parallel(64):
                                for j in T.parallel(64):
                                    p[i, j] = T.Cast("float32", As[i, j]) * T.Cast("float32", Xs[j])
                            T.reduce(T.region(p[0, 0], 1, 64, 64), T.region(acc[0], 2, 64), "sum", 1, T.bool(False))
                        T.copy(T.region(acc[0], 1, 64), T.region(Y[bx * 64], 2, 64))