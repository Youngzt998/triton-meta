# from tvm.script import tirx as T

@T.prim_func
def matmul(A: T.handle, B: T.handle, C_handle: T.handle):
    A_1 = T.match_buffer(A, (1024, 8192), "float8_e5m2", strides=(8192, 1))
    B_1 = T.match_buffer(B, (1024, 8192), "float8_e5m2", strides=(8192, 1))
    C = T.match_buffer(C_handle, (1024, 1024), strides=(1024, 1))
    # with T.sblock("root"):
    for bx in T.thread_binding(8, thread="blockIdx.x"):
        for by in T.thread_binding(8, thread="blockIdx.y"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                for ty in T.thread_binding(1, thread="threadIdx.y"):
                    for tz in T.thread_binding(1, thread="threadIdx.z"):
                        with T.sblock("tilelang_root"):
                            T.reads()
                            T.writes()
                            A_shared = T.sblock_alloc_buffer((128, 64), "float8_e5m2", scope="shared.dyn")
                            B_shared = T.sblock_alloc_buffer((128, 64), "float8_e5m2", scope="shared.dyn")
                            C_shared = T.sblock_alloc_buffer((128, 128), scope="shared.dyn")
                            C_local = T.sblock_alloc_buffer((128, 128), scope="local.fragment")
                            C_local_accum = T.sblock_alloc_buffer((128, 128), scope="local.fragment")
                            T.fill(T.region(C_local[0, 0], 2, 128, 128), 0)
                            T.fill(T.region(C_local_accum[0, 0], 2, 128, 128), 0)
                            for k in T.serial(128, annotations={"num_stages": 3}):
                                T.copy(T.region(A_1[by * 128, k * 64], 1, 128, 64), T.region(A_shared[0, 0], 2, 128, 64))
                                T.copy(T.region(B_1[bx * 128, k * 64], 1, 128, 64), T.region(B_shared[0, 0], 2, 128, 64))
                                T.gemm(T.region(A_shared[0, 0], 1, 128, 64), T.region(B_shared[0, 0], 1, 128, 64), T.region(C_local[0, 0], 3, 128, 128), T.bool(False), T.bool(True), 128, 128, 64, 0, T.bool(False), 64, 64, 0, 0, 1, 0, 0, 0, 0)
                                if (k + 1) % 2 == 0:
                                    for i in T.parallel(128):
                                        for j in T.parallel(128):
                                            C_local_accum[i, j] = C_local_accum[i, j] + C_local[i, j]
                                    T.fill(T.region(C_local[0, 0], 2, 128, 128), 0)
                            T.copy(T.region(C_local_accum[0, 0], 1, 128, 128), T.region(C_shared[0, 0], 2, 128, 128))
                            T.copy(T.region(C_shared[0, 0], 1, 128, 128), T.region(C[by * 128, bx * 128], 2, 128, 128))