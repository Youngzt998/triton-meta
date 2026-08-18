# from tvm.script import tirx as T

@T.prim_func
def rand_kernel(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle, D_handle: T.handle, E_handle: T.handle):
    A = T.match_buffer(A_handle, (512,), "uint32", strides=(1,))
    B = T.match_buffer(B_handle, (512,), strides=(1,))
    C = T.match_buffer(C_handle, (512,), "float64", strides=(1,))
    D = T.match_buffer(D_handle, (512,), strides=(1,))
    E = T.match_buffer(E_handle, (512,), "float64", strides=(1,))
    # with T.sblock("root"):
    for bx in T.thread_binding(4, thread="blockIdx.x"):
        for tx in T.thread_binding(1, thread="threadIdx.x"):
            for ty in T.thread_binding(1, thread="threadIdx.y"):
                for tz in T.thread_binding(1, thread="threadIdx.z"):
                    with T.sblock("tilelang_root"):
                        T.reads()
                        T.writes()
                        T.rng_init(123, 0, bx * 128 + tx * 128, "curandStatePhilox4_32_10_t")
                        for i in T.parallel(1):
                            for j in T.parallel(128):
                                offsets: T.int32 = (bx + i) * 128
                                idx: T.int32 = offsets + j
                                if idx < 512:
                                    A[idx] = T.rng_rand()
                        for i in T.parallel(1):
                            for j in T.parallel(128):
                                offsets: T.int32 = (bx + i) * 128
                                idx: T.int32 = offsets + j
                                if idx < 512:
                                    B[idx] = T.rng_rand_float("uniform")
                        for i in T.parallel(1):
                            for j in T.parallel(128):
                                offsets: T.int32 = (bx + i) * 128
                                idx: T.int32 = offsets + j
                                if idx < 512:
                                    C[idx] = T.rng_rand_float("uniform")
                        for i in T.parallel(1):
                            for j in T.parallel(128):
                                offsets: T.int32 = (bx + i) * 128
                                idx: T.int32 = offsets + j
                                if idx < 512:
                                    D[idx] = T.rng_rand_float("normal")
                        for i in T.parallel(1):
                            for j in T.parallel(128):
                                offsets: T.int32 = (bx + i) * 128
                                idx: T.int32 = offsets + j
                                if idx < 512:
                                    E[idx] = T.rng_rand_float("normal")