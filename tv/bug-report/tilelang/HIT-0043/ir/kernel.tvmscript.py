# from tvm.script import tirx as T

@T.prim_func
def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle, Sinks_handle: T.handle):
    Q = T.match_buffer(Q_handle, (1, 1, 256, 128), "float16", strides=(32768, 32768, 128, 1))
    K = T.match_buffer(K_handle, (1, 1, 256, 128), "float16", strides=(32768, 32768, 128, 1))
    V = T.match_buffer(V_handle, (1, 1, 256, 128), "float16", strides=(32768, 32768, 128, 1))
    Output = T.match_buffer(Output_handle, (1, 1, 256, 128), "float16", strides=(32768, 32768, 128, 1))
    Sinks = T.match_buffer(Sinks_handle, (1,), "float16", strides=(1,))
    # with T.sblock("root"):
    for bx in T.thread_binding(2, thread="blockIdx.x"):
        for by in T.thread_binding(1, thread="blockIdx.y"):
            for bz in T.thread_binding(1, thread="blockIdx.z"):
                for tx in T.thread_binding(256, thread="threadIdx.x"):
                    for ty in T.thread_binding(1, thread="threadIdx.y"):
                        for tz in T.thread_binding(1, thread="threadIdx.z"):
                            with T.sblock("tilelang_root"):
                                T.reads()
                                T.writes()
                                Q_shared = T.sblock_alloc_buffer((128, 128), "float16", scope="shared.dyn")
                                K_shared = T.sblock_alloc_buffer((128, 128), "float16", scope="shared.dyn")
                                V_shared = T.sblock_alloc_buffer((128, 128), "float16", scope="shared.dyn")
                                O_shared = T.sblock_alloc_buffer((128, 128), "float16", scope="shared.dyn")
                                acc_s = T.sblock_alloc_buffer((128, 128), scope="local.fragment")
                                acc_s_cast = T.sblock_alloc_buffer((128, 128), "float16", scope="local.fragment")
                                acc_o = T.sblock_alloc_buffer((128, 128), scope="local.fragment")
                                scores_max = T.sblock_alloc_buffer((128,), scope="local.fragment")
                                scores_max_prev = T.sblock_alloc_buffer((128,), scope="local.fragment")
                                scores_scale = T.sblock_alloc_buffer((128,), scope="local.fragment")
                                scores_sum = T.sblock_alloc_buffer((128,), scope="local.fragment")
                                logsum = T.sblock_alloc_buffer((128,), scope="local.fragment")
                                sinks = T.sblock_alloc_buffer((128,), "float16", scope="local.fragment")
                                T.copy(T.region(Q[bz, by, bx * 128, 0], 1, 1, 1, 128, 128), T.region(Q_shared[0, 0], 2, 128, 128))
                                T.fill(T.region(acc_o[0, 0], 2, 128, 128), 0)
                                T.fill(T.region(logsum[0], 2, 128), 0)
                                T.fill(T.region(scores_max[0], 2, 128), T.infinity("float32") * T.float32(-1.0))
                                for i in T.parallel(128):
                                    sinks[i] = Sinks[by]
                                end: T.int32 = T.min(2, ((bx + 1) * 128 + 128 - 1) // 128)
                                for k in T.serial(end, annotations={"num_stages": 2}):
                                    T.copy(T.region(K[bz, by, k * 128, 0], 1, 1, 1, 128, 128), T.region(K_shared[0, 0], 2, 128, 128))
                                    for i in T.parallel(128):
                                        for j in T.parallel(128):
                                            q_idx: T.int32 = bx * 128 + i
                                            k_idx: T.int32 = k * 128 + j
                                            acc_s[i, j] = T.if_then_else(q_idx >= k_idx, T.float32(0.0), T.infinity("float32") * T.float32(-1.0))
                                    T.gemm(T.region(Q_shared[0, 0], 1, 128, 128), T.region(K_shared[0, 0], 1, 128, 128), T.region(acc_s[0, 0], 3, 128, 128), T.bool(False), T.bool(True), 128, 128, 128, 1, T.bool(False), 128, 128, 0, 0, 1, 0, 0, 0, 0)
                                    T.copy(T.region(scores_max[0], 1, 128), T.region(scores_max_prev[0], 2, 128))
                                    T.fill(T.region(scores_max[0], 2, 128), T.infinity("float32") * T.float32(-1.0))
                                    T.reduce(T.region(acc_s[0, 0], 1, 128, 128), T.region(scores_max[0], 2, 128), "max", 1, T.bool(False))
                                    for i in T.parallel(128):
                                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                                    for i in T.parallel(128):
                                        scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.1275174307460247) - scores_max[i] * T.float32(0.1275174307460247))
                                    for i in T.parallel(128):
                                        for j in T.parallel(128):
                                            acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(0.1275174307460247) - scores_max[i] * T.float32(0.1275174307460247))
                                    T.reduce(T.region(acc_s[0, 0], 1, 128, 128), T.region(scores_sum[0], 2, 128), "sum", 1, T.bool(True))
                                    for i in T.parallel(128):
                                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                                    T.copy(T.region(acc_s[0, 0], 1, 128, 128), T.region(acc_s_cast[0, 0], 2, 128, 128))
                                    for i in T.parallel(128):
                                        for j in T.parallel(128):
                                            acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                                    T.copy(T.region(V[bz, by, k * 128, 0], 1, 1, 1, 128, 128), T.region(V_shared[0, 0], 2, 128, 128))
                                    T.gemm(T.region(acc_s_cast[0, 0], 1, 128, 128), T.region(V_shared[0, 0], 1, 128, 128), T.region(acc_o[0, 0], 3, 128, 128), T.bool(False), T.bool(False), 128, 128, 128, 1, T.bool(False), 128, 128, 0, 0, 1, 0, 0, 0, 0)
                                for i in T.parallel(128):
                                    logsum[i] = logsum[i] + T.exp2(T.Cast("float32", sinks[i]) * T.float32(1.44269504) - scores_max[i] * T.float32(0.1275174307460247))
                                for i in T.parallel(128):
                                    for j in T.parallel(128):
                                        acc_o[i, j] = acc_o[i, j] / logsum[i]
                                T.copy(T.region(acc_o[0, 0], 1, 128, 128), T.region(O_shared[0, 0], 2, 128, 128))
                                T.copy(T.region(O_shared[0, 0], 1, 128, 128), T.region(Output[bz, by, bx * 128, 0], 2, 1, 1, 128, 128))