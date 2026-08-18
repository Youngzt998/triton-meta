# from tvm.script import tirx as T

@T.prim_func
def kernel(K_handle: T.handle, V_handle: T.handle, Beta_handle: T.handle, G_handle: T.handle, A_handle: T.handle, dw_handle: T.handle, du_handle: T.handle, dA_handle: T.handle, dk_handle: T.handle, dv_handle: T.handle, dbeta_handle: T.handle, dg_handle: T.handle):
    K = T.match_buffer(K_handle, (1, 1024, 32, 128), "bfloat16", strides=(4194304, 4096, 128, 1))
    V = T.match_buffer(V_handle, (1, 1024, 32, 128), "bfloat16", strides=(4194304, 4096, 128, 1))
    Beta = T.match_buffer(Beta_handle, (1, 1024, 32), "bfloat16", strides=(32768, 32, 1))
    G = T.match_buffer(G_handle, (1, 1024, 32), strides=(32768, 32, 1))
    A = T.match_buffer(A_handle, (1, 1024, 32, 64), "bfloat16", strides=(2097152, 2048, 64, 1))
    dw = T.match_buffer(dw_handle, (1, 1024, 32, 128), "bfloat16", strides=(4194304, 4096, 128, 1))
    du = T.match_buffer(du_handle, (1, 1024, 32, 128), "bfloat16", strides=(4194304, 4096, 128, 1))
    dA = T.match_buffer(dA_handle, (1, 1024, 32, 64), "bfloat16", strides=(2097152, 2048, 64, 1))
    dk = T.match_buffer(dk_handle, (1, 1024, 32, 128), "bfloat16", strides=(4194304, 4096, 128, 1))
    dv = T.match_buffer(dv_handle, (1, 1024, 32, 128), "bfloat16", strides=(4194304, 4096, 128, 1))
    dbeta = T.match_buffer(dbeta_handle, (1, 1024, 32), "bfloat16", strides=(32768, 32, 1))
    dg = T.match_buffer(dg_handle, (1, 1024, 32), strides=(32768, 32, 1))
    # with T.sblock("root"):
    for bx in T.thread_binding(16, thread="blockIdx.x"):
        for by in T.thread_binding(32, thread="blockIdx.y"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                for ty in T.thread_binding(1, thread="threadIdx.y"):
                    for tz in T.thread_binding(1, thread="threadIdx.z"):
                        with T.sblock("tilelang_root"):
                            T.reads()
                            T.writes()
                            A_shared = T.sblock_alloc_buffer((64, 64), "bfloat16", scope="shared.dyn")
                            K_shared = T.sblock_alloc_buffer((64, 64), "bfloat16", scope="shared.dyn")
                            K_shared_beta_g = T.sblock_alloc_buffer((64, 64), "bfloat16", scope="shared.dyn")
                            V_shared = T.sblock_alloc_buffer((64, 32), "bfloat16", scope="shared.dyn")
                            V_shared_beta = T.sblock_alloc_buffer((64, 32), "bfloat16", scope="shared.dyn")
                            Beta_shared = T.sblock_alloc_buffer((64,), "bfloat16", scope="shared.dyn")
                            G_shared = T.sblock_alloc_buffer((64,), scope="shared.dyn")
                            G_shared_exp = T.sblock_alloc_buffer((64,), scope="shared.dyn")
                            dw_shared = T.sblock_alloc_buffer((64, 64), "bfloat16", scope="shared.dyn")
                            du_shared = T.sblock_alloc_buffer((64, 32), "bfloat16", scope="shared.dyn")
                            dA_fragment = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
                            dk_fragment = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
                            dk_fragment_beta_g = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
                            dv_fragment = T.sblock_alloc_buffer((64, 32), scope="local.fragment")
                            dv_fragment_beta = T.sblock_alloc_buffer((64, 32), scope="local.fragment")
                            dbeta_fragment_k = T.sblock_alloc_buffer((64,), scope="local.fragment")
                            dbeta_fragment_v = T.sblock_alloc_buffer((64,), scope="local.fragment")
                            dbeta_fragment_reduce_tmpk = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
                            dbeta_fragment_reduce_tmpv = T.sblock_alloc_buffer((64, 32), scope="local.fragment")
                            dg_fragment = T.sblock_alloc_buffer((64,), scope="local.fragment")
                            dg_fragment_reduce_tmp = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
                            bb: T.int32 = by // 32
                            bh: T.int32 = by % 32
                            T.attr(None, "threadblock_swizzle_pattern", T.tvm_tuple("rasterization2DRow", 10))
                            T.fill(T.region(dA_fragment[0, 0], 2, 64, 64), 0)
                            T.fill(T.region(dk_fragment[0, 0], 2, 64, 64), 0)
                            T.fill(T.region(dk_fragment_beta_g[0, 0], 2, 64, 64), 0)
                            T.fill(T.region(dv_fragment[0, 0], 2, 64, 32), 0)
                            T.fill(T.region(dv_fragment_beta[0, 0], 2, 64, 32), 0)
                            T.fill(T.region(dbeta_fragment_k[0], 2, 64), 0)
                            T.fill(T.region(dbeta_fragment_v[0], 2, 64), 0)
                            T.fill(T.region(dg_fragment[0], 2, 64), 0)
                            T.copy(T.region(A[bb, bx * 64, bh, 0], 1, 1, 64, 1, 64), T.region(A_shared[0, 0], 2, 64, 64))
                            for i_s in T.parallel(64):
                                Beta_shared[i_s] = Beta[bb, bx * 64 + i_s, bh]
                                G_shared[i_s] = G[bb, bx * 64 + i_s, bh]
                                G_shared_exp[i_s] = T.exp(G[bb, bx * 64 + i_s, bh])
                            for i_k in range(2):
                                T.copy(T.region(K[bb, bx * 64, bh, i_k * 64], 1, 1, 64, 1, 64), T.region(K_shared[0, 0], 2, 64, 64))
                                for i_s in T.parallel(64):
                                    for i_k2 in T.parallel(64):
                                        K_shared_beta_g[i_s, i_k2] = T.Cast("bfloat16", T.Cast("float32", K_shared[i_s, i_k2] * Beta_shared[i_s]) * G_shared_exp[i_s])
                                T.copy(T.region(dw[bb, bx * 64, bh, i_k * 64], 1, 1, 64, 1, 64), T.region(dw_shared[0, 0], 2, 64, 64))
                                T.gemm(T.region(dw_shared[0, 0], 1, 64, 64), T.region(K_shared_beta_g[0, 0], 1, 64, 64), T.region(dA_fragment[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 0, T.bool(False), 64, 64, 0, 0, 1, 0, 0, 0, 0)
                                T.gemm(T.region(A_shared[0, 0], 1, 64, 64), T.region(dw_shared[0, 0], 1, 64, 64), T.region(dk_fragment_beta_g[0, 0], 3, 64, 64), T.bool(True), T.bool(False), 64, 64, 64, 0, T.bool(True), 64, 64, 0, 0, 1, 0, 0, 0, 0)
                                for i_s in T.parallel(64):
                                    for i_k2 in T.parallel(64):
                                        dk_fragment[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * T.Cast("float32", Beta_shared[i_s]) * G_shared_exp[i_s]
                                for i_s in T.parallel(64):
                                    for i_k2 in T.parallel(64):
                                        dbeta_fragment_reduce_tmpk[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * T.Cast("float32", K_shared[i_s, i_k2]) * G_shared_exp[i_s]
                                T.reduce(T.region(dbeta_fragment_reduce_tmpk[0, 0], 1, 64, 64), T.region(dbeta_fragment_k[0], 2, 64), "sum", 1, T.bool(False))
                                for i_s in T.parallel(64):
                                    for i_k2 in T.parallel(64):
                                        dg_fragment_reduce_tmp[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * T.Cast("float32", K_shared[i_s, i_k2]) * G_shared_exp[i_s] * T.Cast("float32", Beta_shared[i_s])
                                T.reduce(T.region(dg_fragment_reduce_tmp[0, 0], 1, 64, 64), T.region(dg_fragment[0], 2, 64), "sum", 1, T.bool(False))
                                T.copy(T.region(dk_fragment[0, 0], 1, 64, 64), T.region(dk[bb, bx * 64, bh, i_k * 64], 2, 1, 64, 1, 64))
                            for i_v in range(4):
                                T.copy(T.region(V[bb, bx * 64, bh, i_v * 32], 1, 1, 64, 1, 32), T.region(V_shared[0, 0], 2, 64, 32))
                                for i_s in T.parallel(64):
                                    for i_v2 in T.parallel(32):
                                        V_shared_beta[i_s, i_v2] = V_shared[i_s, i_v2] * Beta_shared[i_s]
                                T.copy(T.region(du[bb, bx * 64, bh, i_v * 32], 1, 1, 64, 1, 32), T.region(du_shared[0, 0], 2, 64, 32))
                                T.gemm(T.region(du_shared[0, 0], 1, 64, 32), T.region(V_shared_beta[0, 0], 1, 64, 32), T.region(dA_fragment[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 32, 0, T.bool(False), 32, 32, 0, 0, 1, 0, 0, 0, 0)
                                T.gemm(T.region(A_shared[0, 0], 1, 64, 64), T.region(du_shared[0, 0], 1, 64, 32), T.region(dv_fragment_beta[0, 0], 3, 64, 32), T.bool(True), T.bool(False), 64, 32, 64, 0, T.bool(True), 64, 32, 0, 0, 1, 0, 0, 0, 0)
                                for i_s in T.parallel(64):
                                    for i_v2 in T.parallel(32):
                                        dv_fragment[i_s, i_v2] = dv_fragment_beta[i_s, i_v2] * T.Cast("float32", Beta_shared[i_s])
                                for i_s in T.parallel(64):
                                    for i_v2 in T.parallel(32):
                                        dbeta_fragment_reduce_tmpv[i_s, i_v2] = dv_fragment_beta[i_s, i_v2] * T.Cast("float32", V_shared[i_s, i_v2])
                                T.reduce(T.region(dbeta_fragment_reduce_tmpv[0, 0], 1, 64, 32), T.region(dbeta_fragment_v[0], 2, 64), "sum", 1, T.bool(False))
                                T.copy(T.region(dv_fragment[0, 0], 1, 64, 32), T.region(dv[bb, bx * 64, bh, i_v * 32], 2, 1, 64, 1, 32))
                            for i_s in T.parallel(64):
                                dbeta[bb, bx * 64 + i_s, bh] = T.Cast("bfloat16", dbeta_fragment_k[i_s] + dbeta_fragment_v[i_s])
                                dg[bb, bx * 64 + i_s, bh] = dg_fragment[i_s]
                            T.copy(T.region(dA_fragment[0, 0], 1, 64, 64), T.region(dA[bb, bx * 64, bh, 0], 2, 1, 64, 1, 64))