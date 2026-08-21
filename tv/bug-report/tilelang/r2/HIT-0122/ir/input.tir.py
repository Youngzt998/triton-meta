# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, O_handle: T.handle):
        Q = T.match_buffer(Q_handle, (1, 4, 192, 32), "bfloat16", strides=(24576, 6144, 32, 1))
        K = T.match_buffer(K_handle, (1, 4, 768, 32), "bfloat16", strides=(98304, 24576, 32, 1))
        V = T.match_buffer(V_handle, (1, 4, 768, 32), "bfloat16", strides=(98304, 24576, 32, 1))
        O = T.match_buffer(O_handle, (1, 4, 192, 32), "bfloat16", strides=(24576, 6144, 32, 1))
        # with T.sblock("root"):
        for bx in T.thread_binding(2, thread="blockIdx.x"):
            for by in T.thread_binding(4, thread="blockIdx.y"):
                for bz in T.thread_binding(1, thread="blockIdx.z"):
                    for tx in T.thread_binding(128, thread="threadIdx.x"):
                        for ty in T.thread_binding(1, thread="threadIdx.y"):
                            for tz in T.thread_binding(1, thread="threadIdx.z"):
                                with T.sblock("tilelang_root"):
                                    T.reads()
                                    T.writes()
                                    Qs = T.sblock_alloc_buffer((128, 32), "bfloat16", scope="shared.dyn")
                                    Ks = T.sblock_alloc_buffer((128, 32), "bfloat16", scope="shared.dyn")
                                    Vs = T.sblock_alloc_buffer((128, 32), "bfloat16", scope="shared.dyn")
                                    Os = T.sblock_alloc_buffer((128, 32), "bfloat16", scope="shared.dyn")
                                    s = T.sblock_alloc_buffer((128, 128), scope="local.fragment")
                                    sc = T.sblock_alloc_buffer((128, 128), "bfloat16", scope="local.fragment")
                                    o = T.sblock_alloc_buffer((128, 32), scope="local.fragment")
                                    mx = T.sblock_alloc_buffer((128,), scope="local.fragment")
                                    mp = T.sblock_alloc_buffer((128,), scope="local.fragment")
                                    ss = T.sblock_alloc_buffer((128,), scope="local.fragment")
                                    sm = T.sblock_alloc_buffer((128,), scope="local.fragment")
                                    ls = T.sblock_alloc_buffer((128,), scope="local.fragment")
                                    T.copy(T.region(Q[bz, by, bx * 128, 0], 1, 1, 1, 128, 32), T.region(Qs[0, 0], 2, 128, 32))
                                    T.fill(T.region(o[0, 0], 2, 128, 32), 0)
                                    T.fill(T.region(ls[0], 2, 128), 0)
                                    T.fill(T.region(mx[0], 2, 128), T.infinity("float32") * T.float32(-1.0))
                                    for k in range(6):
                                        T.copy(T.region(K[bz, by, k * 128, 0], 1, 1, 1, 128, 32), T.region(Ks[0, 0], 2, 128, 32))
                                        T.fill(T.region(s[0, 0], 2, 128, 128), 0)
                                        T.gemm(T.region(Qs[0, 0], 1, 128, 32), T.region(Ks[0, 0], 1, 128, 32), T.region(s[0, 0], 3, 128, 128), T.bool(False), T.bool(True), 128, 128, 32, 1, T.bool(False), 32, 32, 0, 0, 1, 0, 0, 0, 0)
                                        T.copy(T.region(V[bz, by, k * 128, 0], 1, 1, 1, 128, 32), T.region(Vs[0, 0], 2, 128, 32))
                                        T.copy(T.region(mx[0], 1, 128), T.region(mp[0], 2, 128))
                                        T.fill(T.region(mx[0], 2, 128), T.infinity("float32") * T.float32(-1.0))
                                        T.reduce(T.region(s[0, 0], 1, 128, 128), T.region(mx[0], 2, 128), "max", 1, T.bool(False))
                                        for i in T.parallel(128):
                                            ss[i] = T.exp2(mp[i] * T.float32(0.2550348614920494) - mx[i] * T.float32(0.2550348614920494))
                                        for i in T.parallel(128):
                                            for j in T.parallel(32):
                                                o[i, j] = o[i, j] * ss[i]
                                        for i in T.parallel(128):
                                            for j in T.parallel(128):
                                                s[i, j] = T.exp2(s[i, j] * T.float32(0.2550348614920494) - mx[i] * T.float32(0.2550348614920494))
                                        T.copy(T.region(s[0, 0], 1, 128, 128), T.region(sc[0, 0], 2, 128, 128))
                                        T.gemm(T.region(sc[0, 0], 1, 128, 128), T.region(Vs[0, 0], 1, 128, 32), T.region(o[0, 0], 3, 128, 32), T.bool(False), T.bool(False), 128, 32, 128, 1, T.bool(False), 128, 32, 0, 0, 1, 0, 0, 0, 0)
                                        T.reduce(T.region(s[0, 0], 1, 128, 128), T.region(sm[0], 2, 128), "sum", 1, T.bool(True))
                                        for i in T.parallel(128):
                                            ls[i] = ls[i] * ss[i] + sm[i]
                                    for i in T.parallel(128):
                                        for j in T.parallel(32):
                                            o[i, j] = o[i, j] / ls[i]
                                    T.copy(T.region(o[0, 0], 1, 128, 32), T.region(Os[0, 0], 2, 128, 32))
                                    T.copy(T.region(Os[0, 0], 1, 128, 32), T.region(O[bz, by, bx * 128, 0], 2, 1, 1, 128, 32))