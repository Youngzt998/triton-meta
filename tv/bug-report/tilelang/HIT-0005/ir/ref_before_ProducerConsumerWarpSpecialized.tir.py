# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, O_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90a", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        Q = T.match_buffer(Q_handle, (1, 1, 128, 64), "float16", strides=(8192, 8192, 64, 1))
        K = T.match_buffer(K_handle, (1, 1, 256, 64), "float16", strides=(16384, 16384, 64, 1))
        V = T.match_buffer(V_handle, (1, 1, 256, 64), "float16", strides=(16384, 16384, 64, 1))
        O = T.match_buffer(O_handle, (1, 1, 128, 64), "float16", strides=(8192, 8192, 64, 1))
        # with T.sblock("root"):
        bx = T.launch_thread("blockIdx.x", 2)
        by = T.launch_thread("blockIdx.y", 1)
        bz = T.launch_thread("blockIdx.z", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            Qs = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
            Ks = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
            Vs = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
            Os = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
            s = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
            sc = T.sblock_alloc_buffer((64, 64), "float16", scope="local.fragment")
            o = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
            mx = T.sblock_alloc_buffer((64,), scope="local.fragment")
            mp = T.sblock_alloc_buffer((64,), scope="local.fragment")
            ss = T.sblock_alloc_buffer((64,), scope="local.fragment")
            sm = T.sblock_alloc_buffer((64,), scope="local.fragment")
            ls = T.sblock_alloc_buffer((64,), scope="local.fragment")
            T.copy(T.region(Q[0, 0, bx * 64, 0], 1, 1, 1, 64, 64), T.region(Qs[0, 0], 2, 64, 64))
            T.fill(T.region(o[0, 0], 2, 64, 64), 0)
            T.fill(T.region(ls[0], 2, 64), 0)
            T.fill(T.region(mx[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
            for k in T.serial(bx + 3, annotations={"num_stages": 2}):
                T.copy(T.region(K[0, 0, k * 64, 0], 1, 1, 1, 64, 64), T.region(Ks[0, 0], 2, 64, 64))
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        s[i, j] = T.if_then_else(k * 64 + j <= bx * 64 + i + 128, T.float32(0.0), T.infinity("float32") * T.float32(-1.0))
                T.gemm(T.region(Qs[0, 0], 1, 64, 64), T.region(Ks[0, 0], 1, 64, 64), T.region(s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 1, T.bool(False), 64, 64, 0, 0, 1, 0, 0, 0, 0)
                T.copy(T.region(V[0, 0, k * 64, 0], 1, 1, 1, 64, 64), T.region(Vs[0, 0], 2, 64, 64))
                T.copy(T.region(mx[0], 1, 64), T.region(mp[0], 2, 64))
                T.fill(T.region(mx[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                T.reduce(T.region(s[0, 0], 1, 64, 64), T.region(mx[0], 2, 64), "max", 1, T.bool(False))
                for i in T.parallel(64):
                    ss[i] = T.exp2(mp[i] * T.float32(0.18033688) - mx[i] * T.float32(0.18033688))
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        o[i, j] = o[i, j] * ss[i]
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        s[i, j] = T.exp2(s[i, j] * T.float32(0.18033688) - mx[i] * T.float32(0.18033688))
                T.copy(T.region(s[0, 0], 1, 64, 64), T.region(sc[0, 0], 2, 64, 64))
                T.gemm(T.region(sc[0, 0], 1, 64, 64), T.region(Vs[0, 0], 1, 64, 64), T.region(o[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 1, T.bool(False), 64, 64, 0, 0, 1, 0, 0, 0, 0)
                T.reduce(T.region(s[0, 0], 1, 64, 64), T.region(sm[0], 2, 64), "sum", 1, T.bool(True))
                for i in T.parallel(64):
                    ls[i] = ls[i] * ss[i] + sm[i]
            for i in T.parallel(64):
                for j in T.parallel(64):
                    o[i, j] = o[i, j] / ls[i]
            T.copy(T.region(o[0, 0], 1, 64, 64), T.region(Os[0, 0], 2, 64, 64))
            T.copy(T.region(Os[0, 0], 1, 64, 64), T.region(O[0, 0, bx * 64, 0], 2, 1, 1, 64, 64))