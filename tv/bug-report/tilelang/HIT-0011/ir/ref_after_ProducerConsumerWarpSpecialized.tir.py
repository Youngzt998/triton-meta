# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, O_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90a", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tl_tiled_ws_applied": 1})
        Q = T.match_buffer(Q_handle, (1, 2, 130, 64), "float16", strides=(16640, 8320, 64, 1))
        K = T.match_buffer(K_handle, (1, 2, 256, 64), "float16", strides=(32768, 16384, 64, 1))
        V = T.match_buffer(V_handle, (1, 2, 256, 64), "float16", strides=(32768, 16384, 64, 1))
        O = T.match_buffer(O_handle, (1, 2, 130, 64), "float16", strides=(16640, 8320, 64, 1))
        # with T.sblock("root"):
        bx = T.launch_thread("blockIdx.x", 3)
        by = T.launch_thread("blockIdx.y", 2)
        bz = T.launch_thread("blockIdx.z", 1)
        tx = T.launch_thread("threadIdx.x", 256)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            mbarrier = T.handle("uint64", "shared.barrier")
            T.sblock_attr({"barrier_init": {mbarrier: [1, 1, 128, 128, 1]}})
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
            mbarrier_1 = T.sblock_alloc_buffer((5,), "uint64", data=mbarrier, scope="shared.barrier")
            T.tma_copy(T.region(Q[0, by, bx * 64, 0], 1, 1, 1, 64, 64), T.region(Qs[0, 0], 2, 64, 64), barrier=mbarrier_1[4], is_tma_copy=1, emit_arrive=1)
            T.attr([128, 128], "kWarpSpecializationScope", 0)
            if tx < 128:
                for k in range(T.min(4, bx + 3)):
                    T.mbarrier_wait_parity(mbarrier_1[2 + T.FloorMod(k, 1)], T.bitwise_xor(T.FloorDiv(k, 1) % 2, 1))
                    T.tma_copy(T.region(K[0, by, k * 64, 0], 1, 1, 1, 64, 64), T.region(Ks[0, 0], 2, 64, 64), barrier=mbarrier_1[T.FloorMod(k, 1)], is_tma_copy=1, emit_arrive=1, tl.pipeline_mbar_phase_expr=T.FloorDiv(k, 1) % 2)
                    T.mbarrier_wait_parity(mbarrier_1[3 + T.FloorMod(k, 1)], T.bitwise_xor(T.FloorDiv(k, 1) % 2, 1))
                    T.tma_copy(T.region(V[0, by, k * 64, 0], 1, 1, 1, 64, 64), T.region(Vs[0, 0], 2, 64, 64), barrier=mbarrier_1[1 + T.FloorMod(k, 1)], is_tma_copy=1, emit_arrive=1, tl.pipeline_mbar_phase_expr=T.FloorDiv(k, 1) % 2)
            else:
                T.fill(T.region(o[0, 0], 2, 64, 64), 0)
                T.fill(T.region(ls[0], 2, 64), 0)
                T.fill(T.region(mx[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                for k in range(T.min(4, bx + 3)):
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            s[i, j] = T.if_then_else(k * 64 + j <= bx * 64 + i + 126, T.float32(0.0), T.infinity("float32") * T.float32(-1.0))
                    if k == 0:
                        T.mbarrier_wait_parity(mbarrier_1[4], 0)
                    T.mbarrier_wait_parity(mbarrier_1[T.FloorMod(k, 1)], T.FloorDiv(k, 1) % 2)
                    T.gemm(T.region(Qs[0, 0], 1, 64, 64), T.region(Ks[0, 0], 1, 64, 64), T.region(s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 1, T.bool(False), 64, 64, 0, 0, 1, 0, 0, 0, 0, tl.pipeline_mbar_phase_expr=T.FloorDiv(k, 1) % 2)
                    T.ptx_arrive_barrier(mbarrier_1[2 + T.FloorMod(k, 1)])
                    T.copy(T.region(mx[0], 1, 64), T.region(mp[0], 2, 64), tl.pipeline_mbar_phase_expr=T.FloorDiv(k, 1) % 2)
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
                    T.copy(T.region(s[0, 0], 1, 64, 64), T.region(sc[0, 0], 2, 64, 64), tl.pipeline_mbar_phase_expr=T.FloorDiv(k, 1) % 2)
                    T.mbarrier_wait_parity(mbarrier_1[1 + T.FloorMod(k, 1)], T.FloorDiv(k, 1) % 2)
                    T.gemm(T.region(sc[0, 0], 1, 64, 64), T.region(Vs[0, 0], 1, 64, 64), T.region(o[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 1, T.bool(False), 64, 64, 0, 0, 1, 0, 0, 0, 0, tl.pipeline_mbar_phase_expr=T.FloorDiv(k, 1) % 2)
                    T.ptx_arrive_barrier(mbarrier_1[3 + T.FloorMod(k, 1)])
                    T.reduce(T.region(s[0, 0], 1, 64, 64), T.region(sm[0], 2, 64), "sum", 1, T.bool(True))
                    for i in T.parallel(64):
                        ls[i] = ls[i] * ss[i] + sm[i]
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        o[i, j] = o[i, j] / ls[i]
                T.copy(T.region(o[0, 0], 1, 64, 64), T.region(Os[0, 0], 2, 64, 64))
                T.copy(T.region(Os[0, 0], 1, 64, 64), T.region(O[0, by, bx * 64, 0], 2, 1, 1, 64, 64))