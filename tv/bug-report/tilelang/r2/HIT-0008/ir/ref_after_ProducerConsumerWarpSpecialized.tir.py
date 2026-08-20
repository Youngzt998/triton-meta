# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A_handle: T.handle, X_handle: T.handle, Y_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90a", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tl_tiled_ws_applied": 1})
        A = T.match_buffer(A_handle, (511, 1025), strides=(1025, 1))
        X = T.match_buffer(X_handle, (1025,), strides=(1,))
        Y = T.match_buffer(Y_handle, (511,), strides=(1,))
        # with T.sblock("root"):
        bx = T.launch_thread("blockIdx.x", 8)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            mbarrier = T.handle("uint64", "shared.barrier")
            T.sblock_attr({"barrier_init": {mbarrier: [1, 1, 1, 32, 32, 32]}})
            As = T.sblock_alloc_buffer((3, 64, 256), scope="shared.dyn")
            Xs = T.sblock_alloc_buffer((3, 256), scope="shared.dyn")
            p = T.sblock_alloc_buffer((64, 256), scope="local.fragment")
            acc = T.sblock_alloc_buffer((64,), scope="local.fragment")
            mbarrier_1 = T.sblock_alloc_buffer((6,), "uint64", data=mbarrier, scope="shared.barrier")
            T.attr([32, 32], "kWarpSpecializationScope", 0)
            if tx < 32:
                for ko in range(5):
                    T.mbarrier_wait_parity(mbarrier_1[3 + ko % 3], T.bitwise_xor(ko // 3 % 2, 1))
                    T.copy(T.region(A[bx * 64, ko * 256], 1, 64, 256), T.region(As[ko % 3, 0, 0], 2, 1, 64, 256), tl.pipeline_mbar_phase_expr=ko // 3 % 2)
                    T.tma_copy(T.region(X[ko * 256], 1, 256), T.region(Xs[ko % 3, 0], 2, 1, 256), barrier=mbarrier_1[ko % 3], is_tma_copy=1, emit_arrive=1, tl.pipeline_mbar_phase_expr=ko // 3 % 2)
            else:
                T.fill(T.region(acc[0], 2, 64), 0)
                for ko in range(5):
                    T.mbarrier_wait_parity(mbarrier_1[ko % 3], ko // 3 % 2)
                    for i in T.parallel(64):
                        for j in T.parallel(256):
                            p[i, j] = As[ko % 3, i, j] * Xs[ko % 3, j]
                    T.ptx_arrive_barrier(mbarrier_1[3 + ko % 3])
                    T.reduce(T.region(p[0, 0], 1, 64, 256), T.region(acc[0], 2, 64), "sum", 1, T.bool(False))
                T.copy(T.region(acc[0], 1, 64), T.region(Y[bx * 64], 2, 64))