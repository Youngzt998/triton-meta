# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, KV_handle: T.handle, TopkIndices_handle: T.handle, Output_handle: T.handle, Sinks_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90a", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        Q = T.match_buffer(Q_handle, (1, 512, 8, 128), "bfloat16", strides=(524288, 1024, 128, 1))
        KV = T.match_buffer(KV_handle, (1, 1024, 128), "bfloat16", strides=(131072, 128, 1))
        TopkIndices = T.match_buffer(TopkIndices_handle, (1, 512, 256), "int32", strides=(131072, 256, 1))
        Output = T.match_buffer(Output_handle, (1, 512, 8, 128), "bfloat16", strides=(524288, 1024, 128, 1))
        Sinks = T.match_buffer(Sinks_handle, (8,), "bfloat16", strides=(1,))
        # with T.sblock("root"):
        bx = T.launch_thread("blockIdx.x", 1)
        by = T.launch_thread("blockIdx.y", 512)
        bz = T.launch_thread("blockIdx.z", 1)
        tx = T.launch_thread("threadIdx.x", 256)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            Q_shared = T.sblock_alloc_buffer((64, 128), "bfloat16", scope="shared.dyn")
            KV_shared = T.sblock_alloc_buffer((64, 128), "bfloat16", scope="shared.dyn")
            S_shared = T.sblock_alloc_buffer((64, 64), "bfloat16", scope="shared.dyn")
            Sinks_shared = T.sblock_alloc_buffer((64,), "bfloat16", scope="shared.dyn")
            acc_s = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
            acc_o = T.sblock_alloc_buffer((64, 128), scope="local.fragment")
            acc_o_shared = T.sblock_alloc_buffer((64, 128), "bfloat16", scope="shared.dyn")
            scores_max = T.sblock_alloc_buffer((64,), scope="local.fragment")
            scores_max_prev = T.sblock_alloc_buffer((64,), scope="local.fragment")
            scores_scale = T.sblock_alloc_buffer((64,), scope="local.fragment")
            scores_sum = T.sblock_alloc_buffer((64,), scope="local.fragment")
            logsum = T.sblock_alloc_buffer((64,), scope="local.fragment")
            mask = T.sblock_alloc_buffer((64,), "bool", scope="local.fragment")
            T.copy(T.region(Q[0, by, 0, 0], 1, 1, 1, 64, 128), T.region(Q_shared[0, 0], 2, 64, 128))
            T.copy(T.region(Sinks[0], 1, 64), T.region(Sinks_shared[0], 2, 64))
            T.fill(T.region(acc_o[0, 0], 2, 64, 128), 0)
            T.fill(T.region(logsum[0], 2, 64), 0)
            T.fill(T.region(scores_max[0], 2, 64), -1073741824)
            for i_i in T.serial(4, annotations={"num_stages": 2}):
                for bi_i in T.parallel(64):
                    idx: T.int32 = TopkIndices[0, by, i_i * 64 + bi_i]
                    mask[bi_i] = 0 <= idx
                for bi_i in T.parallel(64):
                    for d_i in T.parallel(128):
                        idx: T.int32 = TopkIndices[0, by, i_i * 64 + bi_i]
                        KV_shared[bi_i, d_i] = KV[0, idx, d_i]
                for h_i in T.parallel(64):
                    for bi_i in T.parallel(64):
                        acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], T.float32(0.0), T.infinity("float32") * T.float32(-1.0))
                T.gemm(T.region(Q_shared[0, 0], 1, 64, 128), T.region(KV_shared[0, 0], 1, 64, 128), T.region(acc_s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 128, 1, T.bool(False), 128, 128, 0, 0, 1, 0, 0, 0, 0)
                T.copy(T.region(scores_max[0], 1, 64), T.region(scores_max_prev[0], 2, 64))
                T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_max[0], 2, 64), "max", 1, T.bool(False))
                for h_i in T.parallel(64):
                    scores_max[h_i] = T.max(scores_max[h_i], scores_max_prev[h_i])
                for h_i in T.parallel(64):
                    scores_scale[h_i] = T.exp2(scores_max_prev[h_i] * T.float32(0.1275174307460247) - scores_max[h_i] * T.float32(0.1275174307460247))
                for h_i in T.parallel(64):
                    for bi_i in T.parallel(64):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * T.float32(0.1275174307460247) - scores_max[h_i] * T.float32(0.1275174307460247))
                T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_sum[0], 2, 64), "sum", 1, T.bool(True))
                for h_i in T.parallel(64):
                    for d_i in T.parallel(128):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] * scores_scale[h_i]
                T.copy(T.region(acc_s[0, 0], 1, 64, 64), T.region(S_shared[0, 0], 2, 64, 64))
                T.gemm(T.region(S_shared[0, 0], 1, 64, 64), T.region(KV_shared[0, 0], 1, 64, 128), T.region(acc_o[0, 0], 3, 64, 128), T.bool(False), T.bool(False), 64, 128, 64, 1, T.bool(False), 64, 128, 0, 0, 1, 0, 0, 0, 0)
                for h_i in T.parallel(64):
                    logsum[h_i] = logsum[h_i] * scores_scale[h_i] + scores_sum[h_i]
            for h_i in T.parallel(64):
                logsum[h_i] = logsum[h_i] + T.exp2(T.Cast("float32", Sinks_shared[h_i]) * T.float32(1.44269504) - scores_max[h_i] * T.float32(0.1275174307460247))
            for h_i in T.parallel(64):
                for d_i in T.parallel(128):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] / logsum[h_i]
            T.copy(T.region(acc_o[0, 0], 1, 64, 128), T.region(acc_o_shared[0, 0], 2, 64, 128))
            T.copy(T.region(acc_o_shared[0, 0], 1, 64, 128), T.region(Output[0, by, 0, 0], 2, 1, 1, 64, 128))