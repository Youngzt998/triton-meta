# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle, Sinks_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90a", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tl.has_tma": T.bool(True), "tl.smem_alignment_map": {"K_shared": 1024, "Q_shared": 1024, "V_shared": 1024}, "tl_tiled_ws_applied": 1})
        Q = T.match_buffer(Q_handle, (1, 1, 256, 128), "float16", strides=(32768, 32768, 128, 1))
        K = T.match_buffer(K_handle, (1, 1, 256, 128), "float16", strides=(32768, 32768, 128, 1))
        V = T.match_buffer(V_handle, (1, 1, 256, 128), "float16", strides=(32768, 32768, 128, 1))
        Output = T.match_buffer(Output_handle, (1, 1, 256, 128), "float16", strides=(32768, 32768, 128, 1))
        Sinks = T.match_buffer(Sinks_handle, (1,), "float16", strides=(1,))
        bx = T.launch_thread("blockIdx.x", 2)
        by = T.launch_thread("blockIdx.y", 1)
        bz = T.launch_thread("blockIdx.z", 1)
        tx = T.launch_thread("threadIdx.x", 384)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        mbarrier = T.alloc_buffer((9,), "uint64", scope="shared.barrier")
        mbarrier_1 = T.decl_buffer((9,), "uint64", data=mbarrier.data, scope="shared.barrier")
        Q_shared = T.alloc_buffer((16384,), "float16", scope="shared.dyn")
        Q_shared_1 = T.decl_buffer((16384,), "float16", data=Q_shared.data, scope="shared.dyn")
        K_shared = T.alloc_buffer((32768,), "float16", scope="shared.dyn")
        K_shared_1 = T.decl_buffer((32768,), "float16", data=K_shared.data, scope="shared.dyn")
        V_shared = T.alloc_buffer((32768,), "float16", scope="shared.dyn")
        V_shared_1 = T.decl_buffer((32768,), "float16", data=V_shared.data, scope="shared.dyn")
        O_shared = T.alloc_buffer((16384,), "float16", scope="shared.dyn")
        O_shared_1 = T.decl_buffer((16384,), "float16", data=O_shared.data, scope="shared.dyn")
        acc_o = T.alloc_buffer((64,), scope="local")
        acc_o_1 = T.decl_buffer((64,), data=acc_o.data, scope="local")
        scores_max = T.alloc_buffer((2,), scope="local")
        scores_max_1 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        logsum = T.alloc_buffer((2,), scope="local")
        logsum_1 = T.decl_buffer((2,), data=logsum.data, scope="local")
        sinks = T.alloc_buffer((2,), "float16", scope="local")
        sinks_1 = T.decl_buffer((2,), "float16", data=sinks.data, scope="local")
        if T.tl_shuffle_elect(0):
            T.ptx_init_barrier_thread_count(mbarrier_1[0], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[1], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[2], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[3], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[4], 256)
            T.ptx_init_barrier_thread_count(mbarrier_1[5], 256)
            T.ptx_init_barrier_thread_count(mbarrier_1[6], 256)
            T.ptx_init_barrier_thread_count(mbarrier_1[7], 256)
            T.ptx_init_barrier_thread_count(mbarrier_1[8], 1)
        T.ptx_fence_barrier_init()
        T.tvm_storage_sync("shared")
        if T.tl_shuffle_elect(384):
            T.ptx_arrive_barrier_expect_tx(mbarrier_1[8], 32768)
            for i in T.unroll(2):
                T.tma_load(T.create_tma_descriptor(6, 4, Q.data, 128, 256, 1, 1, 2, 256, 65536, 65536, 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), mbarrier_1[8], T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, i * 8192, 8192, 2), i * 64, bx * 128, 0, 0, 0)
        T.attr([128, 256], "kWarpSpecializationScope", 0)
        if tx < 128:
            for k in range(bx + 1):
                T.mbarrier_wait_parity(mbarrier_1[k + 4], 1)
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(mbarrier_1[k], 32768)
                    for i in T.unroll(2):
                        T.tma_load(T.create_tma_descriptor(6, 4, K.data, 128, 256, 1, 1, 2, 256, 65536, 65536, 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), mbarrier_1[k], T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, k * 16384 + i * 8192, 8192, 2), i * 64, k * 128, 0, 0, 0)
                T.mbarrier_wait_parity(mbarrier_1[k + 6], 1)
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(mbarrier_1[k + 2], 32768)
                    for i in T.unroll(2):
                        T.tma_load(T.create_tma_descriptor(6, 4, V.data, 128, 256, 1, 1, 2, 256, 65536, 65536, 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), mbarrier_1[k + 2], T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, k * 16384 + i * 8192, 8192, 2), i * 64, k * 128, 0, 0, 0)
        else:
            i = T.int32()
            with T.attr(i, "pragma_unroll_explicit", T.bool(False)):
                for i in T.unroll(16):
                    acc_o_1[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
            logsum_1[0:2] = T.Broadcast(T.float32(0.0), 2)
            scores_max_1[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
            Sinks_1 = T.Buffer((1,), "float16", data=Sinks.data)
            sinks_1[0:2] = T.Broadcast(Sinks_1[0], 2)
            for k in range(bx + 1):
                acc_s = T.alloc_buffer((64,), scope="local")
                acc_s_1 = T.decl_buffer((64,), data=acc_s.data, scope="local")
                acc_s_cast = T.alloc_buffer((64,), "float16", scope="local")
                acc_s_cast_1 = T.decl_buffer((64,), "float16", data=acc_s_cast.data, scope="local")
                scores_max_prev = T.alloc_buffer((2,), scope="local")
                scores_max_prev_1 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
                scores_scale = T.alloc_buffer((2,), scope="local")
                scores_scale_1 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
                scores_sum = T.alloc_buffer((2,), scope="local")
                scores_sum_1 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
                i_1 = T.int32()
                with T.attr(i_1, "pragma_unroll_explicit", T.bool(False)):
                    for i_1 in T.unroll(64):
                        acc_s_1[i_1] = T.if_then_else(k * 128 + i_1 // 4 * 8 + tx % 4 * 2 + i_1 % 2 + 64 <= bx * 128 + tx // 32 * 16 + i_1 % 4 // 2 * 8 + tx % 32 // 4, T.float32(0.0), T.infinity("float32") * T.float32(-1.0))
                if k == 0:
                    T.mbarrier_wait_parity(mbarrier_1[8], 0)
                T.mbarrier_wait_parity(mbarrier_1[k], 0)
                with T.attr(0, "lexical_alloc_scope", 1):
                    desc_a = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    desc_a_1 = T.decl_buffer((1,), "uint64", data=desc_a.data, scope="local.descriptor.wgmma")
                    desc_b = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    desc_b_1 = T.decl_buffer((1,), "uint64", data=desc_b.data, scope="local.descriptor.wgmma")
                    T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, 0, 16384, 1), 1, 1, 64)
                    T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, 0, 32768, 1), 1, 1, 64)
                    T.increase_descriptor_offset(desc_b_1[0], k * 32768)
                    T.warpgroup_fence_operand("float32", acc_s.data, 0, 128)
                    T.warpgroup_arrive()
                    ki = T.int32()
                    with T.attr(ki, "pragma_unroll_explicit", T.bool(False)):
                        for ki in T.unroll(8):
                            T.ptx_wgmma_ss("m64n128k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a.data, T.shift_right(ki // 4 * 16384 + (tx // 32 + 4) % 8 // 4 * 8192 + ki % 4 * 32, 4), desc_b.data, T.shift_right(ki // 4 * 16384 + ki % 4 * 32, 4), acc_s.data, 0, 1, 1, 1)
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.warpgroup_fence_operand("float32", acc_s.data, 0, 128)
                T.ptx_arrive_barrier(mbarrier_1[k + 4])
                scores_max_prev_1[0:2] = scores_max_1[0:2]
                scores_max_1[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
                scores_max_clear = T.alloc_buffer((2,), scope="local")
                i_2 = T.int32()
                with T.attr(i_2, "pragma_unroll_explicit", T.bool(False)):
                    for i_2 in T.unroll(2):
                        scores_max_clear[i_2] = T.float32("-inf")
                        rv = T.int32()
                        with T.attr(rv, "pragma_unroll_explicit", T.bool(False)):
                            for rv in T.unroll(32):
                                scores_max_clear[i_2] = T.max(scores_max_clear[i_2], acc_s_1[rv % 16 * 4 + i_2 * 2 + rv // 16])
                        scores_max_clear[i_2] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<256>>::run", scores_max_clear[i_2])
                        scores_max_1[i_2] = T.max(scores_max_1[i_2], scores_max_clear[i_2])
                i_3 = T.int32()
                with T.attr(i_3, "pragma_unroll_explicit", T.bool(False)):
                    for i_3 in T.unroll(2):
                        scores_max_1[i_3] = T.max(scores_max_1[i_3], scores_max_prev_1[i_3])
                i_4 = T.int32()
                with T.attr(i_4, "pragma_unroll_explicit", T.bool(False)):
                    for i_4 in T.unroll(2):
                        scores_scale_1[i_4] = T.exp2(scores_max_prev_1[i_4] * T.float32(0.1275174307460247) - scores_max_1[i_4] * T.float32(0.1275174307460247))
                i_5 = T.int32()
                with T.attr(i_5, "pragma_unroll_explicit", T.bool(False)):
                    for i_5 in T.unroll(64):
                        acc_s_1[i_5] = T.exp2(acc_s_1[i_5] * T.float32(0.1275174307460247) - scores_max_1[i_5 % 4 // 2] * T.float32(0.1275174307460247))
                i_6 = T.int32()
                with T.attr(i_6, "pragma_unroll_explicit", T.bool(False)):
                    for i_6 in T.unroll(2):
                        scores_sum_1[i_6] = T.float32(0.0)
                        rv = T.int32()
                        with T.attr(rv, "pragma_unroll_explicit", T.bool(False)):
                            for rv in T.unroll(32):
                                scores_sum_1[i_6] = scores_sum_1[i_6] + acc_s_1[rv % 16 * 4 + i_6 * 2 + rv // 16]
                        scores_sum_1[i_6] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<256>>::run", scores_sum_1[i_6])
                i_7 = T.int32()
                with T.attr(i_7, "pragma_unroll_explicit", T.bool(False)):
                    for i_7 in T.unroll(2):
                        logsum_1[i_7] = logsum_1[i_7] * scores_scale_1[i_7] + scores_sum_1[i_7]
                i_8 = T.int32()
                with T.attr(i_8, "pragma_unroll_explicit", T.bool(False)):
                    for i_8 in T.unroll(16):
                        acc_s_cast_1[i_8 * 4:i_8 * 4 + 4] = T.Cast("float16x4", acc_s_1[i_8 * 4:i_8 * 4 + 4])
                i_9 = T.int32()
                with T.attr(i_9, "pragma_unroll_explicit", T.bool(False)):
                    for i_9 in T.unroll(64):
                        acc_o_1[i_9] = acc_o_1[i_9] * scores_scale_1[i_9 % 4 // 2]
                T.mbarrier_wait_parity(mbarrier_1[k + 2], 0)
                with T.attr(0, "lexical_alloc_scope", 1):
                    desc_b = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    desc_b_1 = T.decl_buffer((1,), "uint64", data=desc_b.data, scope="local.descriptor.wgmma")
                    T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, 0, 32768, 1), 1, 1024, 64)
                    T.increase_descriptor_offset(desc_b_1[0], k * 32768)
                    T.warpgroup_fence_operand("float16", acc_s_cast.data, 0, 32)
                    T.warpgroup_fence_operand("float32", acc_o.data, 0, 128)
                    T.warpgroup_arrive()
                    ki = T.int32()
                    with T.attr(ki, "pragma_unroll_explicit", T.bool(False)):
                        for ki in T.unroll(8):
                            T.ptx_wgmma_rs("m64n128k16", T.bool(False), "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, desc_b.data, T.shift_right(ki * 2048, 4), acc_o.data, 0, 1, 1, 1)
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.warpgroup_fence_operand("float32", acc_o.data, 0, 128)
                    T.warpgroup_fence_operand("float16", acc_s_cast.data, 0, 32)
                T.ptx_arrive_barrier(mbarrier_1[k + 6])
            logsum_1[0:2] = logsum_1[0:2] + T.exp2(T.Cast("float32x2", sinks_1[0:2]) * T.Broadcast(T.float32(1.44269504), 2) - scores_max_1[0:2] * T.Broadcast(T.float32(0.1275174307460247), 2))
            i_1 = T.int32()
            with T.attr(i_1, "pragma_unroll_explicit", T.bool(False)):
                for i_1 in T.unroll(64):
                    acc_o_1[i_1] = acc_o_1[i_1] / logsum_1[i_1 % 4 // 2]
            i_2 = T.int32()
            with T.attr(i_2, "pragma_unroll_explicit", T.bool(False)):
                for i_2 in T.unroll(8):
                    T.ptx_stmatrix(0, 4, T.tvm_access_ptr(T.type_annotation("float16"), O_shared.data, tx // 32 * 2048 + tx % 16 * 128 + i_2 * 16 + tx % 32 // 16 * 8 - 8192, 8, 2), T.pack_b16(T.Cast("float16", acc_o_1[i_2 * 8]), T.Cast("float16", acc_o_1[i_2 * 8 + 1])), T.pack_b16(T.Cast("float16", acc_o_1[i_2 * 8 + 2]), T.Cast("float16", acc_o_1[i_2 * 8 + 3])), T.pack_b16(T.Cast("float16", acc_o_1[i_2 * 8 + 4]), T.Cast("float16", acc_o_1[i_2 * 8 + 5])), T.pack_b16(T.Cast("float16", acc_o_1[i_2 * 8 + 6]), T.Cast("float16", acc_o_1[i_2 * 8 + 7])), "m8n8")
            if T.tl_shuffle_elect(256):
                T.tma_store(T.tvm_access_ptr(T.type_annotation("float16"), Output.data, bx * 16384, 16384, 2), T.tvm_access_ptr(T.type_annotation("float16"), O_shared.data, 0, 16384, 1), 32768, 0, 0)
                T.tma_store_arrive()
                T.tma_store_wait(0, T.bool(True))