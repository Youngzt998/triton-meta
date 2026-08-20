# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main_kernel(K_desc: T.handle("uint8x128", "grid_constant"), O_desc: T.handle("uint8x128", "grid_constant"), Q_desc: T.handle("uint8x128", "grid_constant"), V_desc: T.handle("uint8x128", "grid_constant")):
        T.func_attr({"target": T.target({"arch": "sm_90a", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tirx.is_global_func": True, "tirx.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1, 2, 3], "tl.smem_alignment_map": {"Ks": 512, "Os": 512, "Qs": 512, "Vs": 512}})
        bx = T.launch_thread("blockIdx.x", 7)
        buf_dyn_shmem = T.alloc_buffer((12288,), "uint8", scope="shared.dyn")
        Os: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        Qs: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        Ks: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 4096)
        Vs: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 8192)
        mbarrier = T.alloc_buffer((5,), "uint64", scope="shared.barrier")
        o = T.alloc_buffer((16,), scope="local")
        ls = T.alloc_buffer((2,), scope="local")
        mx = T.alloc_buffer((2,), scope="local")
        s = T.alloc_buffer((32,), scope="local")
        mp = T.alloc_buffer((2,), scope="local")
        mx_clear = T.alloc_buffer((2,), scope="local")
        ss = T.alloc_buffer((2,), scope="local")
        sc = T.alloc_buffer((32,), "bfloat16", scope="local")
        sm = T.alloc_buffer((2,), scope="local")
        by = T.launch_thread("blockIdx.y", 4)
        bz = T.launch_thread("blockIdx.z", 2)
        tx = T.launch_thread("threadIdx.x", 256)
        if T.tl_shuffle_elect(0):
            T.prefetch_tma_descriptor(Q_desc)
            T.prefetch_tma_descriptor(K_desc)
            T.prefetch_tma_descriptor(V_desc)
            T.prefetch_tma_descriptor(O_desc)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        mbarrier_1 = T.decl_buffer((5,), "uint64", data=mbarrier.data, scope="shared.barrier")
        o_1 = T.decl_buffer((16,), data=o.data, scope="local")
        mx_1 = T.decl_buffer((2,), data=mx.data, scope="local")
        ls_1 = T.decl_buffer((2,), data=ls.data, scope="local")
        if T.tl_shuffle_elect(0):
            T.ptx_init_barrier_thread_count(mbarrier_1[0], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[1], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[2], 128)
            T.ptx_init_barrier_thread_count(mbarrier_1[3], 128)
            T.ptx_init_barrier_thread_count(mbarrier_1[4], 1)
        T.ptx_fence_barrier_init()
        T.tvm_storage_sync("shared")
        if T.tl_shuffle_elect(256):
            T.ptx_arrive_barrier_expect_tx(mbarrier_1[4], 4096)
            T.tma_load(Q_desc, mbarrier_1[4], T.tvm_access_ptr(T.type_annotation("bfloat16"), Qs, 0, 2048, 2), 0, bx * 64, by, bz, 0)
        T.attr([128, 128], "kWarpSpecializationScope", 0)
        T.tvm_storage_sync("shared.dyn")
        if tx < 128:
            for k in range(T.min(2, bx - 3)):
                T.mbarrier_wait_parity(mbarrier_1[2], T.bitwise_xor(k, 1))
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(mbarrier_1[0], 4096)
                    T.tma_load(K_desc, mbarrier_1[0], T.tvm_access_ptr(T.type_annotation("bfloat16"), Ks, 0, 2048, 2), 0, k * 64, by, bz, 0)
                T.mbarrier_wait_parity(mbarrier_1[3], T.bitwise_xor(k, 1))
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(mbarrier_1[1], 4096)
                    T.tma_load(V_desc, mbarrier_1[1], T.tvm_access_ptr(T.type_annotation("bfloat16"), Vs, 0, 2048, 2), 0, k * 64, by, bz, 0)
        else:
            for i in T.unroll(4):
                o_1[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
            ls_1[0:2] = T.Broadcast(T.float32(0.0), 2)
            mx_1[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
            for k in range(T.min(2, bx - 3)):
                s_1 = T.decl_buffer((32,), data=s.data, scope="local")
                sc_1 = T.decl_buffer((32,), "bfloat16", data=sc.data, scope="local")
                mp_1 = T.decl_buffer((2,), data=mp.data, scope="local")
                ss_1 = T.decl_buffer((2,), data=ss.data, scope="local")
                sm_1 = T.decl_buffer((2,), data=sm.data, scope="local")
                for i in T.unroll(32):
                    s_1[i] = T.if_then_else(k * 64 + i // 4 * 8 + tx % 4 * 2 + i % 2 + 322 <= bx * 64 + tx // 32 * 16 + i % 4 // 2 * 8 + tx % 32 // 4, T.float32(0.0), T.infinity("float32") * T.float32(-1.0))
                if k == 0:
                    T.mbarrier_wait_parity(mbarrier_1[4], 0)
                T.mbarrier_wait_parity(mbarrier_1[0], k)
                with T.attr(0, "lexical_alloc_scope", 1):
                    desc_a = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    desc_b = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    desc_a_1 = T.decl_buffer((1,), "uint64", data=desc_a.data, scope="local.descriptor.wgmma")
                    desc_b_1 = T.decl_buffer((1,), "uint64", data=desc_b.data, scope="local.descriptor.wgmma")
                    T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("bfloat16"), Qs, 0, 2048, 1), 2, 1, 32)
                    T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("bfloat16"), Ks, 0, 2048, 1), 2, 1, 32)
                    T.warpgroup_fence_operand("float32", s.data, 0, 32)
                    T.warpgroup_arrive()
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n64k16", T.bool(True), T.bool(True), "bf16", "bf16", "fp32", desc_a.data, T.shift_right(ki * 32, 4), desc_b.data, T.shift_right(ki * 32, 4), s.data, 0, 1, 1, 1)
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.warpgroup_fence_operand("float32", s.data, 0, 32)
                T.ptx_arrive_barrier(mbarrier_1[2])
                mp_1[0:2] = mx_1[0:2]
                mx_1[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
                for i in T.unroll(2):
                    mx_clear[i] = T.float32("-inf")
                    for rv in T.unroll(16):
                        mx_clear[i] = T.max(mx_clear[i], s_1[rv % 8 * 4 + i * 2 + rv // 8])
                    mx_clear[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<128>>::run", mx_clear[i])
                    mx_1[i] = T.max(mx_1[i], mx_clear[i])
                for i in T.unroll(2):
                    ss_1[i] = T.exp2(mp_1[i] * T.float32(0.2550348614920494) - mx_1[i] * T.float32(0.2550348614920494))
                for i in T.unroll(16):
                    o_1[i] = o_1[i] * ss_1[i % 4 // 2]
                for i in T.unroll(32):
                    s_1[i] = T.exp2(s_1[i] * T.float32(0.2550348614920494) - mx_1[i % 4 // 2] * T.float32(0.2550348614920494))
                for i in T.unroll(8):
                    sc_1[i * 4:i * 4 + 4] = T.Cast("bfloat16x4", s_1[i * 4:i * 4 + 4])
                T.mbarrier_wait_parity(mbarrier_1[1], k)
                with T.attr(0, "lexical_alloc_scope", 1):
                    desc_b = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    desc_b_1 = T.decl_buffer((1,), "uint64", data=desc_b.data, scope="local.descriptor.wgmma")
                    T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("bfloat16"), Vs, 0, 2048, 1), 2, 0, 32)
                    T.warpgroup_fence_operand("bfloat16", sc.data, 0, 16)
                    T.warpgroup_fence_operand("float32", o.data, 0, 16)
                    T.warpgroup_arrive()
                    for ki in T.unroll(4):
                        T.ptx_wgmma_rs("m64n32k16", T.bool(False), "bf16", "bf16", "fp32", sc.data, ki * 8, desc_b.data, T.shift_right(ki * 1024, 4), o.data, 0, 1, 1, 1)
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.warpgroup_fence_operand("float32", o.data, 0, 16)
                    T.warpgroup_fence_operand("bfloat16", sc.data, 0, 16)
                T.ptx_arrive_barrier(mbarrier_1[3])
                for i in T.unroll(2):
                    sm_1[i] = T.float32(0.0)
                    for rv in T.unroll(16):
                        sm_1[i] = sm_1[i] + s_1[rv % 8 * 4 + i * 2 + rv // 8]
                    sm_1[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run", sm_1[i])
                for i in T.unroll(2):
                    ls_1[i] = ls_1[i] * ss_1[i] + sm_1[i]
            for i in T.unroll(16):
                o_1[i] = o_1[i] / ls_1[i % 4 // 2]
            T.tvm_storage_sync("shared.dyn", 3, 128)
            for i in T.unroll(2):
                T.ptx_stmatrix(0, 4, T.tvm_access_ptr(T.type_annotation("bfloat16"), Os, tx % 128 // 32 * 512 + tx % 16 * 32 + (tx % 8 // 4 + i) % 2 * 16 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 8, 8, 2), T.pack_b16(T.Cast("bfloat16", o_1[i * 8]), T.Cast("bfloat16", o_1[i * 8 + 1])), T.pack_b16(T.Cast("bfloat16", o_1[i * 8 + 2]), T.Cast("bfloat16", o_1[i * 8 + 3])), T.pack_b16(T.Cast("bfloat16", o_1[i * 8 + 4]), T.Cast("bfloat16", o_1[i * 8 + 5])), T.pack_b16(T.Cast("bfloat16", o_1[i * 8 + 6]), T.Cast("bfloat16", o_1[i * 8 + 7])), "m8n8")
            T.tvm_storage_sync("shared.dyn", 3, 128)
            if T.tl_shuffle_elect(128):
                T.fence_proxy_async()
                T.tma_store(O_desc, T.tvm_access_ptr(T.type_annotation("bfloat16"), Os, 0, 2048, 1), 0, bx * 64, by, bz, 0, 0)
                T.tma_store_arrive()
                T.tma_store_wait(0, T.bool(True))

    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, O_handle: T.handle):
        Q_desc = T.handle("uint8x128", "grid_constant")
        Q = T.handle("bfloat16", "global")
        K_desc = T.handle("uint8x128", "grid_constant")
        K = T.handle("bfloat16", "global")
        V_desc = T.handle("uint8x128", "grid_constant")
        V = T.handle("bfloat16", "global")
        O_desc = T.handle("uint8x128", "grid_constant")
        O = T.handle("bfloat16", "global")
        T.func_attr({"target": T.target({"arch": "sm_90a", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tirx.is_entry_func": True, "tl.has_tma": T.bool(True), "tl.readonly_param_indices": [0, 1, 2, 3], "tl_tiled_ws_applied": 1, "tma_descriptor_args": {Q_desc: ["__tvm_tensormap_create_tiled", Q_desc, 9, 4, Q, 32, 385, 4, 2, 2, 64, 24640, 98560, 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0], K_desc: ["__tvm_tensormap_create_tiled", K_desc, 9, 4, K, 32, 127, 4, 2, 2, 64, 8128, 32512, 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0], V_desc: ["__tvm_tensormap_create_tiled", V_desc, 9, 4, V, 32, 127, 4, 2, 2, 64, 8128, 32512, 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0], O_desc: ["__tvm_tensormap_create_tiled", O_desc, 9, 4, O, 32, 385, 4, 2, 2, 64, 24640, 98560, 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0]}})
        Q_1 = T.match_buffer(Q_handle, (2, 4, 385, 32), "bfloat16", data=Q, strides=(49280, 12320, 32, 1))
        K_1 = T.match_buffer(K_handle, (2, 4, 127, 32), "bfloat16", data=K, strides=(16256, 4064, 32, 1))
        V_1 = T.match_buffer(V_handle, (2, 4, 127, 32), "bfloat16", data=V, strides=(16256, 4064, 32, 1))
        O_1 = T.match_buffer(O_handle, (2, 4, 385, 32), "bfloat16", data=O, strides=(49280, 12320, 32, 1))
        Q_desc = T.tvm_stack_alloca("tvm_ffi_any", 16)
        T.call_packed("__tvm_tensormap_create_tiled", Q_desc, 9, 4, Q, 32, 385, 4, 2, 2, 64, 24640, 98560, 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0)
        K_desc = T.tvm_stack_alloca("tvm_ffi_any", 16)
        T.call_packed("__tvm_tensormap_create_tiled", K_desc, 9, 4, K, 32, 127, 4, 2, 2, 64, 8128, 32512, 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0)
        V_desc = T.tvm_stack_alloca("tvm_ffi_any", 16)
        T.call_packed("__tvm_tensormap_create_tiled", V_desc, 9, 4, V, 32, 127, 4, 2, 2, 64, 8128, 32512, 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0)
        O_desc = T.tvm_stack_alloca("tvm_ffi_any", 16)
        T.call_packed("__tvm_tensormap_create_tiled", O_desc, 9, 4, O, 32, 385, 4, 2, 2, 64, 24640, 98560, 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0)
        Module.main_kernel(K_desc, O_desc, Q_desc, V_desc)