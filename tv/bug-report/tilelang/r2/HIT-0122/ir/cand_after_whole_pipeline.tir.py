# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel(K: T.handle("bfloat16", "global"), O_desc: T.handle("uint8x128", "grid_constant"), Q: T.handle("bfloat16", "global"), V: T.handle("bfloat16", "global")):
        T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": T.int64(24576), "target": T.target({"arch": "sm_90a", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": T.int64(2), "blockIdx.y": T.int64(4), "blockIdx.z": T.int64(1), "threadIdx.x": T.int64(128), "threadIdx.y": T.int64(1), "threadIdx.z": T.int64(1)}, "tirx.is_global_func": T.bool(True), "tirx.kernel_launch_params": ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tirx.use_dyn_shared_memory"], "tirx.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1, 2, 3], "tl.smem_alignment_map": {"Os": 512}})
        V_1 = T.decl_buffer((98304,), "bfloat16", data=V)
        K_1 = T.decl_buffer((98304,), "bfloat16", data=K)
        Q_1 = T.decl_buffer((24576,), "bfloat16", data=Q)
        bx = T.launch_thread("blockIdx.x", T.int64(2))
        buf_dyn_shmem = T.alloc_buffer((T.int64(24576),), "uint8", scope="shared.dyn")
        Os: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, T.int64(0))
        Qs: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, T.int64(0))
        Ks: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, T.int64(8192))
        Vs: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, T.int64(16384))
        o = T.alloc_buffer((T.int64(32),), scope="local")
        ls = T.alloc_buffer((T.int64(4),), scope="local")
        mx = T.alloc_buffer((T.int64(4),), scope="local")
        s = T.alloc_buffer((T.int64(128),), scope="local")
        mp = T.alloc_buffer((T.int64(4),), scope="local")
        mx_clear = T.alloc_buffer((T.int64(4),), scope="local")
        ss = T.alloc_buffer((T.int64(4),), scope="local")
        sc = T.alloc_buffer((T.int64(128),), "bfloat16", scope="local")
        sm = T.alloc_buffer((T.int64(4),), scope="local")
        by = T.launch_thread("blockIdx.y", T.int64(4))
        bz = T.launch_thread("blockIdx.z", T.int64(1))
        tx = T.launch_thread("threadIdx.x", T.int64(128))
        if T.tl_shuffle_elect(0):
            T.prefetch_tma_descriptor(O_desc)
        ty = T.launch_thread("threadIdx.y", T.int64(1))
        tz = T.launch_thread("threadIdx.z", T.int64(1))
        Qs_1 = T.decl_buffer((T.int64(4096),), "bfloat16", data=Qs, scope="shared.dyn")
        o_1 = T.decl_buffer((T.int64(32),), data=o.data, scope="local")
        mx_1 = T.decl_buffer((T.int64(4),), data=mx.data, scope="local")
        ls_1 = T.decl_buffer((T.int64(4),), data=ls.data, scope="local")
        for i in T.unroll(T.int64(4)):
            for vec in range(T.int64(8)):
                Qs_1[i * T.int64(1024) + tx // T.int64(4) * T.int64(32) + (tx % T.int64(32) // T.int64(16) + tx % T.int64(4) // T.int64(2)) % T.int64(2) * T.int64(16) + (tx % T.int64(16) // T.int64(8) + tx % T.int64(2)) % T.int64(2) * T.int64(8) + vec] = T.if_then_else(bx * T.int64(2) + i // T.int64(2) < T.int64(3), Q_1[by * T.int64(6144) + bx * T.int64(4096) + i * T.int64(1024) + tx * T.int64(8) + vec], T.bfloat16(0.0))
        for i in T.unroll(T.int64(8)):
            for vec in range(T.int64(4)):
                o_1[i * T.int64(4) + vec] = T.float32(0.0)
        for i in range(T.int64(4)):
            ls_1[i] = T.float32(0.0)
        for i in range(T.int64(4)):
            mx_1[i] = T.infinity("float32") * T.float32(-1.0)
        for k in range(T.int64(6)):
            Ks_1 = T.decl_buffer((T.int64(4096),), "bfloat16", data=Ks, scope="shared.dyn")
            Vs_1 = T.decl_buffer((T.int64(4096),), "bfloat16", data=Vs, scope="shared.dyn")
            s_1 = T.decl_buffer((T.int64(128),), data=s.data, scope="local")
            sc_1 = T.decl_buffer((T.int64(128),), "bfloat16", data=sc.data, scope="local")
            mp_1 = T.decl_buffer((T.int64(4),), data=mp.data, scope="local")
            ss_1 = T.decl_buffer((T.int64(4),), data=ss.data, scope="local")
            sm_1 = T.decl_buffer((T.int64(4),), data=sm.data, scope="local")
            T.tvm_storage_sync("shared.dyn")
            for i in T.unroll(T.int64(4)):
                for vec in range(T.int64(8)):
                    Ks_1[i * T.int64(1024) + tx // T.int64(4) * T.int64(32) + (tx % T.int64(32) // T.int64(16) + tx % T.int64(4) // T.int64(2)) % T.int64(2) * T.int64(16) + (tx % T.int64(16) // T.int64(8) + tx % T.int64(2)) % T.int64(2) * T.int64(8) + vec] = K_1[by * T.int64(24576) + k * T.int64(4096) + i * T.int64(1024) + tx * T.int64(8) + vec]
            for i in T.unroll(T.int64(32)):
                for vec in range(T.int64(4)):
                    s_1[i * T.int64(4) + vec] = T.float32(0.0)
            with T.attr(0, "lexical_alloc_scope", T.int64(1)):
                desc_a = T.alloc_buffer((T.int64(1),), "uint64", scope="local.descriptor.wgmma")
                desc_b = T.alloc_buffer((T.int64(1),), "uint64", scope="local.descriptor.wgmma")
                desc_a_1 = T.decl_buffer((T.int64(1),), "uint64", data=desc_a.data, scope="local.descriptor.wgmma")
                desc_b_1 = T.decl_buffer((T.int64(1),), "uint64", data=desc_b.data, scope="local.descriptor.wgmma")
                T.tvm_storage_sync("shared.dyn")
                T.initialize_wgmma_descriptor(desc_a_1[T.int64(0)], T.tvm_access_ptr(T.type_annotation("bfloat16"), Qs, T.int64(0), T.int64(4096), T.int64(1)), T.int64(2), T.int64(1), T.int64(32))
                T.initialize_wgmma_descriptor(desc_b_1[T.int64(0)], T.tvm_access_ptr(T.type_annotation("bfloat16"), Ks, T.int64(0), T.int64(4096), T.int64(1)), T.int64(2), T.int64(1), T.int64(32))
                T.warpgroup_fence_operand("float32", s.data, T.int64(0), T.int64(128))
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(T.int64(2)):
                    for ki in T.unroll(T.int64(2)):
                        T.ptx_wgmma_ss("m64n128k16", T.bool(True), T.bool(True), "bf16", "bf16", "fp32", desc_a.data, T.shift_right(i * T.int64(4096) + ki * T.int64(32), T.int64(4)), desc_b.data, T.shift_right(ki * T.int64(32), T.int64(4)), s.data, i * T.int64(64), T.int64(1), T.int64(1), T.int64(1))
                T.warpgroup_commit_batch()
                T.warpgroup_wait(T.int64(0))
                T.warpgroup_fence_operand("float32", s.data, T.int64(0), T.int64(128))
            T.tvm_storage_sync("shared.dyn")
            for i in T.unroll(T.int64(4)):
                for vec in range(T.int64(8)):
                    Vs_1[i * T.int64(1024) + tx // T.int64(4) * T.int64(32) + (tx % T.int64(32) // T.int64(16) + tx % T.int64(4) // T.int64(2)) % T.int64(2) * T.int64(16) + (tx % T.int64(16) // T.int64(8) + tx % T.int64(2)) % T.int64(2) * T.int64(8) + vec] = V_1[by * T.int64(24576) + k * T.int64(4096) + i * T.int64(1024) + tx * T.int64(8) + vec]
            for i in range(T.int64(4)):
                mp_1[i] = mx_1[i]
            for i in range(T.int64(4)):
                mx_1[i] = T.infinity("float32") * T.float32(-1.0)
            for i in T.unroll(T.int64(4)):
                mx_clear[i] = T.float32("-inf")
                for rv in T.unroll(T.int64(32)):
                    mx_clear[i] = T.max(mx_clear[i], s_1[i // T.int64(2) * T.int64(64) + rv % T.int64(16) * T.int64(4) + i % T.int64(2) * T.int64(2) + rv // T.int64(16)])
                mx_clear[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run", mx_clear[i])
                mx_1[i] = T.max(mx_1[i], mx_clear[i])
            for i in T.unroll(T.int64(4)):
                ss_1[i] = T.exp2(mp_1[i] * T.float32(0.2550348614920494) - mx_1[i] * T.float32(0.2550348614920494))
            for i in T.unroll(T.int64(32)):
                o_1[i] = o_1[i] * ss_1[i // T.int64(16) * T.int64(2) + i % T.int64(4) // T.int64(2)]
            for i in T.unroll(T.int64(128)):
                s_1[i] = T.exp2(s_1[i] * T.float32(0.2550348614920494) - mx_1[i // T.int64(64) * T.int64(2) + i % T.int64(4) // T.int64(2)] * T.float32(0.2550348614920494))
            for i in T.unroll(T.int64(32)):
                for vec in range(T.int64(4)):
                    sc_1[i * T.int64(4) + vec] = T.Cast("bfloat16", s_1[i % T.int64(4) // T.int64(2) * T.int64(64) + i // T.int64(4) * T.int64(8) + i % T.int64(2) * T.int64(4) + vec])
            with T.attr(0, "lexical_alloc_scope", T.int64(1)):
                desc_b = T.alloc_buffer((T.int64(1),), "uint64", scope="local.descriptor.wgmma")
                desc_b_1 = T.decl_buffer((T.int64(1),), "uint64", data=desc_b.data, scope="local.descriptor.wgmma")
                T.tvm_storage_sync("shared.dyn")
                T.initialize_wgmma_descriptor(desc_b_1[T.int64(0)], T.tvm_access_ptr(T.type_annotation("bfloat16"), Vs, T.int64(0), T.int64(4096), T.int64(1)), T.int64(2), T.int64(0), T.int64(32))
                T.warpgroup_fence_operand("bfloat16", sc.data, T.int64(0), T.int64(64))
                T.warpgroup_fence_operand("float32", o.data, T.int64(0), T.int64(32))
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(T.int64(2)):
                    for ki in T.unroll(T.int64(8)):
                        T.ptx_wgmma_rs("m64n32k16", T.bool(False), "bf16", "bf16", "fp32", sc.data, ki * T.int64(16) + i * T.int64(8), desc_b.data, T.shift_right(ki * T.int64(1024), T.int64(4)), o.data, i * T.int64(16), T.int64(1), T.int64(1), T.int64(1))
                T.warpgroup_commit_batch()
                T.warpgroup_wait(T.int64(0))
                T.warpgroup_fence_operand("float32", o.data, T.int64(0), T.int64(32))
                T.warpgroup_fence_operand("bfloat16", sc.data, T.int64(0), T.int64(64))
            for i in T.unroll(T.int64(4)):
                sm_1[i] = T.float32(0.0)
                for rv in T.unroll(T.int64(32)):
                    sm_1[i] = sm_1[i] + s_1[i // T.int64(2) * T.int64(64) + rv % T.int64(16) * T.int64(4) + i % T.int64(2) * T.int64(2) + rv // T.int64(16)]
                sm_1[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run", sm_1[i])
            for i in T.unroll(T.int64(4)):
                ls_1[i] = ls_1[i] * ss_1[i] + sm_1[i]
        for i in T.unroll(T.int64(32)):
            o_1[i] = o_1[i] / ls_1[i // T.int64(16) * T.int64(2) + i % T.int64(4) // T.int64(2)]
        T.tvm_storage_sync("shared.dyn")
        for i in T.unroll(T.int64(4)):
            T.ptx_stmatrix(T.int64(0), T.int64(4), T.tvm_access_ptr(T.type_annotation("bfloat16"), Os, i // T.int64(2) * T.int64(2048) + tx // T.int64(32) * T.int64(512) + tx % T.int64(16) * T.int64(32) + (tx % T.int64(8) // T.int64(4) + i % T.int64(2)) % T.int64(2) * T.int64(16) + (tx % T.int64(32) // T.int64(16) + tx % T.int64(4) // T.int64(2)) % T.int64(2) * T.int64(8), T.int64(8), T.int64(2)), T.pack_b16(T.Cast("bfloat16", o_1[i * T.int64(8)]), T.Cast("bfloat16", o_1[i * T.int64(8) + T.int64(1)])), T.pack_b16(T.Cast("bfloat16", o_1[i * T.int64(8) + T.int64(2)]), T.Cast("bfloat16", o_1[i * T.int64(8) + T.int64(3)])), T.pack_b16(T.Cast("bfloat16", o_1[i * T.int64(8) + T.int64(4)]), T.Cast("bfloat16", o_1[i * T.int64(8) + T.int64(5)])), T.pack_b16(T.Cast("bfloat16", o_1[i * T.int64(8) + T.int64(6)]), T.Cast("bfloat16", o_1[i * T.int64(8) + T.int64(7)])), "m8n8")
        T.tvm_storage_sync("shared.dyn")
        if T.tl_shuffle_elect(T.int64(128)):
            T.fence_proxy_async()
            T.tma_store(O_desc, T.tvm_access_ptr(T.type_annotation("bfloat16"), Os, T.int64(0), T.int64(4096), T.int64(1)), T.int64(0), bx * T.int64(128), by, T.int64(0), T.int64(0), T.int64(0))
            T.tma_store_arrive()
            T.tma_store_wait(T.int64(0), T.bool(True))

    @T.prim_func
    def main(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle("void", "global")) -> T.int32:
        O_desc = T.handle("uint8x128", "grid_constant")
        O = T.handle("bfloat16", "global")
        T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_main", "target": T.target({"keys": ["cpu"], "kind": "c", "tag": ""}), "thread_extent": {}, "tirx.is_entry_func": True, "tl.has_tma": T.bool(True), "tl.readonly_param_indices": [0, 1, 2, 3], "tma_descriptor_args": {O_desc: ["__tvm_tensormap_create_tiled", O_desc, T.int64(9), T.int64(4), O, T.int64(32), T.int64(192), T.int64(4), T.int64(1), T.int64(2), T.int64(64), T.int64(12288), T.int64(49152), T.int64(32), T.int64(128), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(0), T.int64(2), T.int64(2), T.int64(0)]}})
        assert num_args == 4, ("RuntimeError", ["main: num_args should be 4"])
        assert not T.isnullptr(args), ("RuntimeError", ["main: args pointer is NULL"])
        Q_handle_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
        assert Q_handle_type_index == 0 or Q_handle_type_index == 4 or Q_handle_type_index == 7 or 64 <= Q_handle_type_index, ("RuntimeError", ["kernel main input Q expected pointer or tensor handle"])
        K_handle_type_index: T.int32 = T.tvm_struct_get(args, 1, 13, "int32")
        assert K_handle_type_index == 0 or K_handle_type_index == 4 or K_handle_type_index == 7 or 64 <= K_handle_type_index, ("RuntimeError", ["kernel main input K expected pointer or tensor handle"])
        V_handle_type_index: T.int32 = T.tvm_struct_get(args, 2, 13, "int32")
        assert V_handle_type_index == 0 or V_handle_type_index == 4 or V_handle_type_index == 7 or 64 <= V_handle_type_index, ("RuntimeError", ["kernel main input V expected pointer or tensor handle"])
        O_handle_type_index: T.int32 = T.tvm_struct_get(args, 3, 13, "int32")
        assert O_handle_type_index == 0 or O_handle_type_index == 4 or O_handle_type_index == 7 or 64 <= O_handle_type_index, ("RuntimeError", ["kernel main input O expected pointer or tensor handle"])
        Q_handle: T.handle = T.Select(Q_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 0, 15, "handle"), 24), T.tvm_struct_get(args, 0, 15, "handle"))
        K_handle: T.handle = T.Select(K_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 1, 15, "handle"), 24), T.tvm_struct_get(args, 1, 15, "handle"))
        V_handle: T.handle = T.Select(V_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 2, 15, "handle"), 24), T.tvm_struct_get(args, 2, 15, "handle"))
        O_handle: T.handle = T.Select(O_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 3, 15, "handle"), 24), T.tvm_struct_get(args, 3, 15, "handle"))
        main_Q_is_null: T.bool = T.isnullptr(Q_handle)
        assert not main_Q_is_null, ("RuntimeError", ["main.Q is expected to have non-NULL pointer"])
        main_K_is_null: T.bool = T.isnullptr(K_handle)
        assert not main_K_is_null, ("RuntimeError", ["main.K is expected to have non-NULL pointer"])
        main_V_is_null: T.bool = T.isnullptr(V_handle)
        assert not main_V_is_null, ("RuntimeError", ["main.V is expected to have non-NULL pointer"])
        main_O_is_null: T.bool = T.isnullptr(O_handle)
        assert not main_O_is_null, ("RuntimeError", ["main.O is expected to have non-NULL pointer"])
        main_Q_shape: T.handle("int64", "global") = T.tvm_struct_get(Q_handle, 0, 2, "handle")
        main_Q_shape_1 = T.decl_buffer((4,), "int64", data=main_Q_shape)
        main_K_shape: T.handle("int64", "global") = T.tvm_struct_get(K_handle, 0, 2, "handle")
        main_K_shape_1 = T.decl_buffer((4,), "int64", data=main_K_shape)
        main_V_shape: T.handle("int64", "global") = T.tvm_struct_get(V_handle, 0, 2, "handle")
        main_V_shape_1 = T.decl_buffer((4,), "int64", data=main_V_shape)
        main_O_shape: T.handle("int64", "global") = T.tvm_struct_get(O_handle, 0, 2, "handle")
        main_O_shape_1 = T.decl_buffer((4,), "int64", data=main_O_shape)
        assert T.tvm_struct_get(Q_handle, 0, 4, "int32") == 4, ("RuntimeError", ["kernel main input Q ndim mismatch, expected 4"])
        main_Q_strides: T.handle("int64", "global") = T.tvm_struct_get(Q_handle, 0, 3, "handle")
        main_Q_strides_1 = T.decl_buffer((4,), "int64", data=main_Q_strides)
        dev_id: T.int32 = T.tvm_struct_get(Q_handle, 0, 9, "int32")
        Q: T.handle("bfloat16", "global") = T.tvm_struct_get(Q_handle, 0, 1, "handle")
        T.attr(Q, "storage_alignment", 64)
        assert T.tvm_struct_get(K_handle, 0, 4, "int32") == 4, ("RuntimeError", ["kernel main input K ndim mismatch, expected 4"])
        main_K_strides: T.handle("int64", "global") = T.tvm_struct_get(K_handle, 0, 3, "handle")
        main_K_strides_1 = T.decl_buffer((4,), "int64", data=main_K_strides)
        K: T.handle("bfloat16", "global") = T.tvm_struct_get(K_handle, 0, 1, "handle")
        T.attr(K, "storage_alignment", 64)
        assert T.tvm_struct_get(V_handle, 0, 4, "int32") == 4, ("RuntimeError", ["kernel main input V ndim mismatch, expected 4"])
        main_V_strides: T.handle("int64", "global") = T.tvm_struct_get(V_handle, 0, 3, "handle")
        main_V_strides_1 = T.decl_buffer((4,), "int64", data=main_V_strides)
        V: T.handle("bfloat16", "global") = T.tvm_struct_get(V_handle, 0, 1, "handle")
        T.attr(V, "storage_alignment", 64)
        assert T.tvm_struct_get(O_handle, 0, 4, "int32") == 4, ("RuntimeError", ["kernel main input O ndim mismatch, expected 4"])
        main_O_strides: T.handle("int64", "global") = T.tvm_struct_get(O_handle, 0, 3, "handle")
        main_O_strides_1 = T.decl_buffer((4,), "int64", data=main_O_strides)
        O = T.tvm_struct_get(O_handle, 0, 1, "handle")
        T.attr(O, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        assert T.tvm_struct_get(Q_handle, 0, 5, "uint8") == T.uint8(4) and T.tvm_struct_get(Q_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(Q_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input Q dtype mismatch, expected bfloat16"])
        assert T.Cast("int32", main_Q_shape_1[0]) == 1, ("RuntimeError", ["kernel main input Q shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_Q_shape_1[1]) == 4, ("RuntimeError", ["kernel main input Q shape[1] violates packed ABI constraint"])
        assert T.Cast("int32", main_Q_shape_1[2]) == 192, ("RuntimeError", ["kernel main input Q shape[2] violates packed ABI constraint"])
        assert T.Cast("int32", main_Q_shape_1[3]) == 32, ("RuntimeError", ["kernel main input Q shape[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Q_strides), 1, T.Cast("int32", main_Q_strides_1[3])) == 1, ("RuntimeError", ["kernel main input Q strides[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Q_strides), 1, T.Cast("int32", main_Q_strides_1[2])) == 32, ("RuntimeError", ["kernel main input Q strides[2] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Q_strides), 1, T.Cast("int32", main_Q_strides_1[1])) == 6144, ("RuntimeError", ["kernel main input Q strides[1] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(Q_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input Q byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(Q_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input Q device_type mismatch, expected cuda"])
        assert not T.isnullptr(Q), ("RuntimeError", ["kernel main input Q data pointer is NULL"])
        assert T.tvm_struct_get(K_handle, 0, 5, "uint8") == T.uint8(4) and T.tvm_struct_get(K_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(K_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input K dtype mismatch, expected bfloat16"])
        assert T.Cast("int32", main_K_shape_1[0]) == 1, ("RuntimeError", ["kernel main input K shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_K_shape_1[1]) == 4, ("RuntimeError", ["kernel main input K shape[1] violates packed ABI constraint"])
        assert T.Cast("int32", main_K_shape_1[2]) == 768, ("RuntimeError", ["kernel main input K shape[2] violates packed ABI constraint"])
        assert T.Cast("int32", main_K_shape_1[3]) == 32, ("RuntimeError", ["kernel main input K shape[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_K_strides), 1, T.Cast("int32", main_K_strides_1[3])) == 1, ("RuntimeError", ["kernel main input K strides[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_K_strides), 1, T.Cast("int32", main_K_strides_1[2])) == 32, ("RuntimeError", ["kernel main input K strides[2] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_K_strides), 1, T.Cast("int32", main_K_strides_1[1])) == 24576, ("RuntimeError", ["kernel main input K strides[1] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(K_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input K byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(K_handle, 0, 9, "int32") == T.tvm_struct_get(Q_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input K device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(K_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input K device_type mismatch, expected cuda"])
        assert not T.isnullptr(K), ("RuntimeError", ["kernel main input K data pointer is NULL"])
        assert T.tvm_struct_get(V_handle, 0, 5, "uint8") == T.uint8(4) and T.tvm_struct_get(V_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(V_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input V dtype mismatch, expected bfloat16"])
        assert T.Cast("int32", main_V_shape_1[0]) == 1, ("RuntimeError", ["kernel main input V shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_V_shape_1[1]) == 4, ("RuntimeError", ["kernel main input V shape[1] violates packed ABI constraint"])
        assert T.Cast("int32", main_V_shape_1[2]) == 768, ("RuntimeError", ["kernel main input V shape[2] violates packed ABI constraint"])
        assert T.Cast("int32", main_V_shape_1[3]) == 32, ("RuntimeError", ["kernel main input V shape[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_V_strides), 1, T.Cast("int32", main_V_strides_1[3])) == 1, ("RuntimeError", ["kernel main input V strides[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_V_strides), 1, T.Cast("int32", main_V_strides_1[2])) == 32, ("RuntimeError", ["kernel main input V strides[2] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_V_strides), 1, T.Cast("int32", main_V_strides_1[1])) == 24576, ("RuntimeError", ["kernel main input V strides[1] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(V_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input V byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(V_handle, 0, 9, "int32") == T.tvm_struct_get(Q_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input V device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(V_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input V device_type mismatch, expected cuda"])
        assert not T.isnullptr(V), ("RuntimeError", ["kernel main input V data pointer is NULL"])
        assert T.tvm_struct_get(O_handle, 0, 5, "uint8") == T.uint8(4) and T.tvm_struct_get(O_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(O_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input O dtype mismatch, expected bfloat16"])
        assert T.Cast("int32", main_O_shape_1[0]) == 1, ("RuntimeError", ["kernel main input O shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_O_shape_1[1]) == 4, ("RuntimeError", ["kernel main input O shape[1] violates packed ABI constraint"])
        assert T.Cast("int32", main_O_shape_1[2]) == 192, ("RuntimeError", ["kernel main input O shape[2] violates packed ABI constraint"])
        assert T.Cast("int32", main_O_shape_1[3]) == 32, ("RuntimeError", ["kernel main input O shape[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_O_strides), 1, T.Cast("int32", main_O_strides_1[3])) == 1, ("RuntimeError", ["kernel main input O strides[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_O_strides), 1, T.Cast("int32", main_O_strides_1[2])) == 32, ("RuntimeError", ["kernel main input O strides[2] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_O_strides), 1, T.Cast("int32", main_O_strides_1[1])) == 6144, ("RuntimeError", ["kernel main input O strides[1] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(O_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input O byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(O_handle, 0, 9, "int32") == T.tvm_struct_get(Q_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input O device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(O_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input O device_type mismatch, expected cuda"])
        assert not T.isnullptr(O), ("RuntimeError", ["kernel main input O data pointer is NULL"])
        Q_1 = T.decl_buffer((1, 4, 192, 32), "bfloat16", data=Q, strides=(24576, 6144, 32, 1))
        K_1 = T.decl_buffer((1, 4, 768, 32), "bfloat16", data=K, strides=(98304, 24576, 32, 1))
        V_1 = T.decl_buffer((1, 4, 768, 32), "bfloat16", data=V, strides=(98304, 24576, 32, 1))
        O_1 = T.decl_buffer((1, 4, 192, 32), "bfloat16", data=O, strides=(24576, 6144, 32, 1))
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "main_compute_"):
            O_desc = T.tvm_stack_alloca("tvm_ffi_any", 16)
            T.call_packed("__tvm_tensormap_create_tiled", O_desc, T.int64(9), T.int64(4), O, T.int64(32), T.int64(192), T.int64(4), T.int64(1), T.int64(2), T.int64(64), T.int64(12288), T.int64(49152), T.int64(32), T.int64(128), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(0), T.int64(2), T.int64(2), T.int64(0))
            T.call_packed("main_kernel", K, O_desc, Q, V, T.int64(2), T.int64(4), T.int64(1), T.int64(128), T.int64(1), T.int64(1), T.int64(24576))
        return 0