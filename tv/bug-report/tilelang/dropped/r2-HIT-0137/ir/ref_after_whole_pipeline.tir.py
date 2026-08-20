# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel(A_desc: T.handle("uint8x128", "grid_constant"), X_desc: T.handle("uint8x128", "grid_constant"), Y: T.handle("float32", "global")):
        T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 41600, "target": T.target({"arch": "sm_90a", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 32, "threadIdx.x": 160, "threadIdx.y": 1, "threadIdx.z": 1}, "tirx.is_global_func": T.bool(True), "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tirx.use_dyn_shared_memory"], "tirx.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1], "tl.smem_alignment_map": {"As": 128, "Xs": 128}})
        Y_1 = T.decl_buffer((2037,), data=Y)
        bx = T.launch_thread("blockIdx.x", 32)
        buf_dyn_shmem = T.alloc_buffer((41600,), "uint8", scope="shared.dyn")
        As: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        Xs: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 40960)
        mbarrier = T.alloc_buffer((10,), "uint64", scope="shared.barrier")
        acc = T.alloc_buffer((16,), scope="local")
        p = T.alloc_buffer((128,), scope="local")
        As_local_cast = T.alloc_buffer((8,), "bfloat16", scope="local")
        Xs_local_cast_1 = T.alloc_buffer((8,), "bfloat16", scope="local")
        acc_clear = T.alloc_buffer((16,), scope="local")
        tx = T.launch_thread("threadIdx.x", 160)
        if T.tl_shuffle_elect(0):
            T.prefetch_tma_descriptor(A_desc)
            T.prefetch_tma_descriptor(X_desc)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        mbarrier_1 = T.decl_buffer((10,), "uint64", data=mbarrier.data, scope="shared.barrier")
        As_1 = T.decl_buffer((20480,), "bfloat16", data=As, scope="shared.dyn")
        Xs_1 = T.decl_buffer((320,), "bfloat16", data=Xs, scope="shared.dyn")
        acc_1 = T.decl_buffer((16,), data=acc.data, scope="local")
        if T.tl_shuffle_elect(0):
            T.ptx_init_barrier_thread_count(mbarrier_1[0], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[1], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[2], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[3], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[4], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[5], 32)
            T.ptx_init_barrier_thread_count(mbarrier_1[6], 32)
            T.ptx_init_barrier_thread_count(mbarrier_1[7], 32)
            T.ptx_init_barrier_thread_count(mbarrier_1[8], 32)
            T.ptx_init_barrier_thread_count(mbarrier_1[9], 32)
        T.ptx_fence_barrier_init()
        T.tvm_storage_sync("shared")
        T.attr([128, 32], "kWarpSpecializationScope", 0)
        if tx < 128:
            T.set_max_nreg(24, 0)
            for ko in range(4):
                T.mbarrier_wait_parity(mbarrier_1[ko + 5], 1)
                if T.tl_shuffle_elect(128):
                    T.mbarrier_expect_tx(mbarrier_1[ko], 8192)
                    T.tma_load(A_desc, mbarrier_1[ko], T.tvm_access_ptr(T.type_annotation("bfloat16"), As, ko * 4096, 4096, 2), ko * 64, bx * 64, 0)
                    T.ptx_arrive_barrier_expect_tx(mbarrier_1[ko], 128)
                    T.tma_load(X_desc, mbarrier_1[ko], T.tvm_access_ptr(T.type_annotation("bfloat16"), Xs, ko * 64, 64, 2), ko * 64, 0)
        else:
            T.set_max_nreg(240, 1)
            for i in T.unroll(4):
                acc_1[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
            for ko in range(4):
                p_1 = T.decl_buffer((128,), data=p.data, scope="local")
                T.mbarrier_wait_parity(mbarrier_1[ko], 0)
                for i in T.unroll(16):
                    As_local_cast[0:8] = As_1[ko * 4096 + i * 256 + tx * 8 - 1024:ko * 4096 + i * 256 + tx * 8 - 1024 + 8]
                    Xs_local_cast_1[0:8] = Xs_1[ko * 64 + tx % 8 * 8:ko * 64 + tx % 8 * 8 + 8]
                    for vec in range(2):
                        p_1[i * 8 + vec * 4:i * 8 + vec * 4 + 4] = T.Cast("float32x4", As_local_cast[vec * 4:vec * 4 + 4]) * T.Cast("float32x4", Xs_local_cast_1[vec * 4:vec * 4 + 4])
                T.ptx_arrive_barrier(mbarrier_1[ko + 5])
                for i in T.unroll(16):
                    acc_clear[i] = T.float32(0.0)
                    for rv in T.unroll(8):
                        acc_clear[i] = acc_clear[i] + p_1[i * 8 + rv]
                    acc_clear[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 8, 1, 128, tl::NamedBarrier<32>>::run", acc_clear[i])
                    acc_1[i] = acc_1[i] + acc_clear[i]
            if tx % 8 == 0:
                for i in T.unroll(16):
                    if bx * 64 + i * 4 + tx // 8 < 2053:
                        Y_1[bx * 64 + i * 4 + tx // 8 - 16] = acc_1[i]

    @T.prim_func
    def main(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle("void", "global")) -> T.int32:
        A_desc = T.handle("uint8x128", "grid_constant")
        A = T.handle("bfloat16", "global")
        X_desc = T.handle("uint8x128", "grid_constant")
        X = T.handle("bfloat16", "global")
        T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_main", "target": T.target({"keys": ["cpu"], "kind": "c", "tag": ""}), "thread_extent": {}, "tirx.is_entry_func": True, "tl.has_tma": T.bool(True), "tl.readonly_param_indices": [0, 1, 2], "tl_tiled_ws_applied": 1, "tma_descriptor_args": {A_desc: ["__tvm_tensormap_create_tiled", A_desc, 9, 2, A, 224, 2037, 2, 448, 64, 64, 1, 1, 0, 0, 2, 0], X_desc: ["__tvm_tensormap_create_tiled", X_desc, 9, 1, X, 224, 2, 64, 1, 0, 0, 2, 0]}})
        assert num_args == 3, ("RuntimeError", ["main: num_args should be 3"])
        assert not T.isnullptr(args), ("RuntimeError", ["main: args pointer is NULL"])
        A_handle_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
        assert A_handle_type_index == 0 or A_handle_type_index == 4 or A_handle_type_index == 7 or 64 <= A_handle_type_index, ("RuntimeError", ["kernel main input A expected pointer or tensor handle"])
        X_handle_type_index: T.int32 = T.tvm_struct_get(args, 1, 13, "int32")
        assert X_handle_type_index == 0 or X_handle_type_index == 4 or X_handle_type_index == 7 or 64 <= X_handle_type_index, ("RuntimeError", ["kernel main input X expected pointer or tensor handle"])
        Y_handle_type_index: T.int32 = T.tvm_struct_get(args, 2, 13, "int32")
        assert Y_handle_type_index == 0 or Y_handle_type_index == 4 or Y_handle_type_index == 7 or 64 <= Y_handle_type_index, ("RuntimeError", ["kernel main input Y expected pointer or tensor handle"])
        A_handle: T.handle = T.Select(A_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 0, 15, "handle"), 24), T.tvm_struct_get(args, 0, 15, "handle"))
        X_handle: T.handle = T.Select(X_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 1, 15, "handle"), 24), T.tvm_struct_get(args, 1, 15, "handle"))
        Y_handle: T.handle = T.Select(Y_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 2, 15, "handle"), 24), T.tvm_struct_get(args, 2, 15, "handle"))
        main_A_is_null: T.bool = T.isnullptr(A_handle)
        assert not main_A_is_null, ("RuntimeError", ["main.A is expected to have non-NULL pointer"])
        main_X_is_null: T.bool = T.isnullptr(X_handle)
        assert not main_X_is_null, ("RuntimeError", ["main.X is expected to have non-NULL pointer"])
        main_Y_is_null: T.bool = T.isnullptr(Y_handle)
        assert not main_Y_is_null, ("RuntimeError", ["main.Y is expected to have non-NULL pointer"])
        main_A_shape: T.handle("int64", "global") = T.tvm_struct_get(A_handle, 0, 2, "handle")
        main_A_shape_1 = T.decl_buffer((2,), "int64", data=main_A_shape)
        main_X_shape: T.handle("int64", "global") = T.tvm_struct_get(X_handle, 0, 2, "handle")
        main_X_shape_1 = T.decl_buffer((1,), "int64", data=main_X_shape)
        main_Y_shape: T.handle("int64", "global") = T.tvm_struct_get(Y_handle, 0, 2, "handle")
        main_Y_shape_1 = T.decl_buffer((1,), "int64", data=main_Y_shape)
        assert T.tvm_struct_get(A_handle, 0, 4, "int32") == 2, ("RuntimeError", ["kernel main input A ndim mismatch, expected 2"])
        main_A_strides: T.handle("int64", "global") = T.tvm_struct_get(A_handle, 0, 3, "handle")
        main_A_strides_1 = T.decl_buffer((2,), "int64", data=main_A_strides)
        dev_id: T.int32 = T.tvm_struct_get(A_handle, 0, 9, "int32")
        A = T.tvm_struct_get(A_handle, 0, 1, "handle")
        T.attr(A, "storage_alignment", 64)
        assert T.tvm_struct_get(X_handle, 0, 4, "int32") == 1, ("RuntimeError", ["kernel main input X ndim mismatch, expected 1"])
        main_X_strides: T.handle("int64", "global") = T.tvm_struct_get(X_handle, 0, 3, "handle")
        main_X_strides_1 = T.decl_buffer((1,), "int64", data=main_X_strides)
        X = T.tvm_struct_get(X_handle, 0, 1, "handle")
        T.attr(X, "storage_alignment", 64)
        assert T.tvm_struct_get(Y_handle, 0, 4, "int32") == 1, ("RuntimeError", ["kernel main input Y ndim mismatch, expected 1"])
        main_Y_strides: T.handle("int64", "global") = T.tvm_struct_get(Y_handle, 0, 3, "handle")
        main_Y_strides_1 = T.decl_buffer((1,), "int64", data=main_Y_strides)
        Y: T.handle("float32", "global") = T.tvm_struct_get(Y_handle, 0, 1, "handle")
        T.attr(Y, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        assert T.tvm_struct_get(A_handle, 0, 5, "uint8") == T.uint8(4) and T.tvm_struct_get(A_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(A_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input A dtype mismatch, expected bfloat16"])
        assert T.Cast("int32", main_A_shape_1[0]) == 2037, ("RuntimeError", ["kernel main input A shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_A_shape_1[1]) == 224, ("RuntimeError", ["kernel main input A shape[1] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_A_strides), 1, T.Cast("int32", main_A_strides_1[1])) == 1, ("RuntimeError", ["kernel main input A strides[1] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_A_strides), 1, T.Cast("int32", main_A_strides_1[0])) == 224, ("RuntimeError", ["kernel main input A strides[0] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(A_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input A byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(A_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input A device_type mismatch, expected cuda"])
        assert not T.isnullptr(A), ("RuntimeError", ["kernel main input A data pointer is NULL"])
        assert T.tvm_struct_get(X_handle, 0, 5, "uint8") == T.uint8(4) and T.tvm_struct_get(X_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(X_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input X dtype mismatch, expected bfloat16"])
        assert T.Cast("int32", main_X_shape_1[0]) == 224, ("RuntimeError", ["kernel main input X shape[0] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_X_strides), 1, T.Cast("int32", main_X_strides_1[0])) == 1, ("RuntimeError", ["kernel main input X strides[0] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(X_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input X byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(X_handle, 0, 9, "int32") == T.tvm_struct_get(A_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input X device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(X_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input X device_type mismatch, expected cuda"])
        assert not T.isnullptr(X), ("RuntimeError", ["kernel main input X data pointer is NULL"])
        assert T.tvm_struct_get(Y_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(Y_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(Y_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input Y dtype mismatch, expected float32"])
        assert T.Cast("int32", main_Y_shape_1[0]) == 2037, ("RuntimeError", ["kernel main input Y shape[0] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Y_strides), 1, T.Cast("int32", main_Y_strides_1[0])) == 1, ("RuntimeError", ["kernel main input Y strides[0] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(Y_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input Y byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(Y_handle, 0, 9, "int32") == T.tvm_struct_get(A_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input Y device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(Y_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input Y device_type mismatch, expected cuda"])
        assert not T.isnullptr(Y), ("RuntimeError", ["kernel main input Y data pointer is NULL"])
        A_1 = T.decl_buffer((2037, 224), "bfloat16", data=A, strides=(224, 1))
        X_1 = T.decl_buffer((224,), "bfloat16", data=X, strides=(1,))
        Y_1 = T.decl_buffer((2037,), data=Y, strides=(1,))
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "main_compute_"):
            A_desc = T.tvm_stack_alloca("tvm_ffi_any", 16)
            T.call_packed("__tvm_tensormap_create_tiled", A_desc, 9, 2, A, 224, 2037, 2, 448, 64, 64, 1, 1, 0, 0, 2, 0)
            X_desc = T.tvm_stack_alloca("tvm_ffi_any", 16)
            T.call_packed("__tvm_tensormap_create_tiled", X_desc, 9, 1, X, 224, 2, 64, 1, 0, 0, 2, 0)
            T.call_packed("main_kernel", A_desc, X_desc, Y, 32, 160, 1, 1, 41600)
        return 0