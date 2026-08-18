# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel(A_desc: T.handle("uint8x128", "grid_constant"), Out: T.handle("float32", "global")):
        T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 66048, "target": T.target({"arch": "sm_90a", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 8, "threadIdx.x": 256, "threadIdx.y": 1, "threadIdx.z": 1}, "tirx.is_global_func": T.bool(True), "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tirx.use_dyn_shared_memory"], "tirx.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0], "tl.smem_alignment_map": {"As": 128}})
        Out_1 = T.decl_buffer((256,), data=Out)
        bx = T.launch_thread("blockIdx.x", 8)
        buf_dyn_shmem = T.alloc_buffer((66048,), "uint8", scope="shared.dyn")
        As: T.handle("float32", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        workspace: T.handle("float32", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 65536)
        mbarrier = T.alloc_buffer((4,), "uint64", scope="shared.barrier")
        acc = T.alloc_buffer((16,), scope="local")
        As_frag = T.alloc_buffer((64,), scope="local")
        tmp = T.alloc_buffer((16,), scope="local")
        tx = T.launch_thread("threadIdx.x", 256)
        if T.tl_shuffle_elect(0):
            T.prefetch_tma_descriptor(A_desc)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        mbarrier_1 = T.decl_buffer((4,), "uint64", data=mbarrier.data, scope="shared.barrier")
        As_1 = T.decl_buffer((16384,), data=As, scope="shared.dyn")
        acc_1 = T.decl_buffer((16,), data=acc.data, scope="local")
        if T.tl_shuffle_elect(0):
            T.ptx_init_barrier_thread_count(mbarrier_1[0], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[1], 1)
            T.ptx_init_barrier_thread_count(mbarrier_1[2], 128)
            T.ptx_init_barrier_thread_count(mbarrier_1[3], 128)
        T.ptx_fence_barrier_init()
        T.tvm_storage_sync("shared")
        T.attr([128, 128], "kWarpSpecializationScope", 0)
        if tx < 128:
            T.set_max_nreg(24, 0)
            for ko in range(16):
                T.mbarrier_wait_parity(mbarrier_1[ko % 2 + 2], T.bitwise_xor(ko % 4 // 2, 1))
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(mbarrier_1[ko % 2], 32768)
                    T.tma_load(A_desc, mbarrier_1[ko % 2], T.tvm_access_ptr(T.type_annotation("float32"), As, ko % 2 * 8192, 8192, 2), ko * 256, bx * 32, 0)
        else:
            T.set_max_nreg(240, 1)
            for i in T.unroll(4):
                acc_1[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
            for ko in range(16):
                tmp_1 = T.decl_buffer((16,), data=tmp.data, scope="local")
                As_frag_1 = T.decl_buffer((64,), data=As_frag.data, scope="local")
                T.mbarrier_wait_parity(mbarrier_1[ko % 2], ko % 4 // 2)
                T.tvm_storage_sync("shared.dyn", 3, 128)
                for i in T.unroll(16):
                    As_frag_1[i * 4:i * 4 + 4] = As_1[ko % 2 * 8192 + i * 512 + tx * 4 - 512:ko % 2 * 8192 + i * 512 + tx * 4 - 512 + 4]
                T.ptx_arrive_barrier(mbarrier_1[ko % 2 + 2])
                T.tvm_storage_sync("shared.dyn", 3, 128)
                for i in T.unroll(16):
                    tmp_1[i] = T.float32(0.0)
                    for rv in T.unroll(4):
                        tmp_1[i] = tmp_1[i] + As_frag_1[i * 4 + rv]
                    tmp_1[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 64, 1, 128, tl::NamedBarrier<128>>::run", tmp_1[i], T.tvm_access_ptr(T.type_annotation("float32"), workspace, 0, 128, 2))
                for i in T.unroll(16):
                    acc_1[i] = acc_1[i] + tmp_1[i]
            if tx % 64 == 0:
                for i in T.unroll(16):
                    Out_1[bx * 32 + i * 2 + tx // 64 - 2] = acc_1[i]

    @T.prim_func
    def main(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle("void", "global")) -> T.int32:
        A_desc = T.handle("uint8x128", "grid_constant")
        A = T.handle("float32", "global")
        T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_main", "target": T.target({"keys": ["cpu"], "kind": "c", "tag": ""}), "thread_extent": {}, "tirx.is_entry_func": True, "tl.has_tma": T.bool(True), "tl.readonly_param_indices": [0, 1], "tl_tiled_ws_applied": 1, "tma_descriptor_args": {A_desc: ["__tvm_tensormap_create_tiled", A_desc, 7, 2, A, 4096, 256, 4, 16384, 256, 32, 1, 1, 0, 0, 2, 0]}})
        assert num_args == 2, ("RuntimeError", ["main: num_args should be 2"])
        assert not T.isnullptr(args), ("RuntimeError", ["main: args pointer is NULL"])
        A_handle_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
        assert A_handle_type_index == 0 or A_handle_type_index == 4 or A_handle_type_index == 7 or 64 <= A_handle_type_index, ("RuntimeError", ["kernel main input A expected pointer or tensor handle"])
        Out_handle_type_index: T.int32 = T.tvm_struct_get(args, 1, 13, "int32")
        assert Out_handle_type_index == 0 or Out_handle_type_index == 4 or Out_handle_type_index == 7 or 64 <= Out_handle_type_index, ("RuntimeError", ["kernel main input Out expected pointer or tensor handle"])
        A_handle: T.handle = T.Select(A_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 0, 15, "handle"), 24), T.tvm_struct_get(args, 0, 15, "handle"))
        Out_handle: T.handle = T.Select(Out_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 1, 15, "handle"), 24), T.tvm_struct_get(args, 1, 15, "handle"))
        main_A_is_null: T.bool = T.isnullptr(A_handle)
        assert not main_A_is_null, ("RuntimeError", ["main.A is expected to have non-NULL pointer"])
        main_Out_is_null: T.bool = T.isnullptr(Out_handle)
        assert not main_Out_is_null, ("RuntimeError", ["main.Out is expected to have non-NULL pointer"])
        main_A_shape: T.handle("int64", "global") = T.tvm_struct_get(A_handle, 0, 2, "handle")
        main_A_shape_1 = T.decl_buffer((2,), "int64", data=main_A_shape)
        main_Out_shape: T.handle("int64", "global") = T.tvm_struct_get(Out_handle, 0, 2, "handle")
        main_Out_shape_1 = T.decl_buffer((1,), "int64", data=main_Out_shape)
        assert T.tvm_struct_get(A_handle, 0, 4, "int32") == 2, ("RuntimeError", ["kernel main input A ndim mismatch, expected 2"])
        main_A_strides: T.handle("int64", "global") = T.tvm_struct_get(A_handle, 0, 3, "handle")
        main_A_strides_1 = T.decl_buffer((2,), "int64", data=main_A_strides)
        dev_id: T.int32 = T.tvm_struct_get(A_handle, 0, 9, "int32")
        A = T.tvm_struct_get(A_handle, 0, 1, "handle")
        T.attr(A, "storage_alignment", 64)
        assert T.tvm_struct_get(Out_handle, 0, 4, "int32") == 1, ("RuntimeError", ["kernel main input Out ndim mismatch, expected 1"])
        main_Out_strides: T.handle("int64", "global") = T.tvm_struct_get(Out_handle, 0, 3, "handle")
        main_Out_strides_1 = T.decl_buffer((1,), "int64", data=main_Out_strides)
        Out: T.handle("float32", "global") = T.tvm_struct_get(Out_handle, 0, 1, "handle")
        T.attr(Out, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        assert T.tvm_struct_get(A_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(A_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(A_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input A dtype mismatch, expected float32"])
        assert T.Cast("int32", main_A_shape_1[0]) == 256, ("RuntimeError", ["kernel main input A shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_A_shape_1[1]) == 4096, ("RuntimeError", ["kernel main input A shape[1] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_A_strides), 1, T.Cast("int32", main_A_strides_1[1])) == 1, ("RuntimeError", ["kernel main input A strides[1] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_A_strides), 1, T.Cast("int32", main_A_strides_1[0])) == 4096, ("RuntimeError", ["kernel main input A strides[0] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(A_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input A byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(A_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input A device_type mismatch, expected cuda"])
        assert not T.isnullptr(A), ("RuntimeError", ["kernel main input A data pointer is NULL"])
        assert T.tvm_struct_get(Out_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(Out_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(Out_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input Out dtype mismatch, expected float32"])
        assert T.Cast("int32", main_Out_shape_1[0]) == 256, ("RuntimeError", ["kernel main input Out shape[0] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Out_strides), 1, T.Cast("int32", main_Out_strides_1[0])) == 1, ("RuntimeError", ["kernel main input Out strides[0] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(Out_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input Out byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(Out_handle, 0, 9, "int32") == T.tvm_struct_get(A_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input Out device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(Out_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input Out device_type mismatch, expected cuda"])
        assert not T.isnullptr(Out), ("RuntimeError", ["kernel main input Out data pointer is NULL"])
        A_1 = T.decl_buffer((256, 4096), data=A, strides=(4096, 1))
        Out_1 = T.decl_buffer((256,), data=Out, strides=(1,))
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "main_compute_"):
            A_desc = T.tvm_stack_alloca("tvm_ffi_any", 16)
            T.call_packed("__tvm_tensormap_create_tiled", A_desc, 7, 2, A, 4096, 256, 4, 16384, 256, 32, 1, 1, 0, 0, 2, 0)
            T.call_packed("main_kernel", A_desc, Out, 8, 256, 1, 1, 66048)
        return 0