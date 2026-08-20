# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel(A: T.handle("float32", "global"), X: T.handle("float32", "global"), Y: T.handle("float32", "global")):
        T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 33280, "target": T.target({"arch": "sm_90a", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 1, "threadIdx.x": 32, "threadIdx.y": 1, "threadIdx.z": 1}, "tirx.is_global_func": T.bool(True), "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tirx.use_dyn_shared_memory"], "tirx.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1]})
        Y_1 = T.decl_buffer((31,), data=Y)
        bx = T.launch_thread("blockIdx.x", 1)
        buf_dyn_shmem = T.alloc_buffer((33280,), "uint8", scope="shared.dyn")
        As: T.handle("float32", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        Xs: T.handle("float32", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 32768)
        acc = T.alloc_buffer((2,), scope="local")
        p = T.alloc_buffer((256,), scope="local")
        acc_clear = T.alloc_buffer((64,), scope="local")
        acc_clear_1 = T.alloc_buffer((64,), scope="local")
        tx = T.launch_thread("threadIdx.x", 32)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        As_1 = T.decl_buffer((8192,), data=As, scope="shared.dyn")
        Xs_1 = T.decl_buffer((128,), data=Xs, scope="shared.dyn")
        p_1 = T.decl_buffer((256,), data=p.data, scope="local")
        acc_1 = T.decl_buffer((2,), data=acc.data, scope="local")
        acc_2 = T.decl_buffer((2,), data=acc.data, scope="local")
        p_2 = T.decl_buffer((256,), data=p.data, scope="local")
        p_3 = T.decl_buffer((256,), data=p.data, scope="local")
        acc_3 = T.decl_buffer((2,), data=acc.data, scope="local")
        p_4 = T.decl_buffer((256,), data=p.data, scope="local")
        acc_2[0:2] = T.Broadcast(T.float32(0.0), 2)
        for i in T.unroll(64):
            for vec in range(4):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float32"), As, i * 128 + tx * 4 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float32"), A, i * 1027 + tx * 4 + vec, 1, 1), 1, i < 31 and i < 31)
        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float32"), Xs, tx * 4, 1, 2), T.tvm_access_ptr(T.type_annotation("float32"), X, tx * 4, 1, 1), 4)
        T.ptx_commit_group()
        for ko in T.serial(8, annotations={"tl_pipelined_num_stages": 1}):
            T.ptx_wait_group(0)
            T.tvm_storage_sync("shared")
            for i in T.unroll(64):
                p_2[i * 4:i * 4 + 4] = As_1[i * 128 + tx * 4:i * 128 + tx * 4 + 4] * Xs_1[tx * 4:tx * 4 + 4]
            T.tvm_storage_sync("shared.dyn")
            for i in T.unroll(256):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float32"), As, i * 32 + tx, 1, 2), T.tvm_access_ptr(T.type_annotation("float32"), A, i // 4 * 1027 + ko * 128 + i % 4 * 32 + tx + 128, 1, 1), 1, i < 124 and ko * 128 + i % 4 * 32 + tx < 899 and i < 124 and ko * 128 + i % 4 * 32 + tx < 899)
            for i in T.unroll(4):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float32"), Xs, i * 32 + tx, 1, 2), T.tvm_access_ptr(T.type_annotation("float32"), X, ko * 128 + i * 32 + tx + 128, 1, 1), 1, ko * 128 + i * 32 + tx < 899 and ko * 128 + i * 32 + tx < 899)
            T.ptx_commit_group()
            for i in T.unroll(64):
                acc_clear[i] = T.float32(0.0)
                for rv in T.unroll(4):
                    acc_clear[i] = acc_clear[i] + p_3[i * 4 + rv]
                acc_clear[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<32>>::run", acc_clear[i])
                if i // 32 * 32 + tx == i:
                    acc_3[i // 32] = acc_3[i // 32] + acc_clear[i]
        T.ptx_wait_group(0)
        T.tvm_storage_sync("shared")
        for i in T.unroll(64):
            p_4[i * 4:i * 4 + 4] = As_1[i * 128 + tx * 4:i * 128 + tx * 4 + 4] * Xs_1[tx * 4:tx * 4 + 4]
        for i in T.unroll(64):
            acc_clear_1[i] = T.float32(0.0)
            for rv in T.unroll(4):
                acc_clear_1[i] = acc_clear_1[i] + p_1[i * 4 + rv]
            acc_clear_1[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<32>>::run", acc_clear_1[i])
            if i // 32 * 32 + tx == i:
                acc_1[i // 32] = acc_1[i // 32] + acc_clear_1[i]
        for i in T.unroll(2):
            if i * 32 + tx < 31:
                Y_1[i * 32 + tx] = acc_1[i]

    @T.prim_func
    def main(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle("void", "global")) -> T.int32:
        T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_main", "target": T.target({"keys": ["cpu"], "kind": "c", "tag": ""}), "thread_extent": {}, "tirx.is_entry_func": True, "tl.has_tma": T.bool(False), "tl.readonly_param_indices": [0, 1, 2], "tma_descriptor_args": {}})
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
        A: T.handle("float32", "global") = T.tvm_struct_get(A_handle, 0, 1, "handle")
        T.attr(A, "storage_alignment", 64)
        assert T.tvm_struct_get(X_handle, 0, 4, "int32") == 1, ("RuntimeError", ["kernel main input X ndim mismatch, expected 1"])
        main_X_strides: T.handle("int64", "global") = T.tvm_struct_get(X_handle, 0, 3, "handle")
        main_X_strides_1 = T.decl_buffer((1,), "int64", data=main_X_strides)
        X: T.handle("float32", "global") = T.tvm_struct_get(X_handle, 0, 1, "handle")
        T.attr(X, "storage_alignment", 64)
        assert T.tvm_struct_get(Y_handle, 0, 4, "int32") == 1, ("RuntimeError", ["kernel main input Y ndim mismatch, expected 1"])
        main_Y_strides: T.handle("int64", "global") = T.tvm_struct_get(Y_handle, 0, 3, "handle")
        main_Y_strides_1 = T.decl_buffer((1,), "int64", data=main_Y_strides)
        Y: T.handle("float32", "global") = T.tvm_struct_get(Y_handle, 0, 1, "handle")
        T.attr(Y, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        assert T.tvm_struct_get(A_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(A_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(A_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input A dtype mismatch, expected float32"])
        assert T.Cast("int32", main_A_shape_1[0]) == 31, ("RuntimeError", ["kernel main input A shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_A_shape_1[1]) == 1027, ("RuntimeError", ["kernel main input A shape[1] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_A_strides), 1, T.Cast("int32", main_A_strides_1[1])) == 1, ("RuntimeError", ["kernel main input A strides[1] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_A_strides), 1, T.Cast("int32", main_A_strides_1[0])) == 1027, ("RuntimeError", ["kernel main input A strides[0] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(A_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input A byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(A_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input A device_type mismatch, expected cuda"])
        assert not T.isnullptr(A), ("RuntimeError", ["kernel main input A data pointer is NULL"])
        assert T.tvm_struct_get(X_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(X_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(X_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input X dtype mismatch, expected float32"])
        assert T.Cast("int32", main_X_shape_1[0]) == 1027, ("RuntimeError", ["kernel main input X shape[0] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_X_strides), 1, T.Cast("int32", main_X_strides_1[0])) == 1, ("RuntimeError", ["kernel main input X strides[0] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(X_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input X byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(X_handle, 0, 9, "int32") == T.tvm_struct_get(A_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input X device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(X_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input X device_type mismatch, expected cuda"])
        assert not T.isnullptr(X), ("RuntimeError", ["kernel main input X data pointer is NULL"])
        assert T.tvm_struct_get(Y_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(Y_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(Y_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input Y dtype mismatch, expected float32"])
        assert T.Cast("int32", main_Y_shape_1[0]) == 31, ("RuntimeError", ["kernel main input Y shape[0] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Y_strides), 1, T.Cast("int32", main_Y_strides_1[0])) == 1, ("RuntimeError", ["kernel main input Y strides[0] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(Y_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input Y byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(Y_handle, 0, 9, "int32") == T.tvm_struct_get(A_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input Y device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(Y_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input Y device_type mismatch, expected cuda"])
        assert not T.isnullptr(Y), ("RuntimeError", ["kernel main input Y data pointer is NULL"])
        A_1 = T.decl_buffer((31, 1027), data=A, strides=(1027, 1))
        X_1 = T.decl_buffer((1027,), data=X, strides=(1,))
        Y_1 = T.decl_buffer((31,), data=Y, strides=(1,))
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "main_compute_"):
            T.call_packed("main_kernel", A, X, Y, 1, 32, 1, 1, 33280)
        return 0