# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel(a_out: T.handle("float32", "global")):
        T.func_attr({"calling_conv": 2, "target": T.target({"arch": "sm_90a", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1, "threadIdx.x": 128, "threadIdx.y": 1, "threadIdx.z": 1}, "tirx.is_global_func": T.bool(True), "tirx.kernel_launch_params": ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z"], "tirx.noalias": True, "tl.non_restrict_params": []})
        a_out_1 = T.decl_buffer((2048,), data=a_out)
        bx = T.launch_thread("blockIdx.x", 1)
        a_fp32_local = T.alloc_buffer((16,), scope="local")
        by = T.launch_thread("blockIdx.y", 1)
        bz = T.launch_thread("blockIdx.z", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        for i in T.unroll(4):
            a_fp32_local_1 = T.decl_buffer((16,), data=a_fp32_local.data, scope="local")
            a_out_1[i * 512 + tx * 4:i * 512 + tx * 4 + 4] = a_fp32_local_1[i * 4:i * 4 + 4]

    @T.prim_func
    def main(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle("void", "global")) -> T.int32:
        T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_main", "target": T.target({"keys": ["cpu"], "kind": "c", "tag": ""}), "thread_extent": {}, "tirx.is_entry_func": True, "tl.has_tma": T.bool(False), "tl.readonly_param_indices": [0, 1], "tma_descriptor_args": {}})
        assert num_args == 2, ("RuntimeError", ["main: num_args should be 2"])
        assert not T.isnullptr(args), ("RuntimeError", ["main: args pointer is NULL"])
        a_handle_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
        assert a_handle_type_index == 0 or a_handle_type_index == 4 or a_handle_type_index == 7 or 64 <= a_handle_type_index, ("RuntimeError", ["kernel main input a expected pointer or tensor handle"])
        a_out_handle_type_index: T.int32 = T.tvm_struct_get(args, 1, 13, "int32")
        assert a_out_handle_type_index == 0 or a_out_handle_type_index == 4 or a_out_handle_type_index == 7 or 64 <= a_out_handle_type_index, ("RuntimeError", ["kernel main input a_out expected pointer or tensor handle"])
        a_handle: T.handle = T.Select(a_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 0, 15, "handle"), 24), T.tvm_struct_get(args, 0, 15, "handle"))
        a_out_handle: T.handle = T.Select(a_out_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 1, 15, "handle"), 24), T.tvm_struct_get(args, 1, 15, "handle"))
        main_a_is_null: T.bool = T.isnullptr(a_handle)
        main_a_out_is_null: T.bool = T.isnullptr(a_out_handle)
        assert not main_a_out_is_null, ("RuntimeError", ["main.a_out is expected to have non-NULL pointer"])
        main_a_shape: T.handle("int64", "global") = T.if_then_else(not main_a_is_null, T.tvm_struct_get(a_handle, 0, 2, "handle"), T.reinterpret("handle", T.uint64(0)))
        main_a_shape_1 = T.decl_buffer((3,), "int64", data=main_a_shape)
        main_a_out_shape: T.handle("int64", "global") = T.tvm_struct_get(a_out_handle, 0, 2, "handle")
        main_a_out_shape_1 = T.decl_buffer((3,), "int64", data=main_a_out_shape)
        assert main_a_is_null or T.tvm_struct_get(a_handle, 0, 4, "int32") == 3, ("RuntimeError", ["kernel main input a ndim mismatch, expected 3"])
        main_a_strides: T.handle("int64", "global") = T.if_then_else(not main_a_is_null, T.tvm_struct_get(a_handle, 0, 3, "handle"), T.reinterpret("handle", T.uint64(0)))
        main_a_strides_1 = T.decl_buffer((0,), "int64", data=main_a_strides)
        dev_id: T.int32 = T.if_then_else(not main_a_is_null, T.tvm_struct_get(a_handle, 0, 9, "int32"), 0)
        a: T.handle("bfloat16", "global") = T.if_then_else(not main_a_is_null, T.tvm_struct_get(a_handle, 0, 1, "handle"), T.reinterpret("handle", T.uint64(0)))
        T.attr(a, "storage_alignment", 64)
        assert T.tvm_struct_get(a_out_handle, 0, 4, "int32") == 3, ("RuntimeError", ["kernel main input a_out ndim mismatch, expected 3"])
        main_a_out_strides: T.handle("int64", "global") = T.tvm_struct_get(a_out_handle, 0, 3, "handle")
        main_a_out_strides_1 = T.decl_buffer((0,), "int64", data=main_a_out_strides)
        a_out: T.handle("float32", "global") = T.tvm_struct_get(a_out_handle, 0, 1, "handle")
        T.attr(a_out, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        assert main_a_is_null or T.if_then_else(not main_a_is_null, T.tvm_struct_get(a_handle, 0, 5, "uint8"), T.uint8(4)) == T.uint8(4) and T.if_then_else(not main_a_is_null, T.tvm_struct_get(a_handle, 0, 6, "uint8"), T.uint8(16)) == T.uint8(16) and T.if_then_else(not main_a_is_null, T.tvm_struct_get(a_handle, 0, 7, "uint16"), T.uint16(1)) == T.uint16(1), ("RuntimeError", ["kernel main input a dtype mismatch, expected bfloat16"])
        assert main_a_is_null or T.if_then_else(not main_a_is_null, T.Cast("int32", main_a_shape_1[0]), 0) == 1, ("RuntimeError", ["kernel main input a shape[0] violates packed ABI constraint"])
        assert main_a_is_null or T.if_then_else(not main_a_is_null, T.Cast("int32", main_a_shape_1[1]), 0) == 32, ("RuntimeError", ["kernel main input a shape[1] violates packed ABI constraint"])
        assert main_a_is_null or T.if_then_else(not main_a_is_null, T.Cast("int32", main_a_shape_1[2]), 0) == 64, ("RuntimeError", ["kernel main input a shape[2] violates packed ABI constraint"])
        assert T.isnullptr(main_a_strides) or T.Cast("int32", main_a_strides_1[2]) == 1 and T.Cast("int32", main_a_strides_1[1]) == 64, ("RuntimeError", ["main.a.strides: expected to be compact array, but got non-compact strides"])
        assert main_a_is_null or T.uint64(0) == T.if_then_else(not main_a_is_null, T.tvm_struct_get(a_handle, 0, 8, "uint64"), T.uint64(0)), ("RuntimeError", ["kernel main input a byte_offset violates packed ABI constraint"])
        assert main_a_is_null or T.if_then_else(not main_a_is_null, T.tvm_struct_get(a_handle, 0, 10, "int32"), 0) == 2, ("RuntimeError", ["kernel main input a device_type mismatch, expected cuda"])
        assert main_a_is_null or not T.isnullptr(a), ("RuntimeError", ["kernel main input a data pointer is NULL"])
        assert T.tvm_struct_get(a_out_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(a_out_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(a_out_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input a_out dtype mismatch, expected float32"])
        assert T.Cast("int32", main_a_out_shape_1[0]) == 1, ("RuntimeError", ["kernel main input a_out shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_a_out_shape_1[1]) == 32, ("RuntimeError", ["kernel main input a_out shape[1] violates packed ABI constraint"])
        assert T.Cast("int32", main_a_out_shape_1[2]) == 64, ("RuntimeError", ["kernel main input a_out shape[2] violates packed ABI constraint"])
        assert T.isnullptr(main_a_out_strides) or T.Cast("int32", main_a_out_strides_1[2]) == 1 and T.Cast("int32", main_a_out_strides_1[1]) == 64, ("RuntimeError", ["main.a_out.strides: expected to be compact array, but got non-compact strides"])
        assert T.uint64(0) == T.tvm_struct_get(a_out_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input a_out byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(a_out_handle, 0, 9, "int32") == T.if_then_else(not main_a_is_null, T.tvm_struct_get(a_handle, 0, 9, "int32"), 0), ("RuntimeError", ["kernel main input a_out device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(a_out_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input a_out device_type mismatch, expected cuda"])
        assert not T.isnullptr(a_out), ("RuntimeError", ["kernel main input a_out data pointer is NULL"])
        a_1 = T.decl_buffer((1, 32, 64), "bfloat16", data=a)
        a_out_1 = T.decl_buffer((1, 32, 64), data=a_out)
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "main_compute_"):
            T.call_packed("main_kernel", a_out, 1, 1, 1, 128, 1, 1)
        return 0