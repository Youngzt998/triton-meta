# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel(decoded_values: T.handle("float32", "global"), packed_values: T.handle("uint32", "global")):
        T.func_attr({"calling_conv": 2, "target": T.target({"arch": "sm_90a", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 1, "threadIdx.x": 1, "threadIdx.y": 1, "threadIdx.z": 1}, "tirx.is_global_func": T.bool(True), "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "threadIdx.y", "threadIdx.z"], "tirx.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [1]})
        packed_values_1 = T.decl_buffer((2,), "uint32", data=packed_values)
        decoded_values_1 = T.decl_buffer((7,), data=decoded_values)
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        for i in range(7):
            decoded_values_1[i] = T.Cast("float32", T.shift_right(T.shift_left(T.Cast("int32", T.bitwise_and(T.shift_right(packed_values_1[i // 4], T.Cast("uint32", i % 4 * 8)), T.uint32(255))), 24), 24))

    @T.prim_func
    def main(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle("void", "global")) -> T.int32:
        T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_main", "target": T.target({"keys": ["cpu"], "kind": "c", "tag": ""}), "thread_extent": {}, "tirx.is_entry_func": True, "tl.has_tma": T.bool(False), "tl.readonly_param_indices": [0, 1], "tma_descriptor_args": {}})
        assert num_args == 2, ("RuntimeError", ["main: num_args should be 2"])
        assert not T.isnullptr(args), ("RuntimeError", ["main: args pointer is NULL"])
        packed_values_handle_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
        assert packed_values_handle_type_index == 0 or packed_values_handle_type_index == 4 or packed_values_handle_type_index == 7 or 64 <= packed_values_handle_type_index, ("RuntimeError", ["kernel main input packed_values expected pointer or tensor handle"])
        decoded_values_handle_type_index: T.int32 = T.tvm_struct_get(args, 1, 13, "int32")
        assert decoded_values_handle_type_index == 0 or decoded_values_handle_type_index == 4 or decoded_values_handle_type_index == 7 or 64 <= decoded_values_handle_type_index, ("RuntimeError", ["kernel main input decoded_values expected pointer or tensor handle"])
        packed_values_handle: T.handle = T.Select(packed_values_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 0, 15, "handle"), 24), T.tvm_struct_get(args, 0, 15, "handle"))
        decoded_values_handle: T.handle = T.Select(decoded_values_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 1, 15, "handle"), 24), T.tvm_struct_get(args, 1, 15, "handle"))
        main_packed_values_is_null: T.bool = T.isnullptr(packed_values_handle)
        assert not main_packed_values_is_null, ("RuntimeError", ["main.packed_values is expected to have non-NULL pointer"])
        main_decoded_values_is_null: T.bool = T.isnullptr(decoded_values_handle)
        assert not main_decoded_values_is_null, ("RuntimeError", ["main.decoded_values is expected to have non-NULL pointer"])
        main_packed_values_shape: T.handle("int64", "global") = T.tvm_struct_get(packed_values_handle, 0, 2, "handle")
        main_packed_values_shape_1 = T.decl_buffer((1,), "int64", data=main_packed_values_shape)
        main_decoded_values_shape: T.handle("int64", "global") = T.tvm_struct_get(decoded_values_handle, 0, 2, "handle")
        main_decoded_values_shape_1 = T.decl_buffer((1,), "int64", data=main_decoded_values_shape)
        assert T.tvm_struct_get(packed_values_handle, 0, 4, "int32") == 1, ("RuntimeError", ["kernel main input packed_values ndim mismatch, expected 1"])
        main_packed_values_strides: T.handle("int64", "global") = T.tvm_struct_get(packed_values_handle, 0, 3, "handle")
        main_packed_values_strides_1 = T.decl_buffer((1,), "int64", data=main_packed_values_strides)
        dev_id: T.int32 = T.tvm_struct_get(packed_values_handle, 0, 9, "int32")
        packed_values: T.handle("uint32", "global") = T.tvm_struct_get(packed_values_handle, 0, 1, "handle")
        T.attr(packed_values, "storage_alignment", 64)
        assert T.tvm_struct_get(decoded_values_handle, 0, 4, "int32") == 1, ("RuntimeError", ["kernel main input decoded_values ndim mismatch, expected 1"])
        main_decoded_values_strides: T.handle("int64", "global") = T.tvm_struct_get(decoded_values_handle, 0, 3, "handle")
        main_decoded_values_strides_1 = T.decl_buffer((1,), "int64", data=main_decoded_values_strides)
        decoded_values: T.handle("float32", "global") = T.tvm_struct_get(decoded_values_handle, 0, 1, "handle")
        T.attr(decoded_values, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        assert T.tvm_struct_get(packed_values_handle, 0, 5, "uint8") == T.uint8(1) and T.tvm_struct_get(packed_values_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(packed_values_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input packed_values dtype mismatch, expected uint32"])
        assert T.Cast("int32", main_packed_values_shape_1[0]) == 2, ("RuntimeError", ["kernel main input packed_values shape[0] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_packed_values_strides), 1, T.Cast("int32", main_packed_values_strides_1[0])) == 1, ("RuntimeError", ["kernel main input packed_values strides[0] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(packed_values_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input packed_values byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(packed_values_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input packed_values device_type mismatch, expected cuda"])
        assert not T.isnullptr(packed_values), ("RuntimeError", ["kernel main input packed_values data pointer is NULL"])
        assert T.tvm_struct_get(decoded_values_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(decoded_values_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(decoded_values_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input decoded_values dtype mismatch, expected float32"])
        assert T.Cast("int32", main_decoded_values_shape_1[0]) == 7, ("RuntimeError", ["kernel main input decoded_values shape[0] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_decoded_values_strides), 1, T.Cast("int32", main_decoded_values_strides_1[0])) == 1, ("RuntimeError", ["kernel main input decoded_values strides[0] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(decoded_values_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input decoded_values byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(decoded_values_handle, 0, 9, "int32") == T.tvm_struct_get(packed_values_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input decoded_values device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(decoded_values_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input decoded_values device_type mismatch, expected cuda"])
        assert not T.isnullptr(decoded_values), ("RuntimeError", ["kernel main input decoded_values data pointer is NULL"])
        packed_values_1 = T.decl_buffer((2,), "uint32", data=packed_values, strides=(1,))
        decoded_values_1 = T.decl_buffer((7,), data=decoded_values, strides=(1,))
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "main_compute_"):
            T.call_packed("main_kernel", decoded_values, packed_values, 1, 1, 1, 1)
        return 0