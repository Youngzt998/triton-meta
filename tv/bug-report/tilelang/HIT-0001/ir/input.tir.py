# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(packed_values_handle: T.handle, decoded_values_handle: T.handle):
        packed_values = T.match_buffer(packed_values_handle, (2,), "uint32", strides=(1,))
        decoded_values = T.match_buffer(decoded_values_handle, (7,), strides=(1,))
        # with T.sblock("root"):
        for bx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(1, thread="threadIdx.x"):
                for ty in T.thread_binding(1, thread="threadIdx.y"):
                    for tz in T.thread_binding(1, thread="threadIdx.z"):
                        with T.sblock("tilelang_root"):
                            T.reads()
                            T.writes()
                            for i in range(7):
                                decoded_values[i] = T.Cast("float32", T.shift_right(T.shift_left(T.Cast("int32", T.bitwise_and(T.shift_right(packed_values[i // 4], T.Cast("uint32", i % 4 * 8)), T.uint32(255))), 24), 24))