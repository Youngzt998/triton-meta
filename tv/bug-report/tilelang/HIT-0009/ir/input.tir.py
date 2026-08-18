# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
        A = T.match_buffer(A_handle, (128, 128), strides=(128, 1))
        B = T.match_buffer(B_handle, (128, 128), strides=(128, 1))
        C = T.match_buffer(C_handle, (128, 64), strides=(64, 1))
        # with T.sblock("root"):
        for bx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                for ty in T.thread_binding(1, thread="threadIdx.y"):
                    for tz in T.thread_binding(1, thread="threadIdx.z"):
                        with T.sblock("tilelang_root"):
                            T.reads()
                            T.writes()
                            row: T.int32 = bx * 128 + tx
                            for col in T.vectorized(128):
                                B[row, col] = A[row, col] * C[row, col // 2]