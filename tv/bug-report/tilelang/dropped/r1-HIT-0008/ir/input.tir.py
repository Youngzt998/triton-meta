# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def cumsum(A_handle: T.handle, B_handle: T.handle):
        A = T.match_buffer(A_handle, (192, 160), strides=(160, 1))
        B = T.match_buffer(B_handle, (192, 160), strides=(160, 1))
        # with T.sblock("root"):
        for bx in T.thread_binding(5, thread="blockIdx.x"):
            for by in T.thread_binding(3, thread="blockIdx.y"):
                for tx in T.thread_binding(256, thread="threadIdx.x"):
                    for ty in T.thread_binding(1, thread="threadIdx.y"):
                        for tz in T.thread_binding(1, thread="threadIdx.z"):
                            with T.sblock("tilelang_root"):
                                T.reads()
                                T.writes()
                                A_shared = T.sblock_alloc_buffer((64, 32), scope="shared.dyn")
                                T.copy(T.region(A[by * 64, bx * 32], 1, 64, 32), T.region(A_shared[0, 0], 2, 64, 32))
                                T.cumsum(T.region(A_shared[0, 0], 1, 64, 32), T.region(A_shared[0, 0], 2, 64, 32), 0, T.bool(True))
                                T.copy(T.region(A_shared[0, 0], 1, 64, 32), T.region(B[by * 64, bx * 32], 2, 64, 32))