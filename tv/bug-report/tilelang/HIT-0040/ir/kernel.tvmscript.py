# from tvm.script import tirx as T

@T.prim_func
def main(A_handle: T.handle, Out_handle: T.handle):
    A = T.match_buffer(A_handle, (256, 4096), strides=(4096, 1))
    Out = T.match_buffer(Out_handle, (256,), strides=(1,))
    # with T.sblock("root"):
    for bx in T.thread_binding(8, thread="blockIdx.x"):
        for tx in T.thread_binding(128, thread="threadIdx.x"):
            for ty in T.thread_binding(1, thread="threadIdx.y"):
                for tz in T.thread_binding(1, thread="threadIdx.z"):
                    with T.sblock("tilelang_root"):
                        T.reads()
                        T.writes()
                        As = T.sblock_alloc_buffer((32, 256), scope="shared.dyn")
                        acc = T.sblock_alloc_buffer((32,), scope="local.fragment")
                        tmp = T.sblock_alloc_buffer((32,), scope="local.fragment")
                        As_frag = T.sblock_alloc_buffer((32, 256), scope="local.fragment")
                        T.fill(T.region(acc[0], 2, 32), 0)
                        for ko in T.serial(16, annotations={"num_stages": 2}):
                            T.copy(T.region(A[bx * 32, ko * 256], 1, 32, 256), T.region(As[0, 0], 2, 32, 256))
                            T.copy(T.region(As[0, 0], 1, 32, 256), T.region(As_frag[0, 0], 2, 32, 256))
                            T.reduce(T.region(As_frag[0, 0], 1, 32, 256), T.region(tmp[0], 2, 32), "sum", 1, T.bool(True))
                            for i in T.parallel(32):
                                acc[i] = acc[i] + tmp[i]
                        T.copy(T.region(acc[0], 1, 32), T.region(Out[bx * 32], 2, 32))