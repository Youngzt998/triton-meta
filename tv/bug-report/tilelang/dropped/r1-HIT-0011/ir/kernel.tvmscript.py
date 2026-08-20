# from tvm.script import tirx as T

@T.prim_func
def main(a: T.Buffer((1, 32, 64), "bfloat16"), a_out: T.Buffer((1, 32, 64), "float32")):
    # with T.sblock("root"):
    for bx in T.thread_binding(1, thread="blockIdx.x"):
        for by in T.thread_binding(1, thread="blockIdx.y"):
            for bz in T.thread_binding(1, thread="blockIdx.z"):
                for tx in T.thread_binding(128, thread="threadIdx.x"):
                    for ty in T.thread_binding(1, thread="threadIdx.y"):
                        for tz in T.thread_binding(1, thread="threadIdx.z"):
                            with T.sblock("tilelang_root"):
                                T.reads()
                                T.writes()
                                a_fp32_local = T.sblock_alloc_buffer((64, 32), scope="local.fragment")
                                offs_m: T.int32 = bx * 32
                                offs_n: T.int32 = by * 64
                                for i in T.parallel(32):
                                    for j in T.parallel(64):
                                        idx: T.int32 = i * 64 + j
                                        a_out[bz, offs_m + i, offs_n + j] = a_fp32_local[idx // 32, idx % 32]