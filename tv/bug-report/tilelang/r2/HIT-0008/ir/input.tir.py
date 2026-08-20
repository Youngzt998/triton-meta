# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A_handle: T.handle, X_handle: T.handle, Y_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90a", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        A = T.match_buffer(A_handle, (511, 1025), strides=(1025, 1))
        X = T.match_buffer(X_handle, (1025,), strides=(1,))
        Y = T.match_buffer(Y_handle, (511,), strides=(1,))
        # with T.sblock("root"):
        bx = T.launch_thread("blockIdx.x", 8)
        tx = T.launch_thread("threadIdx.x", 32)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            As = T.sblock_alloc_buffer((64, 256), scope="shared.dyn")
            Xs = T.sblock_alloc_buffer((256,), scope="shared.dyn")
            p = T.sblock_alloc_buffer((64, 256), scope="local.fragment")
            acc = T.sblock_alloc_buffer((64,), scope="local.fragment")
            T.fill(T.region(acc[0], 2, 64), 0)
            for ko in T.serial(5, annotations={"num_stages": 3}):
                T.copy(T.region(A[bx * 64, ko * 256], 1, 64, 256), T.region(As[0, 0], 2, 64, 256))
                T.copy(T.region(X[ko * 256], 1, 256), T.region(Xs[0], 2, 256))
                for i in T.parallel(64):
                    for j in T.parallel(256):
                        p[i, j] = As[i, j] * Xs[j]
                T.reduce(T.region(p[0, 0], 1, 64, 256), T.region(acc[0], 2, 64), "sum", 1, T.bool(False))
            T.copy(T.region(acc[0], 1, 64), T.region(Y[bx * 64], 2, 64))