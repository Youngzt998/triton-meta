# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def mhc_pre_big_fuse_tilelang(gemm_out_mul: T.handle, gemm_out_sqrsum: T.handle, hc_scale: T.handle, hc_base: T.handle, residual: T.handle, post_mix: T.handle, comb_mix: T.handle, layer_input: T.handle):
        num_tokens = T.int32()
        gemm_out_mul_1 = T.match_buffer(gemm_out_mul, (1, num_tokens, 24), strides=(24 * num_tokens, 24, 1))
        gemm_out_sqrsum_1 = T.match_buffer(gemm_out_sqrsum, (1, num_tokens), strides=(num_tokens, 1))
        hc_scale_1 = T.match_buffer(hc_scale, (3,), strides=(1,))
        hc_base_1 = T.match_buffer(hc_base, (24,), strides=(1,))
        residual_1 = T.match_buffer(residual, (num_tokens, 4, 4096), "bfloat16", strides=(16384, 4096, 1))
        post_mix_1 = T.match_buffer(post_mix, (num_tokens, 4), strides=(4, 1))
        comb_mix_1 = T.match_buffer(comb_mix, (num_tokens, 16), strides=(16, 1))
        layer_input_1 = T.match_buffer(layer_input, (num_tokens, 4096), "bfloat16", strides=(4096, 1))
        # with T.sblock("root"):
        for bx in T.thread_binding(num_tokens, thread="blockIdx.x"):
            for tx in T.thread_binding(96, thread="threadIdx.x"):
                for ty in T.thread_binding(1, thread="threadIdx.y"):
                    for tz in T.thread_binding(1, thread="threadIdx.z"):
                        with T.sblock("tilelang_root"):
                            T.reads()
                            T.writes()
                            rms = T.sblock_alloc_buffer((1,), scope="local.fragment")
                            mixes = T.sblock_alloc_buffer((24,), scope="local.fragment")
                            mixes_shared = T.sblock_alloc_buffer((24,), scope="shared.dyn")
                            cm = T.sblock_alloc_buffer((4, 4), scope="local.fragment")
                            row_sum = T.sblock_alloc_buffer((4,), scope="local.fragment")
                            col_sum = T.sblock_alloc_buffer((4,), scope="local.fragment")
                            row_max = T.sblock_alloc_buffer((4,), scope="local.fragment")
                            pre_mix_shared = T.sblock_alloc_buffer((4,), scope="shared.dyn")
                            xs = T.sblock_alloc_buffer((4, 512), scope="shared.dyn")
                            xl = T.sblock_alloc_buffer((4, 512), scope="local.fragment")
                            ol = T.sblock_alloc_buffer((512,), scope="local.fragment")
                            T.fill(T.region(mixes[0], 2, 24), 0)
                            rms[0] = T.float32(0.0)
                            for i_split in range(1):
                                rms[0] = rms[0] + gemm_out_sqrsum_1[i_split, bx]
                            rms[0] = T.rsqrt(rms[0] / T.float32(16384.0) + T.float32(9.9999999999999995e-07))
                            for j in T.parallel(24):
                                mixes[j] = T.float32(0.0)
                                for i_split in range(1):
                                    mixes[j] = mixes[j] + gemm_out_mul_1[i_split, bx, j]
                                mixes[j] = mixes[j] * rms[0]
                            T.copy(T.region(mixes[0], 1, 24), T.region(mixes_shared[0], 2, 24))
                            if tx < 32:
                                for j in T.parallel(4):
                                    post_mix_1[bx, j] = T.sigmoid(mixes_shared[j + 4] * hc_scale_1[1] + hc_base_1[j + 4])
                                for j in T.parallel(4):
                                    for k in T.parallel(4):
                                        cm[j, k] = mixes_shared[j * 4 + k + 8] * hc_scale_1[2] + hc_base_1[j * 4 + k + 8]
                                T.reduce(T.region(cm[0, 0], 1, 4, 4), T.region(row_max[0], 2, 4), "max", 1, T.bool(True))
                                for j in T.parallel(4):
                                    for k in T.parallel(4):
                                        cm[j, k] = T.exp(cm[j, k] - row_max[j])
                                T.reduce(T.region(cm[0, 0], 1, 4, 4), T.region(row_sum[0], 2, 4), "sum", 1, T.bool(True))
                                for j in T.parallel(4):
                                    for k in T.parallel(4):
                                        cm[j, k] = cm[j, k] / row_sum[j] + T.float32(9.9999999999999995e-07)
                                T.reduce(T.region(cm[0, 0], 1, 4, 4), T.region(col_sum[0], 2, 4), "sum", 0, T.bool(True))
                                for j in T.parallel(4):
                                    for k in T.parallel(4):
                                        cm[j, k] = cm[j, k] / (col_sum[k] + T.float32(9.9999999999999995e-07))
                                for _ in range(9):
                                    T.reduce(T.region(cm[0, 0], 1, 4, 4), T.region(row_sum[0], 2, 4), "sum", 1, T.bool(True))
                                    for j in T.parallel(4):
                                        for k in T.parallel(4):
                                            cm[j, k] = cm[j, k] / (row_sum[j] + T.float32(9.9999999999999995e-07))
                                    T.reduce(T.region(cm[0, 0], 1, 4, 4), T.region(col_sum[0], 2, 4), "sum", 0, T.bool(True))
                                    for j in T.parallel(4):
                                        for k in T.parallel(4):
                                            cm[j, k] = cm[j, k] / (col_sum[k] + T.float32(9.9999999999999995e-07))
                                for j in T.parallel(4):
                                    for k in T.parallel(4):
                                        comb_mix_1[bx, j * 4 + k] = cm[j, k]
                            else:
                                for j in T.parallel(4):
                                    pre_mix_shared[j] = T.sigmoid(mixes_shared[j] * hc_scale_1[0] + hc_base_1[j]) + T.float32(9.9999999999999995e-07)
                                for i0_h in T.serial(8, annotations={"num_stages": 2}):
                                    T.copy(T.region(residual_1[bx, 0, i0_h * 512], 1, 1, 4, 512), T.region(xs[0, 0], 2, 4, 512))
                                    T.copy(T.region(xs[0, 0], 1, 4, 512), T.region(xl[0, 0], 2, 4, 512))
                                    T.fill(T.region(ol[0], 2, 512), 0)
                                    for i_hc in range(4):
                                        pre: T.float32 = pre_mix_shared[i_hc]
                                        for i1_h in T.parallel(512):
                                            ol[i1_h] = ol[i1_h] + pre * xl[i_hc, i1_h]
                                    T.copy(T.region(ol[0], 1, 512), T.region(layer_input_1[bx, i0_h * 512], 2, 1, 512))