# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, O_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90a", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tl.has_tma": T.bool(True), "tl.smem_alignment_map": {"Ks": 512, "Os": 512, "Qs": 512, "Vs": 512}, "tl_tiled_ws_applied": 1})
        Q = T.match_buffer(Q_handle, (2, 4, 385, 32), "bfloat16", strides=(49280, 12320, 32, 1))
        K = T.match_buffer(K_handle, (2, 4, 127, 32), "bfloat16", strides=(16256, 4064, 32, 1))
        V = T.match_buffer(V_handle, (2, 4, 127, 32), "bfloat16", strides=(16256, 4064, 32, 1))
        O = T.match_buffer(O_handle, (2, 4, 385, 32), "bfloat16", strides=(49280, 12320, 32, 1))
        with T.sblock("root"):
            T.reads()
            T.writes()
            Qs = T.Buffer((1, 8, 256), "bfloat16", scope="shared.dyn")
            Ks = T.Buffer((1, 8, 256), "bfloat16", scope="shared.dyn")
            mp = T.Buffer((2,), scope="local")
            mx = T.Buffer((2,), scope="local")
            s = T.Buffer((32,), scope="local")
            sc = T.Buffer((32,), "bfloat16", scope="local")
            Vs = T.Buffer((1, 8, 256), "bfloat16", scope="shared.dyn")
            ss = T.Buffer((2,), scope="local")
            sm = T.Buffer((2,), scope="local")
            ls = T.Buffer((2,), scope="local")
            o = T.Buffer((16,), scope="local")
            Os = T.Buffer((1, 8, 256), "bfloat16", scope="shared.dyn")
            T.sblock_attr({"layout_map": {Qs: metadata["tl.Layout"][0], Ks: metadata["tl.Layout"][1], mp: metadata["tl.Fragment"][0], mx: metadata["tl.Fragment"][1], s: metadata["tl.Fragment"][2], sc: metadata["tl.Fragment"][3], Vs: metadata["tl.Layout"][2], ss: metadata["tl.Fragment"][4], sm: metadata["tl.Fragment"][5], ls: metadata["tl.Fragment"][6], o: metadata["tl.Fragment"][7], Os: metadata["tl.Layout"][3]}})
            bx = T.launch_thread("blockIdx.x", 7)
            by = T.launch_thread("blockIdx.y", 4)
            bz = T.launch_thread("blockIdx.z", 2)
            tx = T.launch_thread("threadIdx.x", 256)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.sblock("tilelang_root"):
                T.reads()
                T.writes()
                mbarrier = T.handle("uint64", "shared.barrier")
                T.sblock_attr({"barrier_init": {mbarrier: [1, 1, 128, 128, 1]}, "layout_map": {Qs: metadata["tl.Layout"][0], Ks: metadata["tl.Layout"][1], mp: metadata["tl.Fragment"][0], mx: metadata["tl.Fragment"][1], s: metadata["tl.Fragment"][2], sc: metadata["tl.Fragment"][3], Vs: metadata["tl.Layout"][2], ss: metadata["tl.Fragment"][4], sm: metadata["tl.Fragment"][5], ls: metadata["tl.Fragment"][6], o: metadata["tl.Fragment"][7], Os: metadata["tl.Layout"][3]}})
                mbarrier_1 = T.sblock_alloc_buffer((5,), "uint64", data=mbarrier, scope="shared.barrier")
                Qs = T.sblock_alloc_buffer((1, 8, 256), "bfloat16", data=Qs.data, scope="shared.dyn")
                Ks = T.sblock_alloc_buffer((1, 8, 256), "bfloat16", data=Ks.data, scope="shared.dyn")
                Vs = T.sblock_alloc_buffer((1, 8, 256), "bfloat16", data=Vs.data, scope="shared.dyn")
                Os = T.sblock_alloc_buffer((1, 8, 256), "bfloat16", data=Os.data, scope="shared.dyn")
                o = T.sblock_alloc_buffer((16,), data=o.data, scope="local")
                mx = T.sblock_alloc_buffer((2,), data=mx.data, scope="local")
                ls = T.sblock_alloc_buffer((2,), data=ls.data, scope="local")
                if T.tl_shuffle_elect(0):
                    T.ptx_init_barrier_thread_count(mbarrier_1[0], 1)
                    T.ptx_init_barrier_thread_count(mbarrier_1[1], 1)
                    T.ptx_init_barrier_thread_count(mbarrier_1[2], 128)
                    T.ptx_init_barrier_thread_count(mbarrier_1[3], 128)
                    T.ptx_init_barrier_thread_count(mbarrier_1[4], 1)
                T.ptx_fence_barrier_init()
                T.tvm_storage_sync("shared")
                if T.tl_shuffle_elect(256):
                    T.mbarrier_expect_tx(mbarrier_1[4], T.int64(4096))
                    T.tma_load(T.create_tma_descriptor(9, 4, Q.data, 32, 385, 4, 2, T.int64(2), T.int64(64), T.int64(24640), T.int64(98560), 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0), mbarrier_1[4], T.tvm_access_ptr(T.type_annotation("bfloat16"), Qs.data, 0, 2048, 2), 0, bx * 64, by, bz, 0)
                    T.ptx_arrive_barrier(mbarrier_1[4])
                T.attr([128, 128], "kWarpSpecializationScope", 0)
                if tx < 128:
                    for k in range(T.min(2, bx - 3)):
                        T.mbarrier_wait_parity(mbarrier_1[2], T.bitwise_xor(k, 1))
                        if T.tl_shuffle_elect(128):
                            T.mbarrier_expect_tx(mbarrier_1[0], T.int64(4096))
                            T.tma_load(T.create_tma_descriptor(9, 4, K.data, 32, 127, 4, 2, T.int64(2), T.int64(64), T.int64(8128), T.int64(32512), 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0), mbarrier_1[0], T.tvm_access_ptr(T.type_annotation("bfloat16"), Ks.data, 0, 2048, 2), 0, k * 64, by, bz, 0)
                            T.ptx_arrive_barrier(mbarrier_1[0])
                        T.mbarrier_wait_parity(mbarrier_1[3], T.bitwise_xor(k, 1))
                        if T.tl_shuffle_elect(128):
                            T.mbarrier_expect_tx(mbarrier_1[1], T.int64(4096))
                            T.tma_load(T.create_tma_descriptor(9, 4, V.data, 32, 127, 4, 2, T.int64(2), T.int64(64), T.int64(8128), T.int64(32512), 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0), mbarrier_1[1], T.tvm_access_ptr(T.type_annotation("bfloat16"), Vs.data, 0, 2048, 2), 0, k * 64, by, bz, 0)
                            T.ptx_arrive_barrier(mbarrier_1[1])
                else:
                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(4):
                            o[i * 4 + vec] = T.float32(0.0)
                    for i in T.vectorized(2):
                        ls[i] = T.float32(0.0)
                    for i in T.vectorized(2):
                        mx[i] = T.infinity("float32") * T.float32(-1.0)
                    for k in range(T.min(2, bx - 3)):
                        with T.sblock(""):
                            T.reads(mbarrier_1[0:5], Qs[0, 0:8, 0:256], Ks[0, 0:8, 0:256], mx[0:2], Vs[0, 0:8, 0:256], ls[0:2], o[0:16])
                            T.writes(mx[0:2], ls[0:2], o[0:16])
                            s = T.sblock_alloc_buffer((32,), data=s.data, scope="local")
                            sc = T.sblock_alloc_buffer((32,), "bfloat16", data=sc.data, scope="local")
                            mp = T.sblock_alloc_buffer((2,), data=mp.data, scope="local")
                            ss = T.sblock_alloc_buffer((2,), data=ss.data, scope="local")
                            sm = T.sblock_alloc_buffer((2,), data=sm.data, scope="local")
                            for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                s[i] = T.if_then_else(k * 64 + i // 4 * 8 + tx % 4 * 2 + i % 2 + 322 <= bx * 64 + tx // 32 * 16 + i % 4 // 2 * 8 + tx % 32 // 4, T.float32(0.0), T.infinity("float32") * T.float32(-1.0))
                            if k == 0:
                                T.mbarrier_wait_parity(mbarrier_1[4], 0)
                            T.mbarrier_wait_parity(mbarrier_1[0], k)
                            with T.sblock("_gemm_ssr"):
                                T.reads()
                                T.writes()
                                T.sblock_attr({"lexical_alloc_scope": 1})
                                desc_a = T.sblock_alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                                desc_b = T.sblock_alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                                T.initialize_wgmma_descriptor(desc_a[0], T.tvm_access_ptr(T.type_annotation("bfloat16"), Qs.data, 0, 2048, 1), 2, 1, 32)
                                T.initialize_wgmma_descriptor(desc_b[0], T.tvm_access_ptr(T.type_annotation("bfloat16"), Ks.data, 0, 2048, 1), 2, 1, 32)
                                T.warpgroup_fence_operand("float32", s.data, 0, 32)
                                T.warpgroup_arrive()
                                for warp_j in T.unroll(1, annotations={"pragma_unroll_explicit": False}):
                                    for i in T.unroll(1, annotations={"pragma_unroll_explicit": False}):
                                        for ki in T.unroll(2, annotations={"pragma_unroll_explicit": False}):
                                            T.ptx_wgmma_ss("m64n64k16", T.bool(True), T.bool(True), "bf16", "bf16", "fp32", desc_a.data, T.shift_right(ki * 32, 4), desc_b.data, T.shift_right(ki * 32, 4), s.data, 0, 1, 1, 1)
                                T.warpgroup_commit_batch()
                                T.warpgroup_wait(0)
                                T.warpgroup_fence_operand("float32", s.data, 0, 32)
                            T.ptx_arrive_barrier(mbarrier_1[2])
                            for i in T.vectorized(2):
                                mp[i] = mx[i]
                            for i in T.vectorized(2):
                                mx[i] = T.infinity("float32") * T.float32(-1.0)
                            mx_clear = T.alloc_buffer((2,), scope="local")
                            for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                mx_clear[i] = T.float32("-inf")
                                for rv in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                    mx_clear[i] = T.max(mx_clear[i], s[rv % 8 * 4 + i * 2 + rv // 8])
                                mx_clear[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<128>>::run", mx_clear[i])
                                mx[i] = T.max(mx[i], mx_clear[i])
                            for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                ss[i] = T.exp2(mp[i] * T.float32(0.2550348614920494) - mx[i] * T.float32(0.2550348614920494))
                            for i in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                o[i] = o[i] * ss[i % 4 // 2]
                            for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                s[i] = T.exp2(s[i] * T.float32(0.2550348614920494) - mx[i % 4 // 2] * T.float32(0.2550348614920494))
                            for i in T.unroll(8, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                for vec in T.vectorized(4):
                                    sc[i * 4 + vec] = T.Cast("bfloat16", s[i * 4 + vec])
                            T.mbarrier_wait_parity(mbarrier_1[1], k)
                            with T.sblock("_gemm_rsr"):
                                T.reads()
                                T.writes()
                                T.sblock_attr({"lexical_alloc_scope": 1})
                                desc_b = T.sblock_alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                                T.initialize_wgmma_descriptor(desc_b[0], T.tvm_access_ptr(T.type_annotation("bfloat16"), Vs.data, 0, 2048, 1), 2, 0, 32)
                                T.warpgroup_fence_operand("bfloat16", sc.data, 0, 16)
                                T.warpgroup_fence_operand("float32", o.data, 0, 16)
                                T.warpgroup_arrive()
                                for warp_j in T.unroll(1, annotations={"pragma_unroll_explicit": False}):
                                    for i in T.unroll(1, annotations={"pragma_unroll_explicit": False}):
                                        for ki in T.unroll(4, annotations={"pragma_unroll_explicit": False}):
                                            T.ptx_wgmma_rs("m64n32k16", T.bool(False), "bf16", "bf16", "fp32", sc.data, ki * 8, desc_b.data, T.shift_right(ki * 1024, 4), o.data, 0, 1, 1, 1)
                                T.warpgroup_commit_batch()
                                T.warpgroup_wait(0)
                                T.warpgroup_fence_operand("float32", o.data, 0, 16)
                                T.warpgroup_fence_operand("bfloat16", sc.data, 0, 16)
                            T.ptx_arrive_barrier(mbarrier_1[3])
                            for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                sm[i] = T.float32(0.0)
                                for rv in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                    sm[i] = sm[i] + s[rv % 8 * 4 + i * 2 + rv // 8]
                                sm[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run", sm[i])
                            for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                ls[i] = ls[i] * ss[i] + sm[i]
                    for i in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        o[i] = o[i] / ls[i % 4 // 2]
                    for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        T.ptx_stmatrix(0, 4, T.tvm_access_ptr(T.type_annotation("bfloat16"), Os.data, tx % 128 // 32 * 512 + (tx % 16 * 32 + (tx % 8 // 4 + i) % 2 * 16) // 256 * 256 + tx % 8 * 32 + (tx % 8 // 4 + i) % 2 * 16 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 8, 8, 2), T.pack_b16(T.Cast("bfloat16", o[i * 8]), T.Cast("bfloat16", o[i * 8 + 1])), T.pack_b16(T.Cast("bfloat16", o[i * 8 + 2]), T.Cast("bfloat16", o[i * 8 + 3])), T.pack_b16(T.Cast("bfloat16", o[i * 8 + 4]), T.Cast("bfloat16", o[i * 8 + 5])), T.pack_b16(T.Cast("bfloat16", o[i * 8 + 6]), T.Cast("bfloat16", o[i * 8 + 7])), "m8n8")
                    if T.tl_shuffle_elect(128):
                        T.tma_store(T.create_tma_descriptor(9, 4, O.data, 32, 385, 4, 2, T.int64(2), T.int64(64), T.int64(24640), T.int64(98560), 32, 64, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0), T.tvm_access_ptr(T.type_annotation("bfloat16"), Os.data, 0, 2048, 1), 0, bx * 64, by, bz, 0, 0)
                        T.tma_store_arrive()
                        T.tma_store_wait(0, T.bool(True))

# Metadata omitted. Use show_meta=True in script() method to show it.