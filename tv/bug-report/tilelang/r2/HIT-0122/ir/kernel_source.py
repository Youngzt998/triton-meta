import tilelang
import tilelang.language as T


@T.prim_func
def main(Q: T.Tensor((1, 4, 192, 32), "bfloat16"),
         K: T.Tensor((1, 4, 768, 32), "bfloat16"),
         V: T.Tensor((1, 4, 768, 32), "bfloat16"),
         O: T.Tensor((1, 4, 192, 32), "bfloat16")):
    with T.Kernel(T.ceildiv(192, 128), 4, 1, threads=128) as (bx, by, bz):
        Qs = T.alloc_shared([128, 32], "bfloat16")
        Ks = T.alloc_shared([128, 32], "bfloat16")
        Vs = T.alloc_shared([128, 32], "bfloat16")
        Os = T.alloc_shared([128, 32], "bfloat16")
        s = T.alloc_fragment([128, 128], "float32")
        sc = T.alloc_fragment([128, 128], "bfloat16")
        o = T.alloc_fragment([128, 32], "float32")
        mx = T.alloc_fragment([128], "float32")
        mp = T.alloc_fragment([128], "float32")
        ss = T.alloc_fragment([128], "float32")
        sm = T.alloc_fragment([128], "float32")
        ls = T.alloc_fragment([128], "float32")
        T.copy(Q[bz, by, bx * 128:(bx + 1) * 128, :], Qs)
        T.fill(o, 0)
        T.fill(ls, 0)
        T.fill(mx, -T.infinity("float32"))
        for k in T.Pipelined(T.ceildiv(768, 128), num_stages=0):
            T.copy(K[bz, by, k * 128:(k + 1) * 128, :], Ks)
            T.clear(s)
            T.gemm(Qs, Ks, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(V[bz, by, k * 128:(k + 1) * 128, :], Vs)
            T.copy(mx, mp)
            T.fill(mx, -T.infinity("float32"))
            T.reduce_max(s, mx, dim=1, clear=False)
            for i in T.Parallel(128):
                ss[i] = T.exp2(mp[i] * 0.2550348614920494 - mx[i] * 0.2550348614920494)
            for i, j in T.Parallel(128, 32):
                o[i, j] *= ss[i]
            for i, j in T.Parallel(128, 128):
                s[i, j] = T.exp2(s[i, j] * 0.2550348614920494 - mx[i] * 0.2550348614920494)
            T.copy(s, sc)
            T.gemm(sc, Vs, o, policy=T.GemmWarpPolicy.FullRow)
            T.reduce_sum(s, sm, dim=1)
            for i in T.Parallel(128):
                ls[i] = ls[i] * ss[i] + sm[i]
        for i, j in T.Parallel(128, 32):
            o[i, j] /= ls[i]
        T.copy(o, Os)
        T.copy(Os, O[bz, by, bx * 128:(bx + 1) * 128, :])
