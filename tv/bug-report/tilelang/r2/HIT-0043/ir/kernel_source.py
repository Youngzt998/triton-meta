import tilelang
import tilelang.language as T


@T.prim_func
def main(Q: T.Tensor((2, 4, 385, 32), "bfloat16"),
         K: T.Tensor((2, 4, 127, 32), "bfloat16"),
         V: T.Tensor((2, 4, 127, 32), "bfloat16"),
         O: T.Tensor((2, 4, 385, 32), "bfloat16")):
    with T.Kernel(T.ceildiv(385, 64), 4, 2, threads=128) as (bx, by, bz):
        Qs = T.alloc_shared([64, 32], "bfloat16")
        Ks = T.alloc_shared([64, 32], "bfloat16")
        Vs = T.alloc_shared([64, 32], "bfloat16")
        Os = T.alloc_shared([64, 32], "bfloat16")
        s = T.alloc_fragment([64, 64], "float32")
        sc = T.alloc_fragment([64, 64], "bfloat16")
        o = T.alloc_fragment([64, 32], "float32")
        mx = T.alloc_fragment([64], "float32")
        mp = T.alloc_fragment([64], "float32")
        ss = T.alloc_fragment([64], "float32")
        sm = T.alloc_fragment([64], "float32")
        ls = T.alloc_fragment([64], "float32")
        T.copy(Q[bz, by, bx * 64:(bx + 1) * 64, :], Qs)
        T.fill(o, 0)
        T.fill(ls, 0)
        T.fill(mx, -T.infinity("float32"))
        for k in T.Pipelined(T.min(T.ceildiv(127, 64), T.ceildiv((bx + 1) * 64 + -258, 64)), num_stages=1):
            T.copy(K[bz, by, k * 64:(k + 1) * 64, :], Ks)
            for i, j in T.Parallel(64, 64):
                s[i, j] = T.if_then_else(bx * 64 + i + -258 >= k * 64 + j, 0, -T.infinity("float32"))
            T.gemm(Qs, Ks, s, transpose_B=True, policy=T.GemmWarpPolicy.Square)
            T.copy(V[bz, by, k * 64:(k + 1) * 64, :], Vs)
            T.copy(mx, mp)
            T.fill(mx, -T.infinity("float32"))
            T.reduce_max(s, mx, dim=1, clear=False)
            for i in T.Parallel(64):
                ss[i] = T.exp2(mp[i] * 0.2550348614920494 - mx[i] * 0.2550348614920494)
            for i, j in T.Parallel(64, 32):
                o[i, j] *= ss[i]
            for i, j in T.Parallel(64, 64):
                s[i, j] = T.exp2(s[i, j] * 0.2550348614920494 - mx[i] * 0.2550348614920494)
            T.copy(s, sc)
            T.gemm(sc, Vs, o, policy=T.GemmWarpPolicy.Square)
            T.reduce_sum(s, sm, dim=1)
            for i in T.Parallel(64):
                ls[i] = ls[i] * ss[i] + sm[i]
        for i, j in T.Parallel(64, 32):
            o[i, j] /= ls[i]
        T.copy(o, Os)
        T.copy(Os, O[bz, by, bx * 64:(bx + 1) * 64, :])
