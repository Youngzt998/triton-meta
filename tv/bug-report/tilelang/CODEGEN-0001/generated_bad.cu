#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const int* __restrict__ A, int* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const int* __restrict__ A, int* __restrict__ C) {
  extern __shared__ __align__(1024) int As[];
  int Cl[32];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(int4*)(As + ((i * 512) + (((int)threadIdx.x) * 4))) = *(int4*)(A + ((i * 512) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    int broadcast_var = 0;
    int broadcast_var_1 = 2;
    int broadcast_var_2 = 2;
    int broadcast_var_3 = 2;
    int broadcast_var_4 = 1;
    ushort4 __1;
      int4 v_ = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
      int4 __2;
        int4 v__1 = *(int4*)(As + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
        int4 v__2 = make_int4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
        __2.x = (v__1.x%v__2.x);
        __2.y = (v__1.y%v__2.y);
        __2.z = (v__1.z%v__2.z);
        __2.w = (v__1.w%v__2.w);
      __1.x = (v_.x<=__2.x);
      __1.y = (v_.y<=__2.y);
      __1.z = (v_.z<=__2.z);
      __1.w = (v_.w<=__2.w);
    int4 __3;
      int4 v__3 = *(int4*)(As + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
      int4 v__4 = make_int4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
      __3.x = (v__3.x/v__4.x);
      __3.y = (v__3.y/v__4.y);
      __3.z = (v__3.z/v__4.z);
      __3.w = (v__3.w/v__4.w);
    int4 __4;
      int4 __5;
        int4 v__5 = *(int4*)(As + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
        int4 v__6 = make_int4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
        __5.x = (v__5.x/v__6.x);
        __5.y = (v__5.y/v__6.y);
        __5.z = (v__5.z/v__6.z);
        __5.w = (v__5.w/v__6.w);
      int4 v__7 = make_int4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
      __4.x = (__5.x-v__7.x);
      __4.y = (__5.y-v__7.y);
      __4.z = (__5.z-v__7.z);
      __4.w = (__5.w-v__7.w);
    *(int4*)(Cl + (i_1 * 4)) = (__1 ? __3 : __4);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    *(int4*)(C + ((i_2 * 512) + (((int)threadIdx.x) * 4))) = *(int4*)(Cl + (i_2 * 4));
  }
}

