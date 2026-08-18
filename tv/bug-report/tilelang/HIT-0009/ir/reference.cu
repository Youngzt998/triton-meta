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

extern "C" __global__ void main_kernel(const float* __restrict__ A, float* __restrict__ B, const float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ A, float* __restrict__ B, const float* __restrict__ C) {
  for (int col = 0; col < 64; ++col) {
    float2 __1;
      float2 v_ = *(float2*)(A + ((((int)threadIdx.x) * 128) + (col * 2)));
      float2 v__1 = make_float2(C[((((int)threadIdx.x) * 64) + col)], C[((((int)threadIdx.x) * 64) + col)]);
      __1.x = (v_.x*v__1.x);
      __1.y = (v_.y*v__1.y);
    *(float2*)(B + ((((int)threadIdx.x) * 128) + (col * 2))) = __1;
  }
}

