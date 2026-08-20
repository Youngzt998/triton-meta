#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(float* __restrict__ a_out);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(float* __restrict__ a_out) {
  float a_fp32_local[16];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float4 v_ = *(float4*)(a_fp32_local + (i * 4));
    tl::store_global_128((&(a_out[((i * 512) + (((int)threadIdx.x) * 4))])), (*(uint4 *)(&(v_))));
  }
}

