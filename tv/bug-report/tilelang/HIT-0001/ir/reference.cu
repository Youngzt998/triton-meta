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

extern "C" __global__ void main_kernel(float* __restrict__ decoded_values, const uint* __restrict__ packed_values);
extern "C" __global__ void main_kernel(float* __restrict__ decoded_values, const uint* __restrict__ packed_values) {
  for (int i = 0; i < 7; ++i) {
    decoded_values[i] = ((float)((((int)((packed_values[(i >> 2)] >> ((uint)((i & 3) * 8))) & (uint)255)) << 24) >> 24));
  }
}

