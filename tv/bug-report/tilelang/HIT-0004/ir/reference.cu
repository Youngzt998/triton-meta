#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <curand_kernel.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void rand_kernel_kernel(uint* __restrict__ A, float* __restrict__ B, double* __restrict__ C, float* __restrict__ D, double* __restrict__ E);
extern "C" __global__ void rand_kernel_kernel(uint* __restrict__ A, float* __restrict__ B, double* __restrict__ C, float* __restrict__ D, double* __restrict__ E) {
  curandStatePhilox4_32_10_t __random_generator_state;
  curand_init(123, 0, (((int)blockIdx.x) * 128), &__random_generator_state);
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    *(uint4*)(A + ((((int)blockIdx.x) * 128) + (i * 4))) = curand4(&__random_generator_state);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    *(float4*)(B + ((((int)blockIdx.x) * 128) + (i_1 * 4))) = curand_uniform4(&__random_generator_state);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 64; ++i_2) {
    *(double2*)(C + ((((int)blockIdx.x) * 128) + (i_2 * 2))) = curand_uniform2_double(&__random_generator_state);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 32; ++i_3) {
    *(float4*)(D + ((((int)blockIdx.x) * 128) + (i_3 * 4))) = curand_normal4(&__random_generator_state);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 64; ++i_4) {
    *(double2*)(E + ((((int)blockIdx.x) * 128) + (i_4 * 2))) = curand_normal2_double(&__random_generator_state);
  }
}

