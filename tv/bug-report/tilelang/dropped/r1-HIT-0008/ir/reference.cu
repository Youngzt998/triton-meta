#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void cumsum_kernel(const float* __restrict__ A, __grid_constant__ const CUtensorMap B_desc);
extern "C" __global__ void __launch_bounds__(256, 1) cumsum_kernel(const float* __restrict__ A, __grid_constant__ const CUtensorMap B_desc) {
  extern __shared__ __align__(1024) float A_shared[];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(B_desc);
  }
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(float4*)(A_shared + ((i * 1024) + (((int)threadIdx.x) * 4))) = *(float4*)(A + (((((((int)blockIdx.y) * 10240) + (i * 5120)) + ((((int)threadIdx.x) >> 3) * 160)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)));
  }
  __syncthreads();
  tl::CumSum2D<256, 0, true>::run((&(A_shared[0])), (&(A_shared[0])), 64, 32);
  __syncthreads();
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    tl::tma_store(B_desc, (&(A_shared[0])), (((int)blockIdx.x) * 32), (((int)blockIdx.y) * 64));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

