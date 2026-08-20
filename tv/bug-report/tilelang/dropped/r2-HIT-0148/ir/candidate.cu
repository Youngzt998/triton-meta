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

extern "C" __global__ void main_kernel(const float* __restrict__ A, const float* __restrict__ X, float* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(32, 1) main_kernel(const float* __restrict__ A, const float* __restrict__ X, float* __restrict__ Y) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* As = ((void*)((char*)buf_dyn_shmem + 0));
  void* Xs = ((void*)((char*)buf_dyn_shmem + 32768));
  float acc[2];
  float p[256];
  float acc_clear[64];
  float acc_clear_1[64];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(acc + 0) = make_float2(broadcast_var, broadcast_var);
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    for (int vec = 0; vec < 4; ++vec) {
      tl::cp_async_gs_conditional<4>((&(((float*)As)[(((i * 128) + (((int)threadIdx.x) * 4)) + vec)])), (&(A[(((i * 1027) + (((int)threadIdx.x) * 4)) + vec)])), ((i < 31) && (i < 31)));
    }
  }
  tl::cp_async_gs<16>((&(((float*)Xs)[(((int)threadIdx.x) * 4)])), (&(X[(((int)threadIdx.x) * 4)])));
  tl::cp_async_commit();
  for (int ko = 0; ko < 8; ++ko) {
    tl::cp_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      float4 __1;
        float4 v_ = *(float4*)(((float*)As) + ((i_1 * 128) + (((int)threadIdx.x) * 4)));
        float4 v__1 = *(float4*)(((float*)Xs) + (((int)threadIdx.x) * 4));
        __1.x = (v_.x*v__1.x);
        __1.y = (v_.y*v__1.y);
        __1.z = (v_.z*v__1.z);
        __1.w = (v_.w*v__1.w);
      *(float4*)(p + (i_1 * 4)) = __1;
    }
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 256; ++i_2) {
      tl::cp_async_gs_conditional<4>((&(((float*)As)[((i_2 * 32) + ((int)threadIdx.x))])), (&(A[((((((i_2 >> 2) * 1027) + (ko * 128)) + ((i_2 & 3) * 32)) + ((int)threadIdx.x)) + 128)])), ((((i_2 < 124) && ((((ko * 128) + ((i_2 & 3) * 32)) + ((int)threadIdx.x)) < 899)) && (i_2 < 124)) && ((((ko * 128) + ((i_2 & 3) * 32)) + ((int)threadIdx.x)) < 899)));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs_conditional<4>((&(((float*)Xs)[((i_3 * 32) + ((int)threadIdx.x))])), (&(X[((((ko * 128) + (i_3 * 32)) + ((int)threadIdx.x)) + 128)])), (((((ko * 128) + (i_3 * 32)) + ((int)threadIdx.x)) < 899) && ((((ko * 128) + (i_3 * 32)) + ((int)threadIdx.x)) < 899)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      acc_clear[i_4] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        acc_clear[i_4] = (acc_clear[i_4] + p[((i_4 * 4) + rv)]);
      }
      acc_clear[i_4] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<32>>::run(acc_clear[i_4]);
      if ((((i_4 >> 5) * 32) + ((int)threadIdx.x)) == i_4) {
        acc[(i_4 >> 5)] = (acc[(i_4 >> 5)] + acc_clear[i_4]);
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 64; ++i_5) {
    float4 __2;
      float4 v__2 = *(float4*)(((float*)As) + ((i_5 * 128) + (((int)threadIdx.x) * 4)));
      float4 v__3 = *(float4*)(((float*)Xs) + (((int)threadIdx.x) * 4));
      __2.x = (v__2.x*v__3.x);
      __2.y = (v__2.y*v__3.y);
      __2.z = (v__2.z*v__3.z);
      __2.w = (v__2.w*v__3.w);
    *(float4*)(p + (i_5 * 4)) = __2;
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 64; ++i_6) {
    acc_clear_1[i_6] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      acc_clear_1[i_6] = (acc_clear_1[i_6] + p[((i_6 * 4) + rv_1)]);
    }
    acc_clear_1[i_6] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<32>>::run(acc_clear_1[i_6]);
    if ((((i_6 >> 5) * 32) + ((int)threadIdx.x)) == i_6) {
      acc[(i_6 >> 5)] = (acc[(i_6 >> 5)] + acc_clear_1[i_6]);
    }
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    if (((i_7 * 32) + ((int)threadIdx.x)) < 31) {
      Y[((i_7 * 32) + ((int)threadIdx.x))] = acc[i_7];
    }
  }
}

