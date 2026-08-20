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
  void* As = ((void*)((char*)buf_dyn_shmem + (int64_t)0));
  void* Xs = ((void*)((char*)buf_dyn_shmem + (int64_t)196608));
  float acc[2];
  float p[512];
  float acc_clear[64];
  float acc_clear_1[64];
  float acc_clear_2[64];
  float acc_clear_3[64];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(acc + (int64_t)0) = make_float2(broadcast_var, broadcast_var);
  #pragma unroll
  for (int64_t i = (int64_t)0; i < (int64_t)128; ++i) {
    for (int64_t vec = (int64_t)0; vec < (int64_t)4; ++vec) {
      tl::cp_async_gs_conditional<4>((&(((float*)As)[(((i * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + vec)])), (&(A[(((((((int64_t)blockIdx.x) * (int64_t)65600) + ((i >> (int64_t)1) * (int64_t)1025)) + ((i & (int64_t)1) * (int64_t)128)) + (((int64_t)threadIdx.x) * (int64_t)4)) + vec)])), ((((((int64_t)blockIdx.x) * (int64_t)64) + (i >> (int64_t)1)) < (int64_t)511) && (((((int64_t)blockIdx.x) * (int64_t)64) + (i >> (int64_t)1)) < (int64_t)511)));
    }
  }
  #pragma unroll
  for (int64_t i_1 = (int64_t)0; i_1 < (int64_t)2; ++i_1) {
    tl::cp_async_gs<16>((&(((float*)Xs)[((i_1 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4))])), (&(X[((i_1 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int64_t i_2 = (int64_t)0; i_2 < (int64_t)128; ++i_2) {
    for (int64_t vec_1 = (int64_t)0; vec_1 < (int64_t)4; ++vec_1) {
      tl::cp_async_gs_conditional<4>((&(((float*)As)[((((i_2 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + vec_1) + (int64_t)16384)])), (&(A[((((((((int64_t)blockIdx.x) * (int64_t)65600) + ((i_2 >> (int64_t)1) * (int64_t)1025)) + ((i_2 & (int64_t)1) * (int64_t)128)) + (((int64_t)threadIdx.x) * (int64_t)4)) + vec_1) + (int64_t)256)])), ((((((int64_t)blockIdx.x) * (int64_t)64) + (i_2 >> (int64_t)1)) < (int64_t)511) && (((((int64_t)blockIdx.x) * (int64_t)64) + (i_2 >> (int64_t)1)) < (int64_t)511)));
    }
  }
  #pragma unroll
  for (int64_t i_3 = (int64_t)0; i_3 < (int64_t)2; ++i_3) {
    tl::cp_async_gs<16>((&(((float*)Xs)[(((i_3 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + (int64_t)256)])), (&(X[(((i_3 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + (int64_t)256)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int64_t i_4 = (int64_t)0; i_4 < (int64_t)128; ++i_4) {
    for (int64_t vec_2 = (int64_t)0; vec_2 < (int64_t)4; ++vec_2) {
      tl::cp_async_gs_conditional<4>((&(((float*)As)[((((i_4 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + vec_2) + (int64_t)32768)])), (&(A[((((((((int64_t)blockIdx.x) * (int64_t)65600) + ((i_4 >> (int64_t)1) * (int64_t)1025)) + ((i_4 & (int64_t)1) * (int64_t)128)) + (((int64_t)threadIdx.x) * (int64_t)4)) + vec_2) + (int64_t)512)])), ((((((int64_t)blockIdx.x) * (int64_t)64) + (i_4 >> (int64_t)1)) < (int64_t)511) && (((((int64_t)blockIdx.x) * (int64_t)64) + (i_4 >> (int64_t)1)) < (int64_t)511)));
    }
  }
  #pragma unroll
  for (int64_t i_5 = (int64_t)0; i_5 < (int64_t)2; ++i_5) {
    tl::cp_async_gs<16>((&(((float*)Xs)[(((i_5 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + (int64_t)512)])), (&(X[(((i_5 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + (int64_t)512)])));
  }
  tl::cp_async_commit();
  for (int64_t ko = (int64_t)0; ko < (int64_t)2; ++ko) {
    tl::cp_async_wait<2>();
    __syncthreads();
    #pragma unroll
    for (int64_t i_6 = (int64_t)0; i_6 < (int64_t)128; ++i_6) {
      float4 __1;
        float4 v_ = *(float4*)(((float*)As) + (((ko * (int64_t)16384) + (i_6 * (int64_t)128)) + (((int64_t)threadIdx.x) * (int64_t)4)));
        float4 v__1 = *(float4*)(((float*)Xs) + (((ko * (int64_t)256) + ((i_6 & (int64_t)1) * (int64_t)128)) + (((int64_t)threadIdx.x) * (int64_t)4)));
        __1.x = (v_.x*v__1.x);
        __1.y = (v_.y*v__1.y);
        __1.z = (v_.z*v__1.z);
        __1.w = (v_.w*v__1.w);
      *(float4*)(p + (i_6 * (int64_t)4)) = __1;
    }
    __syncthreads();
    #pragma unroll
    for (int64_t i_7 = (int64_t)0; i_7 < (int64_t)512; ++i_7) {
      tl::cp_async_gs_conditional<4>((&(((float*)As)[(((ko * (int64_t)16384) + (i_7 * (int64_t)32)) + ((int64_t)threadIdx.x))])), (&(A[((((((((int64_t)blockIdx.x) * (int64_t)65600) + ((i_7 >> (int64_t)3) * (int64_t)1025)) + (ko * (int64_t)256)) + ((i_7 & (int64_t)7) * (int64_t)32)) + ((int64_t)threadIdx.x)) + (int64_t)768)])), ((((((((int64_t)blockIdx.x) * (int64_t)64) + (i_7 >> (int64_t)3)) < (int64_t)511) && ((((ko * (int64_t)256) + ((i_7 & (int64_t)7) * (int64_t)32)) + ((int64_t)threadIdx.x)) < (int64_t)257)) && (((((int64_t)blockIdx.x) * (int64_t)64) + (i_7 >> (int64_t)3)) < (int64_t)511)) && ((((ko * (int64_t)256) + ((i_7 & (int64_t)7) * (int64_t)32)) + ((int64_t)threadIdx.x)) < (int64_t)257)));
    }
    #pragma unroll
    for (int64_t i_8 = (int64_t)0; i_8 < (int64_t)8; ++i_8) {
      tl::cp_async_gs_conditional<4>((&(((float*)Xs)[(((ko * (int64_t)256) + (i_8 * (int64_t)32)) + ((int64_t)threadIdx.x))])), (&(X[((((ko * (int64_t)256) + (i_8 * (int64_t)32)) + ((int64_t)threadIdx.x)) + (int64_t)768)])), (((((ko * (int64_t)256) + (i_8 * (int64_t)32)) + ((int64_t)threadIdx.x)) < (int64_t)257) && ((((ko * (int64_t)256) + (i_8 * (int64_t)32)) + ((int64_t)threadIdx.x)) < (int64_t)257)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int64_t i_9 = (int64_t)0; i_9 < (int64_t)64; ++i_9) {
      acc_clear[i_9] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int64_t rv = (int64_t)0; rv < (int64_t)8; ++rv) {
        acc_clear[i_9] = (acc_clear[i_9] + p[(((i_9 * (int64_t)8) + ((rv & (int64_t)1) * (int64_t)4)) + (rv >> (int64_t)1))]);
      }
      acc_clear[i_9] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<32>>::run(acc_clear[i_9]);
      if ((((i_9 >> (int64_t)5) * (int64_t)32) + ((int64_t)threadIdx.x)) == i_9) {
        acc[(i_9 >> (int64_t)5)] = (acc[(i_9 >> (int64_t)5)] + acc_clear[i_9]);
      }
    }
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  #pragma unroll
  for (int64_t i_10 = (int64_t)0; i_10 < (int64_t)128; ++i_10) {
    float4 __2;
      float4 v__2 = *(float4*)(((float*)As) + (((i_10 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + (int64_t)32768));
      float4 v__3 = *(float4*)(((float*)Xs) + ((((i_10 & (int64_t)1) * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + (int64_t)512));
      __2.x = (v__2.x*v__3.x);
      __2.y = (v__2.y*v__3.y);
      __2.z = (v__2.z*v__3.z);
      __2.w = (v__2.w*v__3.w);
    *(float4*)(p + (i_10 * (int64_t)4)) = __2;
  }
  #pragma unroll
  for (int64_t i_11 = (int64_t)0; i_11 < (int64_t)64; ++i_11) {
    acc_clear_1[i_11] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int64_t rv_1 = (int64_t)0; rv_1 < (int64_t)8; ++rv_1) {
      acc_clear_1[i_11] = (acc_clear_1[i_11] + p[(((i_11 * (int64_t)8) + ((rv_1 & (int64_t)1) * (int64_t)4)) + (rv_1 >> (int64_t)1))]);
    }
    acc_clear_1[i_11] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<32>>::run(acc_clear_1[i_11]);
    if ((((i_11 >> (int64_t)5) * (int64_t)32) + ((int64_t)threadIdx.x)) == i_11) {
      acc[(i_11 >> (int64_t)5)] = (acc[(i_11 >> (int64_t)5)] + acc_clear_1[i_11]);
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int64_t i_12 = (int64_t)0; i_12 < (int64_t)128; ++i_12) {
    float4 __3;
      float4 v__4 = *(float4*)(((float*)As) + ((i_12 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)));
      float4 v__5 = *(float4*)(((float*)Xs) + (((i_12 & (int64_t)1) * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)));
      __3.x = (v__4.x*v__5.x);
      __3.y = (v__4.y*v__5.y);
      __3.z = (v__4.z*v__5.z);
      __3.w = (v__4.w*v__5.w);
    *(float4*)(p + (i_12 * (int64_t)4)) = __3;
  }
  #pragma unroll
  for (int64_t i_13 = (int64_t)0; i_13 < (int64_t)64; ++i_13) {
    acc_clear_2[i_13] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int64_t rv_2 = (int64_t)0; rv_2 < (int64_t)8; ++rv_2) {
      acc_clear_2[i_13] = (acc_clear_2[i_13] + p[(((i_13 * (int64_t)8) + ((rv_2 & (int64_t)1) * (int64_t)4)) + (rv_2 >> (int64_t)1))]);
    }
    acc_clear_2[i_13] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<32>>::run(acc_clear_2[i_13]);
    if ((((i_13 >> (int64_t)5) * (int64_t)32) + ((int64_t)threadIdx.x)) == i_13) {
      acc[(i_13 >> (int64_t)5)] = (acc[(i_13 >> (int64_t)5)] + acc_clear_2[i_13]);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int64_t i_14 = (int64_t)0; i_14 < (int64_t)128; ++i_14) {
    float4 __4;
      float4 v__6 = *(float4*)(((float*)As) + (((i_14 * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + (int64_t)16384));
      float4 v__7 = *(float4*)(((float*)Xs) + ((((i_14 & (int64_t)1) * (int64_t)128) + (((int64_t)threadIdx.x) * (int64_t)4)) + (int64_t)256));
      __4.x = (v__6.x*v__7.x);
      __4.y = (v__6.y*v__7.y);
      __4.z = (v__6.z*v__7.z);
      __4.w = (v__6.w*v__7.w);
    *(float4*)(p + (i_14 * (int64_t)4)) = __4;
  }
  #pragma unroll
  for (int64_t i_15 = (int64_t)0; i_15 < (int64_t)64; ++i_15) {
    acc_clear_3[i_15] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int64_t rv_3 = (int64_t)0; rv_3 < (int64_t)8; ++rv_3) {
      acc_clear_3[i_15] = (acc_clear_3[i_15] + p[(((i_15 * (int64_t)8) + ((rv_3 & (int64_t)1) * (int64_t)4)) + (rv_3 >> (int64_t)1))]);
    }
    acc_clear_3[i_15] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<32>>::run(acc_clear_3[i_15]);
    if ((((i_15 >> (int64_t)5) * (int64_t)32) + ((int64_t)threadIdx.x)) == i_15) {
      acc[(i_15 >> (int64_t)5)] = (acc[(i_15 >> (int64_t)5)] + acc_clear_3[i_15]);
    }
  }
  #pragma unroll
  for (int64_t i_16 = (int64_t)0; i_16 < (int64_t)2; ++i_16) {
    if ((((((int64_t)blockIdx.x) * (int64_t)64) + (i_16 * (int64_t)32)) + ((int64_t)threadIdx.x)) < (int64_t)511) {
      Y[(((((int64_t)blockIdx.x) * (int64_t)64) + (i_16 * (int64_t)32)) + ((int64_t)threadIdx.x))] = acc[i_16];
    }
  }
}

