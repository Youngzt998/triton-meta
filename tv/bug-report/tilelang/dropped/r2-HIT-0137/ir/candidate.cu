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

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ X, float* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(32, 1) main_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ X, float* __restrict__ Y) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* As = ((void*)((char*)buf_dyn_shmem + 0));
  void* Xs = ((void*)((char*)buf_dyn_shmem + 40960));
  float acc[16];
  float p[128];
  bfloat16_t As_local_cast[8];
  bfloat16_t Xs_local_cast_1[8];
  float acc_clear[16];
  bfloat16_t As_local_cast_2[8];
  bfloat16_t Xs_local_cast_3[8];
  float acc_clear_1[16];
  bfloat16_t As_local_cast_4[8];
  bfloat16_t Xs_local_cast_5[8];
  float acc_clear_2[16];
  bfloat16_t As_local_cast_6[8];
  bfloat16_t Xs_local_cast_7[8];
  float acc_clear_3[16];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)As)[((i_1 * 256) + (((int)threadIdx.x) * 8))])), (&(A[((((((int)blockIdx.x) * 14336) + (i_1 * 896)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)threadIdx.x) & 7) * 8))])), (((((((int)blockIdx.x) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 3)) < 2037) && ((((((int)blockIdx.x) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 3)) < 2037)));
  }
  tl::cp_async_gs<4>((&(((bfloat16_t*)Xs)[(((int)threadIdx.x) * 2)])), (&(X[(((int)threadIdx.x) * 2)])));
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)As)[(((i_2 * 256) + (((int)threadIdx.x) * 8)) + 4096)])), (&(A[(((((((int)blockIdx.x) * 14336) + (i_2 * 896)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])), (((((((int)blockIdx.x) * 64) + (i_2 * 4)) + (((int)threadIdx.x) >> 3)) < 2037) && ((((((int)blockIdx.x) * 64) + (i_2 * 4)) + (((int)threadIdx.x) >> 3)) < 2037)));
  }
  tl::cp_async_gs<4>((&(((bfloat16_t*)Xs)[((((int)threadIdx.x) * 2) + 64)])), (&(X[((((int)threadIdx.x) * 2) + 64)])));
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)As)[(((i_3 * 256) + (((int)threadIdx.x) * 8)) + 8192)])), (&(A[(((((((int)blockIdx.x) * 14336) + (i_3 * 896)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)threadIdx.x) & 7) * 8)) + 128)])), (((((((int)blockIdx.x) * 64) + (i_3 * 4)) + (((int)threadIdx.x) >> 3)) < 2037) && ((((((int)blockIdx.x) * 64) + (i_3 * 4)) + (((int)threadIdx.x) >> 3)) < 2037)));
  }
  tl::cp_async_gs<4>((&(((bfloat16_t*)Xs)[((((int)threadIdx.x) * 2) + 128)])), (&(X[((((int)threadIdx.x) * 2) + 128)])));
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 16; ++i_4) {
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)As)[(((i_4 * 256) + (((int)threadIdx.x) * 8)) + 12288)])), (&(A[(((((((int)blockIdx.x) * 14336) + (i_4 * 896)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)threadIdx.x) & 7) * 8)) + 192)])), (((((((((int)blockIdx.x) * 64) + (i_4 * 4)) + (((int)threadIdx.x) >> 3)) < 2037) && ((((int)threadIdx.x) & 7) < 4)) && ((((((int)blockIdx.x) * 64) + (i_4 * 4)) + (((int)threadIdx.x) >> 3)) < 2037)) && ((((int)threadIdx.x) & 7) < 4)));
  }
  tl::cp_async_gs_conditional<4>((&(((bfloat16_t*)Xs)[((((int)threadIdx.x) * 2) + 192)])), (&(X[((((int)threadIdx.x) * 2) + 192)])), ((((int)threadIdx.x) < 16) && (((int)threadIdx.x) < 16)));
  tl::cp_async_commit();
  tl::cp_async_wait<3>();
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 16; ++i_5) {
    *(uint4*)(As_local_cast + 0) = *(uint4*)(((bfloat16_t*)As) + ((i_5 * 256) + (((int)threadIdx.x) * 8)));
    *(uint4*)(Xs_local_cast_1 + 0) = *(uint4*)(((bfloat16_t*)Xs) + ((((int)threadIdx.x) & 7) * 8));
    for (int vec = 0; vec < 2; ++vec) {
      float4 __1;
        float4 __2;
        uint2 v_ = *(uint2*)(As_local_cast + (vec * 4));
        ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
        ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
        float4 __3;
        uint2 v__1 = *(uint2*)(Xs_local_cast_1 + (vec * 4));
        ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
        ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
        __1.x = (__2.x*__3.x);
        __1.y = (__2.y*__3.y);
        __1.z = (__2.z*__3.z);
        __1.w = (__2.w*__3.w);
      *(float4*)(p + ((i_5 * 8) + (vec * 4))) = __1;
    }
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    acc_clear[i_6] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv = 0; rv < 8; ++rv) {
      acc_clear[i_6] = (acc_clear[i_6] + p[((i_6 * 8) + rv)]);
    }
    acc_clear[i_6] = tl::AllReduce<tl::SumOp, 8, 1, 0, tl::NamedBarrier<32>>::run(acc_clear[i_6]);
    acc[i_6] = (acc[i_6] + acc_clear[i_6]);
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    *(uint4*)(As_local_cast_2 + 0) = *(uint4*)(((bfloat16_t*)As) + (((i_7 * 256) + (((int)threadIdx.x) * 8)) + 4096));
    *(uint4*)(Xs_local_cast_3 + 0) = *(uint4*)(((bfloat16_t*)Xs) + (((((int)threadIdx.x) & 7) * 8) + 64));
    for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
      float4 __4;
        float4 __5;
        uint2 v__2 = *(uint2*)(As_local_cast_2 + (vec_1 * 4));
        ((float2*)(&__5))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[0]);
        ((float2*)(&__5))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[1]);
        float4 __6;
        uint2 v__3 = *(uint2*)(Xs_local_cast_3 + (vec_1 * 4));
        ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
        ((float2*)(&__6))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[1]);
        __4.x = (__5.x*__6.x);
        __4.y = (__5.y*__6.y);
        __4.z = (__5.z*__6.z);
        __4.w = (__5.w*__6.w);
      *(float4*)(p + ((i_7 * 8) + (vec_1 * 4))) = __4;
    }
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    acc_clear_1[i_8] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
      acc_clear_1[i_8] = (acc_clear_1[i_8] + p[((i_8 * 8) + rv_1)]);
    }
    acc_clear_1[i_8] = tl::AllReduce<tl::SumOp, 8, 1, 0, tl::NamedBarrier<32>>::run(acc_clear_1[i_8]);
    acc[i_8] = (acc[i_8] + acc_clear_1[i_8]);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_9 = 0; i_9 < 16; ++i_9) {
    *(uint4*)(As_local_cast_4 + 0) = *(uint4*)(((bfloat16_t*)As) + (((i_9 * 256) + (((int)threadIdx.x) * 8)) + 8192));
    *(uint4*)(Xs_local_cast_5 + 0) = *(uint4*)(((bfloat16_t*)Xs) + (((((int)threadIdx.x) & 7) * 8) + 128));
    for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
      float4 __7;
        float4 __8;
        uint2 v__4 = *(uint2*)(As_local_cast_4 + (vec_2 * 4));
        ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
        ((float2*)(&__8))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[1]);
        float4 __9;
        uint2 v__5 = *(uint2*)(Xs_local_cast_5 + (vec_2 * 4));
        ((float2*)(&__9))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[0]);
        ((float2*)(&__9))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[1]);
        __7.x = (__8.x*__9.x);
        __7.y = (__8.y*__9.y);
        __7.z = (__8.z*__9.z);
        __7.w = (__8.w*__9.w);
      *(float4*)(p + ((i_9 * 8) + (vec_2 * 4))) = __7;
    }
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 16; ++i_10) {
    acc_clear_2[i_10] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
      acc_clear_2[i_10] = (acc_clear_2[i_10] + p[((i_10 * 8) + rv_2)]);
    }
    acc_clear_2[i_10] = tl::AllReduce<tl::SumOp, 8, 1, 0, tl::NamedBarrier<32>>::run(acc_clear_2[i_10]);
    acc[i_10] = (acc[i_10] + acc_clear_2[i_10]);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_11 = 0; i_11 < 16; ++i_11) {
    *(uint4*)(As_local_cast_6 + 0) = *(uint4*)(((bfloat16_t*)As) + (((i_11 * 256) + (((int)threadIdx.x) * 8)) + 12288));
    *(uint4*)(Xs_local_cast_7 + 0) = *(uint4*)(((bfloat16_t*)Xs) + (((((int)threadIdx.x) & 7) * 8) + 192));
    for (int vec_3 = 0; vec_3 < 2; ++vec_3) {
      float4 __10;
        float4 __11;
        uint2 v__6 = *(uint2*)(As_local_cast_6 + (vec_3 * 4));
        ((float2*)(&__11))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[0]);
        ((float2*)(&__11))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[1]);
        float4 __12;
        uint2 v__7 = *(uint2*)(Xs_local_cast_7 + (vec_3 * 4));
        ((float2*)(&__12))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__7))[0]);
        ((float2*)(&__12))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__7))[1]);
        __10.x = (__11.x*__12.x);
        __10.y = (__11.y*__12.y);
        __10.z = (__11.z*__12.z);
        __10.w = (__11.w*__12.w);
      *(float4*)(p + ((i_11 * 8) + (vec_3 * 4))) = __10;
    }
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 16; ++i_12) {
    acc_clear_3[i_12] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
      acc_clear_3[i_12] = (acc_clear_3[i_12] + p[((i_12 * 8) + rv_3)]);
    }
    acc_clear_3[i_12] = tl::AllReduce<tl::SumOp, 8, 1, 0, tl::NamedBarrier<32>>::run(acc_clear_3[i_12]);
    acc[i_12] = (acc[i_12] + acc_clear_3[i_12]);
  }
  if ((((int)threadIdx.x) % 8) == 0) {
    #pragma unroll
    for (int i_13 = 0; i_13 < 16; ++i_13) {
      if ((((((int)blockIdx.x) * 64) + (i_13 * 4)) + (((int)threadIdx.x) >> 3)) < 2037) {
        Y[(((((int)blockIdx.x) * 64) + (i_13 * 4)) + (((int)threadIdx.x) >> 3))] = acc[i_13];
      }
    }
  }
}

