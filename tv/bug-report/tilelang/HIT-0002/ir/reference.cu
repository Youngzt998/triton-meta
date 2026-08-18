#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ Beta, const float* __restrict__ G, const bfloat16_t* __restrict__ K, const bfloat16_t* __restrict__ V, bfloat16_t* __restrict__ dA, bfloat16_t* __restrict__ dbeta, float* __restrict__ dg, bfloat16_t* __restrict__ dk, const bfloat16_t* __restrict__ du, bfloat16_t* __restrict__ dv, const bfloat16_t* __restrict__ dw);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ Beta, const float* __restrict__ G, const bfloat16_t* __restrict__ K, const bfloat16_t* __restrict__ V, bfloat16_t* __restrict__ dA, bfloat16_t* __restrict__ dbeta, float* __restrict__ dg, bfloat16_t* __restrict__ dk, const bfloat16_t* __restrict__ du, bfloat16_t* __restrict__ dv, const bfloat16_t* __restrict__ dw) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* G_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  void* G_shared_exp = ((void*)((char*)buf_dyn_shmem + 8448));
  void* Beta_shared = ((void*)((char*)buf_dyn_shmem + 8704));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 8832));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 8832));
  void* V_shared_beta = ((void*)((char*)buf_dyn_shmem + 13312));
  void* K_shared_beta_g = ((void*)((char*)buf_dyn_shmem + 17408));
  void* du_shared = ((void*)((char*)buf_dyn_shmem + 17408));
  void* dw_shared = ((void*)((char*)buf_dyn_shmem + 25600));
  float dA_fragment[32];
  float dk_fragment[32];
  float dk_fragment_beta_g[32];
  float dv_fragment[16];
  float dv_fragment_beta[16];
  float dbeta_fragment_k[2];
  float dbeta_fragment_v[2];
  float dg_fragment[2];
  bfloat16_t K_shared_local_cast_1[8];
  bfloat16_t Beta_shared_local_cast_2[8];
  float G_shared_exp_local_cast_3[8];
  bfloat16_t K_shared_beta_g_local_cast[8];
  bfloat16_t Beta_shared_local_cast_4[2];
  float G_shared_exp_local_cast_5[2];
  float dbeta_fragment_reduce_tmpk[32];
  bfloat16_t K_shared_local_cast_6[2];
  float G_shared_exp_local_cast_7[2];
  float dbeta_fragment_k_clear[2];
  float dg_fragment_reduce_tmp[32];
  bfloat16_t K_shared_local_cast_8[2];
  float G_shared_exp_local_cast_9[2];
  bfloat16_t Beta_shared_local_cast_10[2];
  float dg_fragment_clear[2];
  bfloat16_t dk_local_cast_11[2];
  bfloat16_t Beta_shared_local_cast_12[2];
  float dbeta_fragment_reduce_tmpv[16];
  bfloat16_t V_shared_local_cast_13[2];
  float dbeta_fragment_v_clear[2];
  bfloat16_t dv_local_cast_14[2];
  bfloat16_t dA_local_cast_15[2];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dA_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dk_fragment + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dk_fragment_beta_g + (i_2 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dv_fragment + (i_3 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dv_fragment_beta + (i_4 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
  }
  float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(dbeta_fragment_k + 0) = make_float2(broadcast_var_5, broadcast_var_5);
  float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(dbeta_fragment_v + 0) = make_float2(broadcast_var_6, broadcast_var_6);
  float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(dg_fragment + 0) = make_float2(broadcast_var_7, broadcast_var_7);
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    *(uint4*)(((bfloat16_t*)A_shared) + (((((i_5 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(A + (((((((int)blockIdx.x) * 131072) + (i_5 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  if (((int)threadIdx.x) < 64) {
    ((bfloat16_t*)Beta_shared)[((int)threadIdx.x)] = Beta[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 32)) + ((int)blockIdx.y))];
    ((float*)G_shared)[((int)threadIdx.x)] = G[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 32)) + ((int)blockIdx.y))];
    ((float*)G_shared_exp)[((int)threadIdx.x)] = expf(G[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 32)) + ((int)blockIdx.y))]);
  }
  for (int i_k = 0; i_k < 2; ++i_k) {
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      *(uint4*)(((bfloat16_t*)K_shared) + ((i_6 * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(K + ((((((((int)blockIdx.x) * 262144) + (i_6 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 128)) + (i_k * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      *(uint4*)(K_shared_local_cast_1 + 0) = *(uint4*)(((bfloat16_t*)K_shared) + ((i_7 * 1024) + (((int)threadIdx.x) * 8)));
      *(uint4*)(Beta_shared_local_cast_2 + 0) = make_uint4(__pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)Beta_shared)[((i_7 * 16) + (((int)threadIdx.x) >> 3))]), __pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)Beta_shared)[((i_7 * 16) + (((int)threadIdx.x) >> 3))]), __pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)Beta_shared)[((i_7 * 16) + (((int)threadIdx.x) >> 3))]), __pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)Beta_shared)[((i_7 * 16) + (((int)threadIdx.x) >> 3))]));
      for (int vec_copy = 0; vec_copy < 2; ++vec_copy) {
        *(float4*)(G_shared_exp_local_cast_3 + (vec_copy * 4)) = make_float4(((float*)G_shared_exp)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((float*)G_shared_exp)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((float*)G_shared_exp)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((float*)G_shared_exp)[((i_7 * 16) + (((int)threadIdx.x) >> 3))]);
      }
      for (int vec = 0; vec < 2; ++vec) {
        uint2 __1;
        float4 __2;
          float4 __3;
          uint2 __4;
            uint2 v_ = *(uint2*)(K_shared_local_cast_1 + (vec * 4));
            uint2 v__1 = *(uint2*)(Beta_shared_local_cast_2 + (vec * 4));
            *(uint1*)(&(__4.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__1.x)))));
            *(uint1*)(&(__4.y)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.y))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__1.y)))));
          ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__4))[0]);
          ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__4))[1]);
          float4 v__2 = *(float4*)(G_shared_exp_local_cast_3 + (vec * 4));
          __2.x = (__3.x*v__2.x);
          __2.y = (__3.y*v__2.y);
          __2.z = (__3.z*v__2.z);
          __2.w = (__3.w*v__2.w);
        (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&__2))[1]);
        *(uint2*)(K_shared_beta_g_local_cast + (vec * 4)) = __1;
      }
      *(uint4*)(((bfloat16_t*)K_shared_beta_g) + (((((i_7 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K_shared_beta_g_local_cast + 0);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      *(uint4*)(((bfloat16_t*)dw_shared) + (((((i_8 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(dw + ((((((((int)blockIdx.x) * 262144) + (i_8 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 128)) + (i_k * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)dw_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((bfloat16_t*)K_shared_beta_g)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dA_fragment + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + ((ki * 32) >> 4)), uint64_t(desc_b + ((ki * 32) >> 4)), ((uint32_t*)(dA_fragment + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dA_fragment + 0), 32);
    }
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_1, (&(((bfloat16_t*)A_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((bfloat16_t*)dw_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk_fragment_beta_g + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 2048) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), ((uint32_t*)(dk_fragment_beta_g + 0)), ((0 < ki_1) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk_fragment_beta_g + 0), 32);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 16; ++i_9) {
      *(uint1*)(Beta_shared_local_cast_4 + 0) = make_uint1(__pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((bfloat16_t*)Beta_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
      *(float2*)(G_shared_exp_local_cast_5 + 0) = make_float2(((float*)G_shared_exp)[((((((int)threadIdx.x) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)G_shared_exp)[((((((int)threadIdx.x) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
      float2 __5;
        float2 __6;
          float2 v__3 = *(float2*)(dk_fragment_beta_g + (i_9 * 2));
          float2 __7;
          uint1 v__4 = *(uint1*)(Beta_shared_local_cast_4 + 0);
          ((float2*)(&__7))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
          __6.x = (v__3.x*__7.x);
          __6.y = (v__3.y*__7.y);
        float2 v__5 = *(float2*)(G_shared_exp_local_cast_5 + 0);
        __5.x = (__6.x*v__5.x);
        __5.y = (__6.y*v__5.y);
      *(float2*)(dk_fragment + (i_9 * 2)) = __5;
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      *(uint1*)(K_shared_local_cast_6 + 0) = *(uint1*)(((bfloat16_t*)K_shared) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_10 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      *(float2*)(G_shared_exp_local_cast_7 + 0) = make_float2(((float*)G_shared_exp)[((((((int)threadIdx.x) >> 5) * 16) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)G_shared_exp)[((((((int)threadIdx.x) >> 5) * 16) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
      float2 __8;
        float2 __9;
          float2 v__6 = *(float2*)(dk_fragment_beta_g + (i_10 * 2));
          float2 __10;
          uint1 v__7 = *(uint1*)(K_shared_local_cast_6 + 0);
          ((float2*)(&__10))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__7))[0]);
          __9.x = (v__6.x*__10.x);
          __9.y = (v__6.y*__10.y);
        float2 v__8 = *(float2*)(G_shared_exp_local_cast_7 + 0);
        __8.x = (__9.x*v__8.x);
        __8.y = (__9.y*v__8.y);
      *(float2*)(dbeta_fragment_reduce_tmpk + (i_10 * 2)) = __8;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      dbeta_fragment_k_clear[i_11] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        dbeta_fragment_k_clear[i_11] = (dbeta_fragment_k_clear[i_11] + dbeta_fragment_reduce_tmpk[((((rv & 7) * 4) + (i_11 * 2)) + (rv >> 3))]);
      }
      dbeta_fragment_k_clear[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(dbeta_fragment_k_clear[i_11]);
      dbeta_fragment_k[i_11] = (dbeta_fragment_k[i_11] + dbeta_fragment_k_clear[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      *(uint1*)(K_shared_local_cast_8 + 0) = *(uint1*)(((bfloat16_t*)K_shared) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_12 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_12 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      *(float2*)(G_shared_exp_local_cast_9 + 0) = make_float2(((float*)G_shared_exp)[((((((int)threadIdx.x) >> 5) * 16) + ((i_12 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)G_shared_exp)[((((((int)threadIdx.x) >> 5) * 16) + ((i_12 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
      *(uint1*)(Beta_shared_local_cast_10 + 0) = make_uint1(__pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((i_12 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((bfloat16_t*)Beta_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((i_12 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
      float2 __11;
        float2 __12;
          float2 __13;
            float2 v__9 = *(float2*)(dk_fragment_beta_g + (i_12 * 2));
            float2 __14;
            uint1 v__10 = *(uint1*)(K_shared_local_cast_8 + 0);
            ((float2*)(&__14))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__10))[0]);
            __13.x = (v__9.x*__14.x);
            __13.y = (v__9.y*__14.y);
          float2 v__11 = *(float2*)(G_shared_exp_local_cast_9 + 0);
          __12.x = (__13.x*v__11.x);
          __12.y = (__13.y*v__11.y);
        float2 __15;
        uint1 v__12 = *(uint1*)(Beta_shared_local_cast_10 + 0);
        ((float2*)(&__15))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__12))[0]);
        __11.x = (__12.x*__15.x);
        __11.y = (__12.y*__15.y);
      *(float2*)(dg_fragment_reduce_tmp + (i_12 * 2)) = __11;
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      dg_fragment_clear[i_13] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        dg_fragment_clear[i_13] = (dg_fragment_clear[i_13] + dg_fragment_reduce_tmp[((((rv_1 & 7) * 4) + (i_13 * 2)) + (rv_1 >> 3))]);
      }
      dg_fragment_clear[i_13] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(dg_fragment_clear[i_13]);
      dg_fragment[i_13] = (dg_fragment[i_13] + dg_fragment_clear[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 16; ++i_14) {
      uint1 __16;
      float2 v__13 = *(float2*)(dk_fragment + (i_14 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__16))[0] = __float22bfloat162_rn(((float2*)(&v__13))[0]);
      *(uint1*)(dk_local_cast_11 + 0) = __16;
      *(uint1*)(dk + ((((((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + ((i_14 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + (i_k * 64)) + ((i_14 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(dk_local_cast_11 + 0);
    }
  }
  for (int i_v = 0; i_v < 4; ++i_v) {
    __syncthreads();
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      *(uint4*)(((bfloat16_t*)V_shared) + ((i_15 * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(V + ((((((((int)blockIdx.x) * 262144) + (i_15 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + (i_v * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      uint4 __17;
        uint4 v__14 = *(uint4*)(((bfloat16_t*)V_shared) + ((i_16 * 1024) + (((int)threadIdx.x) * 8)));
        uint4 v__15 = make_uint4(__pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((i_16 * 32) + (((int)threadIdx.x) >> 2))], ((bfloat16_t*)Beta_shared)[((i_16 * 32) + (((int)threadIdx.x) >> 2))]), __pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((i_16 * 32) + (((int)threadIdx.x) >> 2))], ((bfloat16_t*)Beta_shared)[((i_16 * 32) + (((int)threadIdx.x) >> 2))]), __pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((i_16 * 32) + (((int)threadIdx.x) >> 2))], ((bfloat16_t*)Beta_shared)[((i_16 * 32) + (((int)threadIdx.x) >> 2))]), __pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((i_16 * 32) + (((int)threadIdx.x) >> 2))], ((bfloat16_t*)Beta_shared)[((i_16 * 32) + (((int)threadIdx.x) >> 2))]));
        *(uint1*)(&(__17.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__14.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__15.x)))));
        *(uint1*)(&(__17.y)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__14.y))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__15.y)))));
        *(uint1*)(&(__17.z)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__14.z))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__15.z)))));
        *(uint1*)(&(__17.w)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__14.w))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__15.w)))));
      *(uint4*)(((bfloat16_t*)V_shared_beta) + ((((i_16 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = __17;
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      *(uint4*)(((bfloat16_t*)du_shared) + ((((i_17 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(du + ((((((((int)blockIdx.x) * 262144) + (i_17 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + (i_v * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_2;
      __syncthreads();
      tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a_2, (&(((bfloat16_t*)du_shared)[0])));
      tl::initialize_wgmma_descriptor<2, 1, 32>(desc_b_2, (&(((bfloat16_t*)V_shared_beta)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dA_fragment + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((ki_2 * 32) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 32) >> 4)), ((uint32_t*)(dA_fragment + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dA_fragment + 0), 32);
    }
    {
      tl::GmmaDescriptor desc_a_3;
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_3, (&(((bfloat16_t*)A_shared)[0])));
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_3, (&(((bfloat16_t*)du_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv_fragment_beta + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 32, 16, true, true, 1, 1>(uint64_t(desc_a_3 + ((ki_3 * 2048) >> 4)), uint64_t(desc_b_3 + ((ki_3 * 1024) >> 4)), ((uint32_t*)(dv_fragment_beta + 0)), ((0 < ki_3) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv_fragment_beta + 0), 16);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 8; ++i_18) {
      *(uint1*)(Beta_shared_local_cast_12 + 0) = make_uint1(__pack_nv_bfloat162(((bfloat16_t*)Beta_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((bfloat16_t*)Beta_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
      float2 __18;
        float2 v__16 = *(float2*)(dv_fragment_beta + (i_18 * 2));
        float2 __19;
        uint1 v__17 = *(uint1*)(Beta_shared_local_cast_12 + 0);
        ((float2*)(&__19))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__17))[0]);
        __18.x = (v__16.x*__19.x);
        __18.y = (v__16.y*__19.y);
      *(float2*)(dv_fragment + (i_18 * 2)) = __18;
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 8; ++i_19) {
      *(uint1*)(V_shared_local_cast_13 + 0) = *(uint1*)(((bfloat16_t*)V_shared) + ((((((((int)threadIdx.x) >> 5) * 512) + ((i_19 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((i_19 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __20;
        float2 v__18 = *(float2*)(dv_fragment_beta + (i_19 * 2));
        float2 __21;
        uint1 v__19 = *(uint1*)(V_shared_local_cast_13 + 0);
        ((float2*)(&__21))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__19))[0]);
        __20.x = (v__18.x*__21.x);
        __20.y = (v__18.y*__21.y);
      *(float2*)(dbeta_fragment_reduce_tmpv + (i_19 * 2)) = __20;
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      dbeta_fragment_v_clear[i_20] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
        dbeta_fragment_v_clear[i_20] = (dbeta_fragment_v_clear[i_20] + dbeta_fragment_reduce_tmpv[((((rv_2 & 3) * 4) + (i_20 * 2)) + (rv_2 >> 2))]);
      }
      dbeta_fragment_v_clear[i_20] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(dbeta_fragment_v_clear[i_20]);
      dbeta_fragment_v[i_20] = (dbeta_fragment_v[i_20] + dbeta_fragment_v_clear[i_20]);
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 8; ++i_21) {
      uint1 __22;
      float2 v__20 = *(float2*)(dv_fragment + (i_21 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__22))[0] = __float22bfloat162_rn(((float2*)(&v__20))[0]);
      *(uint1*)(dv_local_cast_14 + 0) = __22;
      *(uint1*)(dv + ((((((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + ((i_21 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + (i_v * 32)) + ((i_21 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(dv_local_cast_14 + 0);
    }
  }
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_22 = 0; i_22 < 2; ++i_22) {
      dbeta[(((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + (i_22 * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((int)blockIdx.y))] = ((bfloat16_t)(dbeta_fragment_k[i_22] + dbeta_fragment_v[i_22]));
      dg[(((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + (i_22 * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((int)blockIdx.y))] = dg_fragment[i_22];
    }
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 16; ++i_23) {
    uint1 __23;
    float2 v__21 = *(float2*)(dA_fragment + (i_23 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__23))[0] = __float22bfloat162_rn(((float2*)(&v__21))[0]);
    *(uint1*)(dA_local_cast_15 + 0) = __23;
    *(uint1*)(dA + (((((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((i_23 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.y) * 64)) + ((i_23 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(dA_local_cast_15 + 0);
  }
}

