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
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ Sinks, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ Sinks, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  float acc_o[64];
  float scores_max[2];
  float logsum[2];
  half_t sinks[2];
  float acc_o_1[64];
  float logsum_1[2];
  float scores_max_1[2];
  half_t sinks_1[2];
  float scores_max_2[2];
  float scores_max_3[2];
  float scores_max_4[2];
  float scores_max_5[2];
  float scores_max_6[2];
  float scores_max_7[2];
  float logsum_2[2];
  float acc_o_2[64];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(uint4*)(((half_t*)Q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((int)blockIdx.x) * 16384) + (i * 2048)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o_2 + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(logsum_2 + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(scores_max_7 + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  *(uint1*)(sinks_1 + 0) = make_uint1(__pack_half2(Sinks[0], Sinks[0]));
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((i_2 * 2048) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)V_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((i_3 * 2048) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  if (0 < ((int)blockIdx.x)) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      tl::cp_async_gs<16>((&(((half_t*)K_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(K[(((i_4 * 2048) + (((int)threadIdx.x) * 8)) + 16384)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)V_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(V[(((i_5 * 2048) + (((int)threadIdx.x) * 8)) + 16384)])));
    }
    tl::cp_async_commit();
  }
  float acc_s[64];
  half_t acc_s_cast[64];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  float acc_s_1[64];
  float scores_max_prev_1[2];
  float acc_s_2[64];
  float scores_max_prev_2[2];
  float scores_scale_1[2];
  float scores_max_prev_3[2];
  float acc_s_3[64];
  float scores_sum_1[2];
  float acc_s_4[64];
  float scores_scale_2[2];
  float scores_sum_2[2];
  half_t acc_s_cast_1[64];
  float acc_s_5[64];
  float scores_scale_3[2];
  if (((int)blockIdx.x) == 1) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 64; ++i_6) {
      acc_s_1[i_6] = 0x0p+0f/*0.000000e+00*/;
    }
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  if (((int)blockIdx.x) == 1) {
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b, (((((int)blockIdx.x) + 1) & 1) * 32768));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_5 + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 16384) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_5 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_5 + 0), 128);
    }
    *(float2*)(scores_max_prev_1 + 0) = *(float2*)(scores_max_7 + 0);
    float broadcast_var_3 = -CUDART_INF_F;
    *(float2*)(scores_max_7 + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    float scores_max_clear[2];
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      scores_max_clear[i_7] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 32; ++rv) {
        scores_max_clear[i_7] = max(scores_max_clear[i_7], acc_s_2[((((rv & 15) * 4) + (i_7 * 2)) + (rv >> 4))]);
      }
      scores_max_clear[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_7]);
      scores_max_7[i_7] = max(scores_max_7[i_7], scores_max_clear[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      scores_max_7[i_8] = max(scores_max_7[i_8], scores_max_prev_2[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      scores_scale_1[i_9] = exp2f(((scores_max_prev_3[i_9] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max_7[i_9] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      acc_s_3[i_10] = exp2f(((acc_s_3[i_10] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max_7[((i_10 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      scores_sum_1[i_11] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
        scores_sum_1[i_11] = (scores_sum_1[i_11] + acc_s_4[((((rv_1 & 15) * 4) + (i_11 * 2)) + (rv_1 >> 4))]);
      }
      scores_sum_1[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum_1[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      logsum_2[i_12] = ((logsum_2[i_12] * scores_scale_2[i_12]) + scores_sum_2[i_12]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 16; ++i_13) {
      uint2 __1;
      float4 v_ = *(float4*)(acc_s_5 + (i_13 * 4));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(acc_s_cast_1 + (i_13 * 4)) = __1;
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 64; ++i_14) {
      acc_o_2[i_14] = (acc_o_2[i_14] * scores_scale_3[((i_14 & 3) >> 1)]);
    }
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  if (((int)blockIdx.x) == 1) {
    {
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_1, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_1, (((((int)blockIdx.x) + 1) & 1) * 32768));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_1 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_2 + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_1 + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_2 + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_2 + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_1 + 0), 32);
    }
  }
  float acc_s_6[64];
  half_t acc_s_cast_2[64];
  float scores_max_prev_4[2];
  float scores_scale_4[2];
  float scores_sum_3[2];
  float acc_s_7[64];
  float scores_max_prev_5[2];
  float acc_s_8[64];
  float scores_max_prev_6[2];
  float scores_scale_5[2];
  float scores_max_prev_7[2];
  float acc_s_9[64];
  float scores_sum_4[2];
  float acc_s_10[64];
  float scores_scale_6[2];
  float scores_sum_5[2];
  half_t acc_s_cast_3[64];
  float acc_s_11[64];
  float scores_scale_7[2];
  #pragma unroll
  for (int i_15 = 0; i_15 < 64; ++i_15) {
    float condval;
    if ((((((((int)blockIdx.x) * 128) + ((i_15 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_15 & 1)) <= ((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_15 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
      condval = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval = -CUDART_INF_F;
    }
    acc_s_7[i_15] = condval;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_1;
    tl::GmmaDescriptor desc_b_2;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)K_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_b_2, (((int)blockIdx.x) * 32768));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_11 + 0), 128);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a_1 + (((((ki_2 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((((ki_2 >> 2) * 16384) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_11 + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_11 + 0), 128);
  }
  *(float2*)(scores_max_prev_5 + 0) = *(float2*)(scores_max_7 + 0);
  float broadcast_var_4 = -CUDART_INF_F;
  *(float2*)(scores_max_7 + 0) = make_float2(broadcast_var_4, broadcast_var_4);
  float scores_max_clear_1[2];
  #pragma unroll
  for (int i_16 = 0; i_16 < 2; ++i_16) {
    scores_max_clear_1[i_16] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 32; ++rv_2) {
      scores_max_clear_1[i_16] = max(scores_max_clear_1[i_16], acc_s_8[((((rv_2 & 15) * 4) + (i_16 * 2)) + (rv_2 >> 4))]);
    }
    scores_max_clear_1[i_16] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_16]);
    scores_max_7[i_16] = max(scores_max_7[i_16], scores_max_clear_1[i_16]);
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 2; ++i_17) {
    scores_max_7[i_17] = max(scores_max_7[i_17], scores_max_prev_6[i_17]);
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    scores_scale_5[i_18] = exp2f(((scores_max_prev_7[i_18] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max_7[i_18] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 64; ++i_19) {
    acc_s_9[i_19] = exp2f(((acc_s_9[i_19] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max_7[((i_19 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 2; ++i_20) {
    scores_sum_4[i_20] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 32; ++rv_3) {
      scores_sum_4[i_20] = (scores_sum_4[i_20] + acc_s_10[((((rv_3 & 15) * 4) + (i_20 * 2)) + (rv_3 >> 4))]);
    }
    scores_sum_4[i_20] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum_4[i_20]);
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    logsum_2[i_21] = ((logsum_2[i_21] * scores_scale_6[i_21]) + scores_sum_5[i_21]);
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 16; ++i_22) {
    uint2 __2;
    float4 v__1 = *(float4*)(acc_s_11 + (i_22 * 4));
    ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
    ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
    *(uint2*)(acc_s_cast_3 + (i_22 * 4)) = __2;
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 64; ++i_23) {
    acc_o_2[i_23] = (acc_o_2[i_23] * scores_scale_7[((i_23 & 3) >> 1)]);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_b_3;
    tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_b_3, (((int)blockIdx.x) * 32768));
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_3 + 0), 32);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_2 + 0), 128);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_3 + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_2 + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_2 + 0), 128);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_3 + 0), 32);
  }
  float broadcast_var_5 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
  float broadcast_var_6 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float2 __3;
    float2 v__2 = *(float2*)(logsum_2 + 0);
    float2 __4;
    float2 __5;
      float2 __6;
        float2 __7;
        uint1 v__3 = *(uint1*)(sinks_1 + 0);
        ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__3))[0]);
        float2 v__4 = make_float2(broadcast_var_5, broadcast_var_5);
        __6.x = (__7.x*v__4.x);
        __6.y = (__7.y*v__4.y);
      float2 __8;
        float2 v__5 = *(float2*)(scores_max_7 + 0);
        float2 v__6 = make_float2(broadcast_var_6, broadcast_var_6);
        __8.x = (v__5.x*v__6.x);
        __8.y = (v__5.y*v__6.y);
      __5.x = (__6.x-__8.x);
      __5.y = (__6.y-__8.y);
    __4.x = exp2f(__5.x);
    __4.y = exp2f(__5.y);
    __3.x = (v__2.x+__4.x);
    __3.y = (v__2.y+__4.y);
  *(float2*)(logsum_2 + 0) = __3;
  #pragma unroll
  for (int i_24 = 0; i_24 < 64; ++i_24) {
    acc_o_2[i_24] = (acc_o_2[i_24] / logsum_2[((i_24 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_25 = 0; i_25 < 8; ++i_25) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_25 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o_2[(i_25 * 8)]), ((half_t)acc_o_2[((i_25 * 8) + 1)])), __pack_half2(((half_t)acc_o_2[((i_25 * 8) + 2)]), ((half_t)acc_o_2[((i_25 * 8) + 3)])), __pack_half2(((half_t)acc_o_2[((i_25 * 8) + 4)]), ((half_t)acc_o_2[((i_25 * 8) + 5)])), __pack_half2(((half_t)acc_o_2[((i_25 * 8) + 6)]), ((half_t)acc_o_2[((i_25 * 8) + 7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(Output[(((int)blockIdx.x) * 16384)])), (&(((half_t*)O_shared)[0])), 32768);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

