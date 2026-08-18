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

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ KV, __grid_constant__ const CUtensorMap Output_desc, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ Sinks, const int* __restrict__ TopkIndices);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const bfloat16_t* __restrict__ KV, __grid_constant__ const CUtensorMap Output_desc, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ Sinks, const int* __restrict__ TopkIndices) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* acc_o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Sinks_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 17408));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 50176));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 58368));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 58368));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 58368));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 58368));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 58368));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 59392));
  float acc_o[32];
  float logsum[2];
  float scores_max[2];
  signed char mask[8];
  float acc_s[16];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  float scores_max_clear[2];
  float scores_max_clear_1[2];
  float scores_max_clear_2[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Output_desc);
  }
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((((i * 2) + (((int)threadIdx.x) >> 7)) < 1)) {
      condval = *(uint4*)(Q + (((i * 2048) + (((int)blockIdx.y) * 1024)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(((bfloat16_t*)Q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  if (((int)threadIdx.x) < 64) {
    bfloat16_t condval_1;
    if ((((int)threadIdx.x) < 8)) {
      condval_1 = Sinks[((int)threadIdx.x)];
    } else {
      condval_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    }
    ((bfloat16_t*)Sinks_shared)[((int)threadIdx.x)] = condval_1;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(logsum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -0x1p+30f/*-1.073742e+09*/;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    int idx = TopkIndices[(((((int)blockIdx.y) * 256) + (i_2 * 16)) + (((int)threadIdx.x) >> 4))];
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)KV_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(KV[((((int64_t)idx) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])), ((idx < 1024) && (0 <= idx)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    int idx_1 = TopkIndices[((((((int)blockIdx.y) * 256) + (i_3 * 16)) + (((int)threadIdx.x) >> 4)) + 64)];
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)KV_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(KV[((((int64_t)idx_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])), ((idx_1 < 1024) && (0 <= idx_1)));
  }
  tl::cp_async_commit();
  for (int i_i = 0; i_i < 2; ++i_i) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      int2 idx_2 = *(int2*)(TopkIndices + (((((((int)blockIdx.y) * 256) + (i_i * 64)) + ((((int)threadIdx.x) >> 7) * 32)) + (i_4 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      int broadcast_var_4 = 0;
      char2 __1;
      ushort2 __2;
        int2 v_ = make_int2(broadcast_var_4, broadcast_var_4);
        __2.x = (v_.x<=idx_2.x);
        __2.y = (v_.y<=idx_2.y);
      __1.x=((signed char)(__2.x));
      __1.y=((signed char)(__2.y));
      *(char2*)(mask + (i_4 * 2)) = __1;
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      float condval_2;
      if (((bool)mask[(((i_5 >> 2) * 2) + (i_5 & 1))])) {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_2 = -CUDART_INF_F;
      }
      acc_s[i_5] = condval_2;
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((bfloat16_t*)KV_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b, (i_i * 16384));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + (((((ki >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_max_clear[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max_clear[i_6] = max(scores_max_clear[i_6], acc_s[((((rv & 3) * 4) + (i_6 * 2)) + (rv >> 2))]);
      }
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6], (&(((float*)workspace_4)[0])));
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6]);
      scores_max[i_6] = max(scores_max[i_6], scores_max_clear[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      scores_max[i_7] = max(scores_max[i_7], scores_max_prev[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      scores_scale[i_8] = exp2f(((scores_max_prev[i_8] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_8] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 16; ++i_9) {
      acc_s[i_9] = exp2f(((acc_s[i_9] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_9 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      scores_sum[i_10] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_10] = (scores_sum[i_10] + acc_s[((((rv_1 & 3) * 4) + (i_10 * 2)) + (rv_1 >> 2))]);
      }
      scores_sum[i_10] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_10], (&(((float*)workspace_5)[0])));
      scores_sum[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      acc_o[i_11] = (acc_o[i_11] * scores_scale[((i_11 & 3) >> 1)]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)S_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_12) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc_s[(i_12 * 8)]), ((bfloat16_t)acc_s[((i_12 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_s[((i_12 * 8) + 2)]), ((bfloat16_t)acc_s[((i_12 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_s[((i_12 * 8) + 4)]), ((bfloat16_t)acc_s[((i_12 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_s[((i_12 * 8) + 6)]), ((bfloat16_t)acc_s[((i_12 * 8) + 7)])));
    }
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((bfloat16_t*)S_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((bfloat16_t*)KV_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_1, (i_i * 16384));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 32) >> 4)), uint64_t(desc_b_1 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_1 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 32);
    }
    __syncthreads();
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      int idx_3 = TopkIndices[(((((((int)blockIdx.y) * 256) + (i_i * 64)) + (i_13 * 16)) + (((int)threadIdx.x) >> 4)) + 128)];
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)KV_shared)[(((((((i_i * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_13 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(KV[((((int64_t)idx_3) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])), ((idx_3 < 1024) && (0 <= idx_3)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      logsum[i_14] = ((logsum[i_14] * scores_scale[i_14]) + scores_sum[i_14]);
    }
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 4; ++i_15) {
    int2 idx_4 = *(int2*)(TopkIndices + (((((((int)blockIdx.y) * 256) + ((((int)threadIdx.x) >> 7) * 32)) + (i_15 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 128));
    int broadcast_var_5 = 0;
    char2 __3;
    ushort2 __4;
      int2 v__1 = make_int2(broadcast_var_5, broadcast_var_5);
      __4.x = (v__1.x<=idx_4.x);
      __4.y = (v__1.y<=idx_4.y);
    __3.x=((signed char)(__4.x));
    __3.y=((signed char)(__4.y));
    *(char2*)(mask + (i_15 * 2)) = __3;
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 16; ++i_16) {
    float condval_3;
    if (((bool)mask[(((i_16 >> 2) * 2) + (i_16 & 1))])) {
      condval_3 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_3 = -CUDART_INF_F;
    }
    acc_s[i_16] = condval_3;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_2;
    tl::GmmaDescriptor desc_b_2;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((bfloat16_t*)Q_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((bfloat16_t*)KV_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + (((((ki_2 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
  }
  *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
  __syncthreads();
  #pragma unroll
  for (int i_17 = 0; i_17 < 2; ++i_17) {
    scores_max_clear_1[i_17] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
      scores_max_clear_1[i_17] = max(scores_max_clear_1[i_17], acc_s[((((rv_2 & 3) * 4) + (i_17 * 2)) + (rv_2 >> 2))]);
    }
    scores_max_clear_1[i_17] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_17], (&(((float*)workspace_3)[0])));
    scores_max_clear_1[i_17] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_17]);
    scores_max[i_17] = max(scores_max[i_17], scores_max_clear_1[i_17]);
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    scores_max[i_18] = max(scores_max[i_18], scores_max_prev[i_18]);
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 2; ++i_19) {
    scores_scale[i_19] = exp2f(((scores_max_prev[i_19] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_19] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 16; ++i_20) {
    acc_s[i_20] = exp2f(((acc_s[i_20] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_20 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    scores_sum[i_21] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
      scores_sum[i_21] = (scores_sum[i_21] + acc_s[((((rv_3 & 3) * 4) + (i_21 * 2)) + (rv_3 >> 2))]);
    }
    scores_sum[i_21] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_21], (&(((float*)workspace_2)[0])));
    scores_sum[i_21] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_21]);
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 32; ++i_22) {
    acc_o[i_22] = (acc_o[i_22] * scores_scale[((i_22 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_23 = 0; i_23 < 2; ++i_23) {
    tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)S_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_23) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc_s[(i_23 * 8)]), ((bfloat16_t)acc_s[((i_23 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_s[((i_23 * 8) + 2)]), ((bfloat16_t)acc_s[((i_23 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_s[((i_23 * 8) + 4)]), ((bfloat16_t)acc_s[((i_23 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_s[((i_23 * 8) + 6)]), ((bfloat16_t)acc_s[((i_23 * 8) + 7)])));
  }
  {
    tl::GmmaDescriptor desc_a_3;
    tl::GmmaDescriptor desc_b_3;
    __syncthreads();
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((bfloat16_t*)S_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_3, (&(((bfloat16_t*)KV_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_3 + ((ki_3 * 32) >> 4)), uint64_t(desc_b_3 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_3 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 32);
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 2; ++i_24) {
    logsum[i_24] = ((logsum[i_24] * scores_scale[i_24]) + scores_sum[i_24]);
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 4; ++i_25) {
    int2 idx_5 = *(int2*)(TopkIndices + (((((((int)blockIdx.y) * 256) + ((((int)threadIdx.x) >> 7) * 32)) + (i_25 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 192));
    int broadcast_var_6 = 0;
    char2 __5;
    ushort2 __6;
      int2 v__2 = make_int2(broadcast_var_6, broadcast_var_6);
      __6.x = (v__2.x<=idx_5.x);
      __6.y = (v__2.y<=idx_5.y);
    __5.x=((signed char)(__6.x));
    __5.y=((signed char)(__6.y));
    *(char2*)(mask + (i_25 * 2)) = __5;
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 16; ++i_26) {
    float condval_4;
    if (((bool)mask[(((i_26 >> 2) * 2) + (i_26 & 1))])) {
      condval_4 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_4 = -CUDART_INF_F;
    }
    acc_s[i_26] = condval_4;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_4;
    tl::GmmaDescriptor desc_b_4;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((bfloat16_t*)Q_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_4, (&(((bfloat16_t*)KV_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_b_4, 16384);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
      tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_4 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), uint64_t(desc_b_4 + (((((ki_4 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_4 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
  }
  *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
  __syncthreads();
  #pragma unroll
  for (int i_27 = 0; i_27 < 2; ++i_27) {
    scores_max_clear_2[i_27] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_4 = 0; rv_4 < 8; ++rv_4) {
      scores_max_clear_2[i_27] = max(scores_max_clear_2[i_27], acc_s[((((rv_4 & 3) * 4) + (i_27 * 2)) + (rv_4 >> 2))]);
    }
    scores_max_clear_2[i_27] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_27], (&(((float*)workspace_1)[0])));
    scores_max_clear_2[i_27] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_27]);
    scores_max[i_27] = max(scores_max[i_27], scores_max_clear_2[i_27]);
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 2; ++i_28) {
    scores_max[i_28] = max(scores_max[i_28], scores_max_prev[i_28]);
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 2; ++i_29) {
    scores_scale[i_29] = exp2f(((scores_max_prev[i_29] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_29] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 16; ++i_30) {
    acc_s[i_30] = exp2f(((acc_s[i_30] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_30 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_31 = 0; i_31 < 2; ++i_31) {
    scores_sum[i_31] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_5 = 0; rv_5 < 8; ++rv_5) {
      scores_sum[i_31] = (scores_sum[i_31] + acc_s[((((rv_5 & 3) * 4) + (i_31 * 2)) + (rv_5 >> 2))]);
    }
    scores_sum[i_31] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_31], (&(((float*)workspace)[0])));
    scores_sum[i_31] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_31]);
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 32; ++i_32) {
    acc_o[i_32] = (acc_o[i_32] * scores_scale[((i_32 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_33 = 0; i_33 < 2; ++i_33) {
    tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)S_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_33) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc_s[(i_33 * 8)]), ((bfloat16_t)acc_s[((i_33 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_s[((i_33 * 8) + 2)]), ((bfloat16_t)acc_s[((i_33 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_s[((i_33 * 8) + 4)]), ((bfloat16_t)acc_s[((i_33 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_s[((i_33 * 8) + 6)]), ((bfloat16_t)acc_s[((i_33 * 8) + 7)])));
  }
  {
    tl::GmmaDescriptor desc_a_5;
    tl::GmmaDescriptor desc_b_5;
    __syncthreads();
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((bfloat16_t*)S_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_5, (&(((bfloat16_t*)KV_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_b_5, 16384);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
      tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_5 + ((ki_5 * 32) >> 4)), uint64_t(desc_b_5 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_5 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 32);
  }
  #pragma unroll
  for (int i_34 = 0; i_34 < 2; ++i_34) {
    logsum[i_34] = ((logsum[i_34] * scores_scale[i_34]) + scores_sum[i_34]);
  }
  #pragma unroll
  for (int i_35 = 0; i_35 < 2; ++i_35) {
    logsum[i_35] = (logsum[i_35] + exp2f(((((float)((bfloat16_t*)Sinks_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + (i_35 * 8)) + ((((int)threadIdx.x) & 31) >> 2))]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/) - (scores_max[i_35] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/))));
  }
  #pragma unroll
  for (int i_36 = 0; i_36 < 32; ++i_36) {
    acc_o[i_36] = (acc_o[i_36] / logsum[((i_36 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_37 = 0; i_37 < 4; ++i_37) {
    tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)acc_o_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_37 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_37 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc_o[(i_37 * 8)]), ((bfloat16_t)acc_o[((i_37 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_o[((i_37 * 8) + 2)]), ((bfloat16_t)acc_o[((i_37 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_o[((i_37 * 8) + 4)]), ((bfloat16_t)acc_o[((i_37 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_o[((i_37 * 8) + 6)]), ((bfloat16_t)acc_o[((i_37 * 8) + 7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    tl::tma_store(Output_desc, (&(((bfloat16_t*)acc_o_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
    tl::tma_store(Output_desc, (&(((bfloat16_t*)acc_o_shared)[4096])), 64, 0, ((int)blockIdx.y), 0);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

