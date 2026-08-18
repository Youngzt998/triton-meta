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
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 33792));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 41984));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 43008));
  float acc_o[32];
  float logsum[2];
  float scores_max[2];
  signed char mask[8];
  float acc_s[16];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_scale[2];
  float scores_sum[2];
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
  for (int i_i = 0; i_i < 4; ++i_i) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      int2 idx = *(int2*)(TopkIndices + (((((((int)blockIdx.y) * 256) + (i_i * 64)) + ((((int)threadIdx.x) >> 7) * 32)) + (i_2 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      int broadcast_var_4 = 0;
      char2 __1;
      ushort2 __2;
        int2 v_ = make_int2(broadcast_var_4, broadcast_var_4);
        __2.x = (v_.x<=idx.x);
        __2.y = (v_.y<=idx.y);
      __1.x=((signed char)(__2.x));
      __1.y=((signed char)(__2.y));
      *(char2*)(mask + (i_2 * 2)) = __1;
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      int idx_1 = TopkIndices[((((((int)blockIdx.y) * 256) + (i_i * 64)) + (i_3 * 16)) + (((int)threadIdx.x) >> 4))];
      bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_2;
      if (((0 <= idx_1) && (idx_1 < 1024))) {
        condval_2 = *(uint4*)(KV + ((((int64_t)idx_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
      }
      *(uint4*)(((bfloat16_t*)KV_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      float condval_3;
      if (((bool)mask[(((i_4 >> 2) * 2) + (i_4 & 1))])) {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_3 = -CUDART_INF_F;
      }
      acc_s[i_4] = condval_3;
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((bfloat16_t*)KV_shared)[0])));
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
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      scores_max_clear[i_5] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max_clear[i_5] = max(scores_max_clear[i_5], acc_s[((((rv & 3) * 4) + (i_5 * 2)) + (rv >> 2))]);
      }
      scores_max_clear[i_5] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_5], (&(((float*)workspace_1)[0])));
      scores_max_clear[i_5] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_5]);
      scores_max[i_5] = max(scores_max[i_5], scores_max_clear[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_max[i_6] = max(scores_max[i_6], scores_max_prev[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      scores_scale[i_7] = exp2f(((scores_max_prev[i_7] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_7] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      acc_s[i_8] = exp2f(((acc_s[i_8] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_8 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      scores_sum[i_9] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_9] = (scores_sum[i_9] + acc_s[((((rv_1 & 3) * 4) + (i_9 * 2)) + (rv_1 >> 2))]);
      }
      scores_sum[i_9] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_9], (&(((float*)workspace)[0])));
      scores_sum[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 32; ++i_10) {
      acc_o[i_10] = (acc_o[i_10] * scores_scale[((i_10 & 3) >> 1)]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)S_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_11) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc_s[(i_11 * 8)]), ((bfloat16_t)acc_s[((i_11 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_s[((i_11 * 8) + 2)]), ((bfloat16_t)acc_s[((i_11 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_s[((i_11 * 8) + 4)]), ((bfloat16_t)acc_s[((i_11 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_s[((i_11 * 8) + 6)]), ((bfloat16_t)acc_s[((i_11 * 8) + 7)])));
    }
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((bfloat16_t*)S_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((bfloat16_t*)KV_shared)[0])));
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
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      logsum[i_12] = ((logsum[i_12] * scores_scale[i_12]) + scores_sum[i_12]);
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 2; ++i_13) {
    logsum[i_13] = (logsum[i_13] + exp2f(((((float)((bfloat16_t*)Sinks_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + (i_13 * 8)) + ((((int)threadIdx.x) & 31) >> 2))]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/) - (scores_max[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/))));
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 32; ++i_14) {
    acc_o[i_14] = (acc_o[i_14] / logsum[((i_14 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_15 = 0; i_15 < 4; ++i_15) {
    tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)acc_o_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_15 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_15 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc_o[(i_15 * 8)]), ((bfloat16_t)acc_o[((i_15 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_o[((i_15 * 8) + 2)]), ((bfloat16_t)acc_o[((i_15 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_o[((i_15 * 8) + 4)]), ((bfloat16_t)acc_o[((i_15 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_o[((i_15 * 8) + 6)]), ((bfloat16_t)acc_o[((i_15 * 8) + 7)])));
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

