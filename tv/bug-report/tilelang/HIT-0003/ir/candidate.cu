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

extern "C" __global__ void main_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Os = ((void*)((char*)buf_dyn_shmem + 0));
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 8192));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 16384));
  float o[32];
  float ls[2];
  float mx[2];
  float s[32];
  float mp[2];
  float ss[2];
  half_t sc[32];
  float sm[2];
  float mx_clear[2];
  float mx_clear_1[2];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(((half_t*)Qs) + (((((i * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((int)blockIdx.x) * 4096) + (i * 1024)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(ls + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(mx + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)Ks)[(((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((i_2 * 1024) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)Vs)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((i_3 * 1024) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < (((int)blockIdx.x) + 2); ++k) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 32; ++i_4) {
      float condval;
      if ((((((k * 64) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) <= (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 128))) {
        condval = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval = -CUDART_INF_F;
      }
      s[i_4] = condval;
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Qs)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)Ks)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + ((ki * 32) >> 4)), uint64_t(desc_b + ((ki * 32) >> 4)), ((uint32_t*)(s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)Ks)[(((((i_5 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((k * 4096) + (i_5 * 1024)) + (((int)threadIdx.x) * 8)) + 4096)])));
    }
    tl::cp_async_commit();
    *(float2*)(mp + 0) = *(float2*)(mx + 0);
    float broadcast_var_3 = -CUDART_INF_F;
    *(float2*)(mx + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      mx_clear[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        mx_clear[i_6] = max(mx_clear[i_6], s[((((rv & 7) * 4) + (i_6 * 2)) + (rv >> 3))]);
      }
      mx_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(mx_clear[i_6]);
      mx[i_6] = max(mx[i_6], mx_clear[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      ss[i_7] = exp2f(((mp[i_7] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[i_7] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 32; ++i_8) {
      o[i_8] = (o[i_8] * ss[((i_8 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      s[i_9] = exp2f(((s[i_9] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[((i_9 & 3) >> 1)] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      uint2 __1;
      float4 v_ = *(float4*)(s + (i_10 * 4));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(sc + (i_10 * 4)) = __1;
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((half_t*)Vs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(sc + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(sc + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(sc + 0), 16);
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      tl::cp_async_gs<16>((&(((half_t*)Vs)[(((((i_11 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((k * 4096) + (i_11 * 1024)) + (((int)threadIdx.x) * 8)) + 4096)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      sm[i_12] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        sm[i_12] = (sm[i_12] + s[((((rv_1 & 7) * 4) + (i_12 * 2)) + (rv_1 >> 3))]);
      }
      sm[i_12] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(sm[i_12]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      ls[i_13] = ((ls[i_13] * ss[i_13]) + sm[i_13]);
    }
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 32; ++i_14) {
    float condval_1;
    if ((((((i_14 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_14 & 1)) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_14 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_1 = -CUDART_INF_F;
    }
    s[i_14] = condval_1;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_1;
    tl::GmmaDescriptor desc_b_2;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Qs)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)Ks)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((ki_2 * 32) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 32) >> 4)), ((uint32_t*)(s + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
  }
  *(float2*)(mp + 0) = *(float2*)(mx + 0);
  float broadcast_var_4 = -CUDART_INF_F;
  *(float2*)(mx + 0) = make_float2(broadcast_var_4, broadcast_var_4);
  #pragma unroll
  for (int i_15 = 0; i_15 < 2; ++i_15) {
    mx_clear_1[i_15] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      mx_clear_1[i_15] = max(mx_clear_1[i_15], s[((((rv_2 & 7) * 4) + (i_15 * 2)) + (rv_2 >> 3))]);
    }
    mx_clear_1[i_15] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(mx_clear_1[i_15]);
    mx[i_15] = max(mx[i_15], mx_clear_1[i_15]);
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 2; ++i_16) {
    ss[i_16] = exp2f(((mp[i_16] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[i_16] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 32; ++i_17) {
    o[i_17] = (o[i_17] * ss[((i_17 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 32; ++i_18) {
    s[i_18] = exp2f(((s[i_18] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[((i_18 & 3) >> 1)] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 8; ++i_19) {
    uint2 __2;
    float4 v__1 = *(float4*)(s + (i_19 * 4));
    ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
    ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
    *(uint2*)(sc + (i_19 * 4)) = __2;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_b_3;
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_3, (&(((half_t*)Vs)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(sc + 0), 16);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(sc + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(o + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(sc + 0), 16);
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 2; ++i_20) {
    sm[i_20] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      sm[i_20] = (sm[i_20] + s[((((rv_3 & 7) * 4) + (i_20 * 2)) + (rv_3 >> 3))]);
    }
    sm[i_20] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(sm[i_20]);
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    ls[i_21] = ((ls[i_21] * ss[i_21]) + sm[i_21]);
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 32; ++i_22) {
    o[i_22] = (o[i_22] / ls[((i_22 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_23 = 0; i_23 < 4; ++i_23) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)Os)[(((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (i_23 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)o[(i_23 * 8)]), ((half_t)o[((i_23 * 8) + 1)])), __pack_half2(((half_t)o[((i_23 * 8) + 2)]), ((half_t)o[((i_23 * 8) + 3)])), __pack_half2(((half_t)o[((i_23 * 8) + 4)]), ((half_t)o[((i_23 * 8) + 5)])), __pack_half2(((half_t)o[((i_23 * 8) + 6)]), ((half_t)o[((i_23 * 8) + 7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(O[(((int)blockIdx.x) * 4096)])), (&(((half_t*)Os)[0])), 8192);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

