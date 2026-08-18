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

extern "C" __global__ void main_kernel(const half_t* __restrict__ K, __grid_constant__ const CUtensorMap O_desc, const half_t* __restrict__ Q, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const half_t* __restrict__ K, __grid_constant__ const CUtensorMap O_desc, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Os = ((void*)((char*)buf_dyn_shmem + 0));
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 8192));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 24576));
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
  float mx_clear_2[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(O_desc);
  }
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if (((((((int)blockIdx.x) * 32) + (i * 8)) + (((int)threadIdx.x) >> 4)) < 65)) {
      condval = *(uint4*)(Q + ((((((int)blockIdx.y) * 8320) + (((int)blockIdx.x) * 4096)) + (i * 1024)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    *(uint4*)(((half_t*)Qs) + (((((i * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(o + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(ls + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -CUDART_INF_F;
  *(float2*)(mx + 0) = make_float2(broadcast_var_3, broadcast_var_3);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)Ks)[(((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((int)blockIdx.y) * 16384) + (i_2 * 1024)) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)Vs)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((int)blockIdx.y) * 16384) + (i_3 * 1024)) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)Ks)[((((((i_4 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(K[((((((int)blockIdx.y) * 16384) + (i_4 * 1024)) + (((int)threadIdx.x) * 8)) + 4096)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    tl::cp_async_gs<16>((&(((half_t*)Vs)[((((((i_5 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(V[((((((int)blockIdx.y) * 16384) + (i_5 * 1024)) + (((int)threadIdx.x) * 8)) + 4096)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 2; ++k) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(s + (i_6 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Qs)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)Ks)[0])));
      tl::increase_descriptor_offset<int>(desc_b, (k * 8192));
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
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::cp_async_gs<16>((&(((half_t*)Ks)[((((((k * 4096) + (i_7 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((int)blockIdx.y) * 16384) + (k * 4096)) + (i_7 * 1024)) + (((int)threadIdx.x) * 8)) + 8192)])));
    }
    tl::cp_async_commit();
    *(float2*)(mp + 0) = *(float2*)(mx + 0);
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(mx + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      mx_clear[i_8] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        mx_clear[i_8] = max(mx_clear[i_8], s[((((rv & 7) * 4) + (i_8 * 2)) + (rv >> 3))]);
      }
      mx_clear[i_8] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(mx_clear[i_8]);
      mx[i_8] = max(mx[i_8], mx_clear[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      ss[i_9] = exp2f(((mp[i_9] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[i_9] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 32; ++i_10) {
      o[i_10] = (o[i_10] * ss[((i_10 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      s[i_11] = exp2f(((s[i_11] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[((i_11 & 3) >> 1)] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 8; ++i_12) {
      uint2 __1;
      float4 v_ = *(float4*)(s + (i_12 * 4));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(sc + (i_12 * 4)) = __1;
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((half_t*)Vs)[0])));
      tl::increase_descriptor_offset<int>(desc_b_1, (k * 8192));
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
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      tl::cp_async_gs<16>((&(((half_t*)Vs)[((((((k * 4096) + (i_13 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((((int)blockIdx.y) * 16384) + (k * 4096)) + (i_13 * 1024)) + (((int)threadIdx.x) * 8)) + 8192)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      sm[i_14] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        sm[i_14] = (sm[i_14] + s[((((rv_1 & 7) * 4) + (i_14 * 2)) + (rv_1 >> 3))]);
      }
      sm[i_14] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(sm[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      ls[i_15] = ((ls[i_15] * ss[i_15]) + sm[i_15]);
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 8; ++i_16) {
    float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(s + (i_16 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
  }
  tl::cp_async_wait<3>();
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
  float broadcast_var_7 = -CUDART_INF_F;
  *(float2*)(mx + 0) = make_float2(broadcast_var_7, broadcast_var_7);
  #pragma unroll
  for (int i_17 = 0; i_17 < 2; ++i_17) {
    mx_clear_1[i_17] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      mx_clear_1[i_17] = max(mx_clear_1[i_17], s[((((rv_2 & 7) * 4) + (i_17 * 2)) + (rv_2 >> 3))]);
    }
    mx_clear_1[i_17] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(mx_clear_1[i_17]);
    mx[i_17] = max(mx[i_17], mx_clear_1[i_17]);
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    ss[i_18] = exp2f(((mp[i_18] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[i_18] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 32; ++i_19) {
    o[i_19] = (o[i_19] * ss[((i_19 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 32; ++i_20) {
    s[i_20] = exp2f(((s[i_20] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[((i_20 & 3) >> 1)] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 8; ++i_21) {
    uint2 __2;
    float4 v__1 = *(float4*)(s + (i_21 * 4));
    ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
    ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
    *(uint2*)(sc + (i_21 * 4)) = __2;
  }
  tl::cp_async_wait<2>();
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
  for (int i_22 = 0; i_22 < 2; ++i_22) {
    sm[i_22] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      sm[i_22] = (sm[i_22] + s[((((rv_3 & 7) * 4) + (i_22 * 2)) + (rv_3 >> 3))]);
    }
    sm[i_22] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(sm[i_22]);
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 2; ++i_23) {
    ls[i_23] = ((ls[i_23] * ss[i_23]) + sm[i_23]);
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 8; ++i_24) {
    float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(s + (i_24 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_2;
    tl::GmmaDescriptor desc_b_4;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)Qs)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_4, (&(((half_t*)Ks)[0])));
    tl::increase_descriptor_offset<int>(desc_b_4, 8192);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((ki_4 * 32) >> 4)), uint64_t(desc_b_4 + ((ki_4 * 32) >> 4)), ((uint32_t*)(s + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
  }
  *(float2*)(mp + 0) = *(float2*)(mx + 0);
  float broadcast_var_9 = -CUDART_INF_F;
  *(float2*)(mx + 0) = make_float2(broadcast_var_9, broadcast_var_9);
  #pragma unroll
  for (int i_25 = 0; i_25 < 2; ++i_25) {
    mx_clear_2[i_25] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
      mx_clear_2[i_25] = max(mx_clear_2[i_25], s[((((rv_4 & 7) * 4) + (i_25 * 2)) + (rv_4 >> 3))]);
    }
    mx_clear_2[i_25] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(mx_clear_2[i_25]);
    mx[i_25] = max(mx[i_25], mx_clear_2[i_25]);
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 2; ++i_26) {
    ss[i_26] = exp2f(((mp[i_26] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[i_26] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 32; ++i_27) {
    o[i_27] = (o[i_27] * ss[((i_27 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 32; ++i_28) {
    s[i_28] = exp2f(((s[i_28] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[((i_28 & 3) >> 1)] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 8; ++i_29) {
    uint2 __3;
    float4 v__2 = *(float4*)(s + (i_29 * 4));
    ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
    ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
    *(uint2*)(sc + (i_29 * 4)) = __3;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_b_5;
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_5, (&(((half_t*)Vs)[0])));
    tl::increase_descriptor_offset<int>(desc_b_5, 8192);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(sc + 0), 16);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(sc + (ki_5 * 8)), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), reinterpret_cast<uint32_t*>(o + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(sc + 0), 16);
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 2; ++i_30) {
    sm[i_30] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
      sm[i_30] = (sm[i_30] + s[((((rv_5 & 7) * 4) + (i_30 * 2)) + (rv_5 >> 3))]);
    }
    sm[i_30] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(sm[i_30]);
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 2; ++i_31) {
    ls[i_31] = ((ls[i_31] * ss[i_31]) + sm[i_31]);
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 32; ++i_32) {
    o[i_32] = (o[i_32] / ls[((i_32 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_33 = 0; i_33 < 4; ++i_33) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)Os)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_33 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_33 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)o[(i_33 * 8)]), ((half_t)o[((i_33 * 8) + 1)])), __pack_half2(((half_t)o[((i_33 * 8) + 2)]), ((half_t)o[((i_33 * 8) + 3)])), __pack_half2(((half_t)o[((i_33 * 8) + 4)]), ((half_t)o[((i_33 * 8) + 5)])), __pack_half2(((half_t)o[((i_33 * 8) + 6)]), ((half_t)o[((i_33 * 8) + 7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store(O_desc, (&(((half_t*)Os)[0])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y), 0);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

