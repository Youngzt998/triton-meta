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

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ K, __grid_constant__ const CUtensorMap O_desc, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const bfloat16_t* __restrict__ K, __grid_constant__ const CUtensorMap O_desc, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Os = ((void*)((char*)buf_dyn_shmem + (int64_t)0));
  void* Qs = ((void*)((char*)buf_dyn_shmem + (int64_t)0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + (int64_t)8192));
  void* Vs = ((void*)((char*)buf_dyn_shmem + (int64_t)16384));
  float o[32];
  float ls[4];
  float mx[4];
  float s[128];
  float mp[4];
  float mx_clear[4];
  float ss[4];
  bfloat16_t sc[128];
  float sm[4];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(O_desc);
  }
  #pragma unroll
  for (int64_t i = (int64_t)0; i < (int64_t)4; ++i) {
    for (int64_t vec = (int64_t)0; vec < (int64_t)8; ++vec) {
      bfloat16_t condval;
      if ((((((int64_t)blockIdx.x) * (int64_t)2) + (i >> (int64_t)1)) < (int64_t)3)) {
        condval = Q[(((((((int64_t)blockIdx.y) * (int64_t)6144) + (((int64_t)blockIdx.x) * (int64_t)4096)) + (i * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)8)) + vec)];
      } else {
        condval = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      }
      ((bfloat16_t*)Qs)[(((((i * (int64_t)1024) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)32)) + (((((((int64_t)threadIdx.x) & (int64_t)31) >> (int64_t)4) + ((((int64_t)threadIdx.x) & (int64_t)3) >> (int64_t)1)) & (int64_t)1) * (int64_t)16)) + (((((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3) + (((int64_t)threadIdx.x) & (int64_t)1)) & (int64_t)1) * (int64_t)8)) + vec)] = condval;
    }
  }
  #pragma unroll
  for (int64_t i_1 = (int64_t)0; i_1 < (int64_t)8; ++i_1) {
    for (int64_t vec_1 = (int64_t)0; vec_1 < (int64_t)4; ++vec_1) {
      o[((i_1 * (int64_t)4) + vec_1)] = 0x0p+0f/*0.000000e+00*/;
    }
  }
  for (int64_t i_2 = (int64_t)0; i_2 < (int64_t)4; ++i_2) {
    ls[i_2] = 0x0p+0f/*0.000000e+00*/;
  }
  for (int64_t i_3 = (int64_t)0; i_3 < (int64_t)4; ++i_3) {
    mx[i_3] = -CUDART_INF_F;
  }
  for (int64_t k = (int64_t)0; k < (int64_t)6; ++k) {
    __syncthreads();
    #pragma unroll
    for (int64_t i_4 = (int64_t)0; i_4 < (int64_t)4; ++i_4) {
      for (int64_t vec_2 = (int64_t)0; vec_2 < (int64_t)8; ++vec_2) {
        ((bfloat16_t*)Ks)[(((((i_4 * (int64_t)1024) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)32)) + (((((((int64_t)threadIdx.x) & (int64_t)31) >> (int64_t)4) + ((((int64_t)threadIdx.x) & (int64_t)3) >> (int64_t)1)) & (int64_t)1) * (int64_t)16)) + (((((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3) + (((int64_t)threadIdx.x) & (int64_t)1)) & (int64_t)1) * (int64_t)8)) + vec_2)] = K[(((((((int64_t)blockIdx.y) * (int64_t)24576) + (k * (int64_t)4096)) + (i_4 * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)8)) + vec_2)];
      }
    }
    #pragma unroll
    for (int64_t i_5 = (int64_t)0; i_5 < (int64_t)32; ++i_5) {
      for (int64_t vec_3 = (int64_t)0; vec_3 < (int64_t)4; ++vec_3) {
        s[((i_5 * (int64_t)4) + vec_3)] = 0x0p+0f/*0.000000e+00*/;
      }
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      __syncthreads();
      tl::initialize_wgmma_descriptor<(int64_t)2, (int64_t)1, (int64_t)32>(desc_a, (&(((bfloat16_t*)Qs)[(int64_t)0])));
      tl::initialize_wgmma_descriptor<(int64_t)2, (int64_t)1, (int64_t)32>(desc_b, (&(((bfloat16_t*)Ks)[(int64_t)0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + (int64_t)0), (int64_t)128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int64_t i_6 = (int64_t)0; i_6 < (int64_t)2; ++i_6) {
        #pragma unroll
        for (int64_t ki = (int64_t)0; ki < (int64_t)2; ++ki) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a + (((i_6 * (int64_t)4096) + (ki * (int64_t)32)) >> (int64_t)4)), uint64_t(desc_b + ((ki * (int64_t)32) >> (int64_t)4)), ((uint32_t*)(s + (i_6 * (int64_t)64))), (int64_t)1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + (int64_t)0), (int64_t)128);
    }
    __syncthreads();
    #pragma unroll
    for (int64_t i_7 = (int64_t)0; i_7 < (int64_t)4; ++i_7) {
      for (int64_t vec_4 = (int64_t)0; vec_4 < (int64_t)8; ++vec_4) {
        ((bfloat16_t*)Vs)[(((((i_7 * (int64_t)1024) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)32)) + (((((((int64_t)threadIdx.x) & (int64_t)31) >> (int64_t)4) + ((((int64_t)threadIdx.x) & (int64_t)3) >> (int64_t)1)) & (int64_t)1) * (int64_t)16)) + (((((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3) + (((int64_t)threadIdx.x) & (int64_t)1)) & (int64_t)1) * (int64_t)8)) + vec_4)] = V[(((((((int64_t)blockIdx.y) * (int64_t)24576) + (k * (int64_t)4096)) + (i_7 * (int64_t)1024)) + (((int64_t)threadIdx.x) * (int64_t)8)) + vec_4)];
      }
    }
    for (int64_t i_8 = (int64_t)0; i_8 < (int64_t)4; ++i_8) {
      mp[i_8] = mx[i_8];
    }
    for (int64_t i_9 = (int64_t)0; i_9 < (int64_t)4; ++i_9) {
      mx[i_9] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int64_t i_10 = (int64_t)0; i_10 < (int64_t)4; ++i_10) {
      mx_clear[i_10] = -CUDART_INF_F;
      #pragma unroll
      for (int64_t rv = (int64_t)0; rv < (int64_t)32; ++rv) {
        mx_clear[i_10] = max(mx_clear[i_10], s[(((((i_10 >> (int64_t)1) * (int64_t)64) + ((rv & (int64_t)15) * (int64_t)4)) + ((i_10 & (int64_t)1) * (int64_t)2)) + (rv >> (int64_t)4))]);
      }
      mx_clear[i_10] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(mx_clear[i_10]);
      mx[i_10] = max(mx[i_10], mx_clear[i_10]);
    }
    #pragma unroll
    for (int64_t i_11 = (int64_t)0; i_11 < (int64_t)4; ++i_11) {
      ss[i_11] = exp2f(((mp[i_11] * 0x1.0527dbd5cafffp-2f/*2.550349e-01*/) - (mx[i_11] * 0x1.0527dbd5cafffp-2f/*2.550349e-01*/)));
    }
    #pragma unroll
    for (int64_t i_12 = (int64_t)0; i_12 < (int64_t)32; ++i_12) {
      o[i_12] = (o[i_12] * ss[(((i_12 >> (int64_t)4) * (int64_t)2) + ((i_12 & (int64_t)3) >> (int64_t)1))]);
    }
    #pragma unroll
    for (int64_t i_13 = (int64_t)0; i_13 < (int64_t)128; ++i_13) {
      s[i_13] = exp2f(((s[i_13] * 0x1.0527dbd5cafffp-2f/*2.550349e-01*/) - (mx[(((i_13 >> (int64_t)6) * (int64_t)2) + ((i_13 & (int64_t)3) >> (int64_t)1))] * 0x1.0527dbd5cafffp-2f/*2.550349e-01*/)));
    }
    #pragma unroll
    for (int64_t i_14 = (int64_t)0; i_14 < (int64_t)32; ++i_14) {
      for (int64_t vec_5 = (int64_t)0; vec_5 < (int64_t)4; ++vec_5) {
        sc[((i_14 * (int64_t)4) + vec_5)] = ((bfloat16_t)s[((((((i_14 & (int64_t)3) >> (int64_t)1) * (int64_t)64) + ((i_14 >> (int64_t)2) * (int64_t)8)) + ((i_14 & (int64_t)1) * (int64_t)4)) + vec_5)]);
      }
    }
    {
      tl::GmmaDescriptor desc_b_1;
      __syncthreads();
      tl::initialize_wgmma_descriptor<(int64_t)2, (int64_t)0, (int64_t)32>(desc_b_1, (&(((bfloat16_t*)Vs)[(int64_t)0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(sc + (int64_t)0), (int64_t)64);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + (int64_t)0), (int64_t)32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int64_t i_15 = (int64_t)0; i_15 < (int64_t)2; ++i_15) {
        #pragma unroll
        for (int64_t ki_1 = (int64_t)0; ki_1 < (int64_t)8; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(sc + ((ki_1 * (int64_t)16) + (i_15 * (int64_t)8))), uint64_t(desc_b_1 + ((ki_1 * (int64_t)1024) >> (int64_t)4)), reinterpret_cast<uint32_t*>(o + (i_15 * (int64_t)16)), (int64_t)1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + (int64_t)0), (int64_t)32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(sc + (int64_t)0), (int64_t)64);
    }
    #pragma unroll
    for (int64_t i_16 = (int64_t)0; i_16 < (int64_t)4; ++i_16) {
      sm[i_16] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int64_t rv_1 = (int64_t)0; rv_1 < (int64_t)32; ++rv_1) {
        sm[i_16] = (sm[i_16] + s[(((((i_16 >> (int64_t)1) * (int64_t)64) + ((rv_1 & (int64_t)15) * (int64_t)4)) + ((i_16 & (int64_t)1) * (int64_t)2)) + (rv_1 >> (int64_t)4))]);
      }
      sm[i_16] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(sm[i_16]);
    }
    #pragma unroll
    for (int64_t i_17 = (int64_t)0; i_17 < (int64_t)4; ++i_17) {
      ls[i_17] = ((ls[i_17] * ss[i_17]) + sm[i_17]);
    }
  }
  #pragma unroll
  for (int64_t i_18 = (int64_t)0; i_18 < (int64_t)32; ++i_18) {
    o[i_18] = (o[i_18] / ls[(((i_18 >> (int64_t)4) * (int64_t)2) + ((i_18 & (int64_t)3) >> (int64_t)1))]);
  }
  __syncthreads();
  #pragma unroll
  for (int64_t i_19 = (int64_t)0; i_19 < (int64_t)4; ++i_19) {
    tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)Os)[((((((i_19 >> (int64_t)1) * (int64_t)2048) + ((((int64_t)threadIdx.x) >> (int64_t)5) * (int64_t)512)) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)32)) + (((((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)2) + (i_19 & (int64_t)1)) & (int64_t)1) * (int64_t)16)) + (((((((int64_t)threadIdx.x) & (int64_t)31) >> (int64_t)4) + ((((int64_t)threadIdx.x) & (int64_t)3) >> (int64_t)1)) & (int64_t)1) * (int64_t)8))])), __pack_half2(((bfloat16_t)o[(i_19 * (int64_t)8)]), ((bfloat16_t)o[((i_19 * (int64_t)8) + (int64_t)1)])), __pack_half2(((bfloat16_t)o[((i_19 * (int64_t)8) + (int64_t)2)]), ((bfloat16_t)o[((i_19 * (int64_t)8) + (int64_t)3)])), __pack_half2(((bfloat16_t)o[((i_19 * (int64_t)8) + (int64_t)4)]), ((bfloat16_t)o[((i_19 * (int64_t)8) + (int64_t)5)])), __pack_half2(((bfloat16_t)o[((i_19 * (int64_t)8) + (int64_t)6)]), ((bfloat16_t)o[((i_19 * (int64_t)8) + (int64_t)7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<(int64_t)128>()) {
    tl::fence_proxy_async();
    tl::tma_store(O_desc, (&(((bfloat16_t*)Os)[(int64_t)0])), (int64_t)0, (((int64_t)blockIdx.x) * (int64_t)128), ((int64_t)blockIdx.y), (int64_t)0);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

