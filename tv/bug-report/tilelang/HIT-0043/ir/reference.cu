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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, const half_t* __restrict__ Sinks, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, const half_t* __restrict__ Sinks, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  __shared__ __align__(16) uint64_t mbarrier_mem[9];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  half_t sinks[2];
  float acc_s[64];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_scale[2];
  float scores_sum[2];
  half_t acc_s_cast[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
    mbarrier[4].init(256);
    mbarrier[5].init(256);
    mbarrier[6].init(256);
    mbarrier[7].init(256);
    mbarrier[8].init(1);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (tl::tl_shuffle_elect<384>()) {
    mbarrier[8].arrive_and_expect_tx(32768);
    tl::tma_load(Q_desc, mbarrier[8], (&(((half_t*)Q_shared)[0])), 0, (((int)blockIdx.x) * 128), 0, 0);
    tl::tma_load(Q_desc, mbarrier[8], (&(((half_t*)Q_shared)[8192])), 64, (((int)blockIdx.x) * 128), 0, 0);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < (((int)blockIdx.x) + 1); ++k) {
      mbarrier[(k + 4)].wait(1);
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[k].arrive_and_expect_tx(32768);
        tl::tma_load(K_desc, mbarrier[k], (&(((half_t*)K_shared)[(k * 16384)])), 0, (k * 128), 0, 0);
        tl::tma_load(K_desc, mbarrier[k], (&(((half_t*)K_shared)[((k * 16384) + 8192)])), 64, (k * 128), 0, 0);
      }
      mbarrier[(k + 6)].wait(1);
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k + 2)].arrive_and_expect_tx(32768);
        tl::tma_load(V_desc, mbarrier[(k + 2)], (&(((half_t*)V_shared)[(k * 16384)])), 0, (k * 128), 0, 0);
        tl::tma_load(V_desc, mbarrier[(k + 2)], (&(((half_t*)V_shared)[((k * 16384) + 8192)])), 64, (k * 128), 0, 0);
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    *(uint1*)(sinks + 0) = make_uint1(__pack_half2(Sinks[0], Sinks[0]));
    for (int k_1 = 0; k_1 < (((int)blockIdx.x) + 1); ++k_1) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 64; ++i_1) {
        float condval;
        if (((((((k_1 * 128) + ((i_1 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_1 & 1)) + 64) <= ((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_1 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval = -CUDART_INF_F;
        }
        acc_s[i_1] = condval;
      }
      if (k_1 == 0) {
        mbarrier[8].wait(0);
      }
      mbarrier[k_1].wait(0);
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b, (k_1 * 32768));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + ((((((int)threadIdx.x) >> 7) + 1) & 1) * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 16384) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 128);
      }
      mbarrier[(k_1 + 4)].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_3 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        scores_max_clear[i_2] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 32; ++rv) {
          scores_max_clear[i_2] = max(scores_max_clear[i_2], acc_s[((((rv & 15) * 4) + (i_2 * 2)) + (rv >> 4))]);
        }
        scores_max_clear[i_2] = tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<256>>::run(scores_max_clear[i_2]);
        scores_max[i_2] = max(scores_max[i_2], scores_max_clear[i_2]);
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        scores_max[i_3] = max(scores_max[i_3], scores_max_prev[i_3]);
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        scores_scale[i_4] = exp2f(((scores_max_prev[i_4] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_4] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 64; ++i_5) {
        acc_s[i_5] = exp2f(((acc_s[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_5 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        scores_sum[i_6] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
          scores_sum[i_6] = (scores_sum[i_6] + acc_s[((((rv_1 & 15) * 4) + (i_6 * 2)) + (rv_1 >> 4))]);
        }
        scores_sum[i_6] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<256>>::run(scores_sum[i_6]);
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        logsum[i_7] = ((logsum[i_7] * scores_scale[i_7]) + scores_sum[i_7]);
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        uint2 __1;
        float4 v_ = *(float4*)(acc_s + (i_8 * 4));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
        *(uint2*)(acc_s_cast + (i_8 * 4)) = __1;
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 64; ++i_9) {
        acc_o[i_9] = (acc_o[i_9] * scores_scale[((i_9 & 3) >> 1)]);
      }
      mbarrier[(k_1 + 2)].wait(0);
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_1, (&(((half_t*)V_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, (k_1 * 32768));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
      }
      mbarrier[(k_1 + 6)].arrive();
    }
    float broadcast_var_4 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float broadcast_var_5 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float2 __2;
      float2 v__1 = *(float2*)(logsum + 0);
      float2 __3;
      float2 __4;
        float2 __5;
          float2 __6;
          uint1 v__2 = *(uint1*)(sinks + 0);
          ((float2*)(&__6))[0] = __half22float2(((half2*)(&v__2))[0]);
          float2 v__3 = make_float2(broadcast_var_4, broadcast_var_4);
          __5.x = (__6.x*v__3.x);
          __5.y = (__6.y*v__3.y);
        float2 __7;
          float2 v__4 = *(float2*)(scores_max + 0);
          float2 v__5 = make_float2(broadcast_var_5, broadcast_var_5);
          __7.x = (v__4.x*v__5.x);
          __7.y = (v__4.y*v__5.y);
        __4.x = (__5.x-__7.x);
        __4.y = (__5.y-__7.y);
      __3.x = exp2f(__4.x);
      __3.y = exp2f(__4.y);
      __2.x = (v__1.x+__3.x);
      __2.y = (v__1.y+__3.y);
    *(float2*)(logsum + 0) = __2;
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      acc_o[i_10] = (acc_o[i_10] / logsum[((i_10 & 3) >> 1)]);
    }
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[((((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_11 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 8192)])), __pack_half2(((half_t)acc_o[(i_11 * 8)]), ((half_t)acc_o[((i_11 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_11 * 8) + 2)]), ((half_t)acc_o[((i_11 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_11 * 8) + 4)]), ((half_t)acc_o[((i_11 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_11 * 8) + 6)]), ((half_t)acc_o[((i_11 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 256);
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      tl::tma_store((&(Output[(((int)blockIdx.x) * 16384)])), (&(((half_t*)O_shared)[0])), 32768);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

