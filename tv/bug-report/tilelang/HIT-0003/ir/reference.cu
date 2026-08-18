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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ O, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ O, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Os = ((void*)((char*)buf_dyn_shmem + 0));
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 8192));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 16384));
  __shared__ __align__(16) uint64_t mbarrier_mem[5];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float o[32];
  float ls[2];
  float mx[2];
  float s[32];
  float mp[2];
  float mx_clear[2];
  float ss[2];
  half_t sc[32];
  float sm[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(1);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (tl::tl_shuffle_elect<256>()) {
    mbarrier[4].arrive_and_expect_tx(8192);
    tl::tma_load(Q_desc, mbarrier[4], (&(((half_t*)Qs)[0])), 0, (((int)blockIdx.x) * 64), 0, 0);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < (((int)blockIdx.x) + 3); ++k) {
      mbarrier[2].wait(((k & 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[0].arrive_and_expect_tx(8192);
        tl::tma_load(K_desc, mbarrier[0], (&(((half_t*)Ks)[0])), 0, (k * 64), 0, 0);
      }
      mbarrier[3].wait(((k & 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[1].arrive_and_expect_tx(8192);
        tl::tma_load(V_desc, mbarrier[1], (&(((half_t*)Vs)[0])), 0, (k * 64), 0, 0);
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(ls + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float2*)(mx + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    for (int k_1 = 0; k_1 < (((int)blockIdx.x) + 3); ++k_1) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 32; ++i_1) {
        float condval;
        if ((((((k_1 * 64) + ((i_1 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_1 & 1)) <= (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_1 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 64))) {
          condval = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval = -CUDART_INF_F;
        }
        s[i_1] = condval;
      }
      if (k_1 == 0) {
        mbarrier[4].wait(0);
      }
      mbarrier[0].wait((k_1 & 1));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Qs)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)Ks)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + ((ki * 32) >> 4)), uint64_t(desc_b + ((ki * 32) >> 4)), ((uint32_t*)(s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
      }
      mbarrier[2].arrive();
      *(float2*)(mp + 0) = *(float2*)(mx + 0);
      float broadcast_var_3 = -CUDART_INF_F;
      *(float2*)(mx + 0) = make_float2(broadcast_var_3, broadcast_var_3);
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        mx_clear[i_2] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 16; ++rv) {
          mx_clear[i_2] = max(mx_clear[i_2], s[((((rv & 7) * 4) + (i_2 * 2)) + (rv >> 3))]);
        }
        mx_clear[i_2] = tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<128>>::run(mx_clear[i_2]);
        mx[i_2] = max(mx[i_2], mx_clear[i_2]);
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        ss[i_3] = exp2f(((mp[i_3] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[i_3] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 32; ++i_4) {
        o[i_4] = (o[i_4] * ss[((i_4 & 3) >> 1)]);
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        s[i_5] = exp2f(((s[i_5] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[((i_5 & 3) >> 1)] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        uint2 __1;
        float4 v_ = *(float4*)(s + (i_6 * 4));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
        *(uint2*)(sc + (i_6 * 4)) = __1;
      }
      mbarrier[1].wait((k_1 & 1));
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((half_t*)Vs)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(sc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(sc + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(o + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(sc + 0), 16);
      }
      mbarrier[3].arrive();
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        sm[i_7] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
          sm[i_7] = (sm[i_7] + s[((((rv_1 & 7) * 4) + (i_7 * 2)) + (rv_1 >> 3))]);
        }
        sm[i_7] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run(sm[i_7]);
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        ls[i_8] = ((ls[i_8] * ss[i_8]) + sm[i_8]);
      }
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      o[i_9] = (o[i_9] / ls[((i_9 & 3) >> 1)]);
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)Os)[((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (i_10 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 4096)])), __pack_half2(((half_t)o[(i_10 * 8)]), ((half_t)o[((i_10 * 8) + 1)])), __pack_half2(((half_t)o[((i_10 * 8) + 2)]), ((half_t)o[((i_10 * 8) + 3)])), __pack_half2(((half_t)o[((i_10 * 8) + 4)]), ((half_t)o[((i_10 * 8) + 5)])), __pack_half2(((half_t)o[((i_10 * 8) + 6)]), ((half_t)o[((i_10 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store((&(O[(((int)blockIdx.x) * 4096)])), (&(((half_t*)Os)[0])), 8192);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

