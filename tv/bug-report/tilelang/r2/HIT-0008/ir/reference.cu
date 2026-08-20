#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ A, __grid_constant__ const CUtensorMap X_desc, float* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(64, 1) main_kernel(const float* __restrict__ A, __grid_constant__ const CUtensorMap X_desc, float* __restrict__ Y) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* As = ((void*)((char*)buf_dyn_shmem + (int64_t)0));
  void* Xs = ((void*)((char*)buf_dyn_shmem + (int64_t)196608));
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float acc[2];
  float p[512];
  float acc_clear[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
  }
  if (tl::tl_shuffle_elect<(int64_t)0>()) {
    mbarrier[(int64_t)0].init((int64_t)1);
    mbarrier[(int64_t)1].init((int64_t)1);
    mbarrier[(int64_t)2].init((int64_t)1);
    mbarrier[(int64_t)3].init((int64_t)32);
    mbarrier[(int64_t)4].init((int64_t)32);
    mbarrier[(int64_t)5].init((int64_t)32);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int64_t)threadIdx.x) < (int64_t)32) {
    tl::__sync_thread_partial(3, 32);
    for (int64_t ko = (int64_t)0; ko < (int64_t)5; ++ko) {
      mbarrier[((ko % (int64_t)3) + (int64_t)3)].wait(((ko / (int64_t)3) ^ (int64_t)1));
      #pragma unroll
      for (int64_t i = (int64_t)0; i < (int64_t)512; ++i) {
        float condval;
        if (((((((int64_t)blockIdx.x) * (int64_t)64) + (i >> (int64_t)3)) < (int64_t)511) && ((((ko * (int64_t)256) + ((i & (int64_t)7) * (int64_t)32)) + ((int64_t)threadIdx.x)) < (int64_t)1025))) {
          condval = A[(((((((int64_t)blockIdx.x) * (int64_t)65600) + ((i >> (int64_t)3) * (int64_t)1025)) + (ko * (int64_t)256)) + ((i & (int64_t)7) * (int64_t)32)) + ((int64_t)threadIdx.x))];
        } else {
          condval = 0x0p+0f/*0.000000e+00*/;
        }
        ((float*)As)[((((ko % (int64_t)3) * (int64_t)16384) + (i * (int64_t)32)) + ((int64_t)threadIdx.x))] = condval;
      }
      tl::__sync_thread_partial(3, 32);
      if (tl::tl_shuffle_elect<(int64_t)32>()) {
        mbarrier[(ko % (int64_t)3)].arrive_and_expect_tx((int64_t)1024);
        tl::fence_proxy_async();
        tl::tma_load(X_desc, mbarrier[(ko % (int64_t)3)], (&(((float*)Xs)[((ko % (int64_t)3) * (int64_t)256)])), (ko * (int64_t)256));
      }
    }
  } else {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(acc + (int64_t)0) = make_float2(broadcast_var, broadcast_var);
    for (int64_t ko_1 = (int64_t)0; ko_1 < (int64_t)5; ++ko_1) {
      mbarrier[(ko_1 % (int64_t)3)].wait((ko_1 / (int64_t)3));
      #pragma unroll
      for (int64_t i_1 = (int64_t)0; i_1 < (int64_t)128; ++i_1) {
        float4 __1;
          float4 v_ = *(float4*)(((float*)As) + (((((ko_1 % (int64_t)3) * (int64_t)16384) + (i_1 * (int64_t)128)) + (((int64_t)threadIdx.x) * (int64_t)4)) - (int64_t)128));
          float4 v__1 = *(float4*)(((float*)Xs) + (((((ko_1 % (int64_t)3) * (int64_t)256) + ((i_1 & (int64_t)1) * (int64_t)128)) + (((int64_t)threadIdx.x) * (int64_t)4)) - (int64_t)128));
          __1.x = (v_.x*v__1.x);
          __1.y = (v_.y*v__1.y);
          __1.z = (v_.z*v__1.z);
          __1.w = (v_.w*v__1.w);
        *(float4*)(p + ((i_1 * (int64_t)4) + (int64_t)4)) = __1;
      }
      mbarrier[((ko_1 % (int64_t)3) + (int64_t)3)].arrive();
      #pragma unroll
      for (int64_t i_2 = (int64_t)0; i_2 < (int64_t)64; ++i_2) {
        acc_clear[i_2] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int64_t rv = (int64_t)0; rv < (int64_t)8; ++rv) {
          acc_clear[i_2] = (acc_clear[i_2] + p[(((i_2 * (int64_t)8) + ((rv & (int64_t)1) * (int64_t)4)) + (rv >> (int64_t)1))]);
        }
        acc_clear[i_2] = tl::AllReduce<tl::SumOp, 32, 1, 32, tl::NamedBarrier<32>>::run(acc_clear[i_2]);
        if (((((i_2 >> (int64_t)5) * (int64_t)32) + ((int64_t)threadIdx.x)) - (int64_t)32) == i_2) {
          acc[(i_2 >> (int64_t)5)] = (acc[(i_2 >> (int64_t)5)] + acc_clear[i_2]);
        }
      }
    }
    #pragma unroll
    for (int64_t i_3 = (int64_t)0; i_3 < (int64_t)2; ++i_3) {
      if ((((((int64_t)blockIdx.x) * (int64_t)64) + (i_3 * (int64_t)32)) + ((int64_t)threadIdx.x)) < (int64_t)543) {
        Y[((((((int64_t)blockIdx.x) * (int64_t)64) + (i_3 * (int64_t)32)) + ((int64_t)threadIdx.x)) - (int64_t)32)] = acc[i_3];
      }
    }
  }
}

