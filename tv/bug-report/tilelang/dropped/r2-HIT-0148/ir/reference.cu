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
  void* As = ((void*)((char*)buf_dyn_shmem + 0));
  void* Xs = ((void*)((char*)buf_dyn_shmem + 32768));
  __shared__ __align__(16) uint64_t mbarrier_mem[2];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float acc[2];
  float p[256];
  float acc_clear[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(32);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    tl::__sync_thread_partial(3, 32);
    for (int ko = 0; ko < 9; ++ko) {
      mbarrier[1].wait(((ko & 1) ^ 1));
      #pragma unroll
      for (int i = 0; i < 256; ++i) {
        float condval;
        if (((i < 124) && ((((ko * 128) + ((i & 3) * 32)) + ((int)threadIdx.x)) < 1027))) {
          condval = A[(((((i >> 2) * 1027) + (ko * 128)) + ((i & 3) * 32)) + ((int)threadIdx.x))];
        } else {
          condval = 0x0p+0f/*0.000000e+00*/;
        }
        ((float*)As)[((i * 32) + ((int)threadIdx.x))] = condval;
      }
      tl::__sync_thread_partial(3, 32);
      if (tl::tl_shuffle_elect<32>()) {
        mbarrier[0].arrive_and_expect_tx(512);
        tl::fence_proxy_async();
        tl::tma_load(X_desc, mbarrier[0], (&(((float*)Xs)[0])), (ko * 128));
      }
    }
  } else {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(acc + 0) = make_float2(broadcast_var, broadcast_var);
    for (int ko_1 = 0; ko_1 < 9; ++ko_1) {
      mbarrier[0].wait((ko_1 & 1));
      #pragma unroll
      for (int i_1 = 0; i_1 < 64; ++i_1) {
        float4 __1;
          float4 v_ = *(float4*)(((float*)As) + (((i_1 * 128) + (((int)threadIdx.x) * 4)) - 128));
          float4 v__1 = *(float4*)(((float*)Xs) + ((((int)threadIdx.x) * 4) - 128));
          __1.x = (v_.x*v__1.x);
          __1.y = (v_.y*v__1.y);
          __1.z = (v_.z*v__1.z);
          __1.w = (v_.w*v__1.w);
        *(float4*)(p + (i_1 * 4)) = __1;
      }
      mbarrier[1].arrive();
      #pragma unroll
      for (int i_2 = 0; i_2 < 64; ++i_2) {
        acc_clear[i_2] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv = 0; rv < 4; ++rv) {
          acc_clear[i_2] = (acc_clear[i_2] + p[((i_2 * 4) + rv)]);
        }
        acc_clear[i_2] = tl::AllReduce<tl::SumOp, 32, 1, 32, tl::NamedBarrier<32>>::run(acc_clear[i_2]);
        if (((((i_2 >> 5) * 32) + ((int)threadIdx.x)) - 32) == i_2) {
          acc[(i_2 >> 5)] = (acc[(i_2 >> 5)] + acc_clear[i_2]);
        }
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      if (((i_3 * 32) + ((int)threadIdx.x)) < 63) {
        Y[(((i_3 * 32) + ((int)threadIdx.x)) - 32)] = acc[i_3];
      }
    }
  }
}

