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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap X_desc, float* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(160, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap X_desc, float* __restrict__ Y) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* As = ((void*)((char*)buf_dyn_shmem + 0));
  void* Xs = ((void*)((char*)buf_dyn_shmem + 8192));
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float acc[8];
  float p[64];
  half_t As_local_cast[8];
  half_t Xs_local_cast_1[8];
  float acc_clear[8];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(X_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(32);
    mbarrier[3].init(32);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int ko = 0; ko < 2; ++ko) {
      mbarrier[(ko + 2)].wait(1);
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[ko].expect_transaction(4096);
        tl::tma_load(A_desc, mbarrier[ko], (&(((half_t*)As)[(ko * 2048)])), (ko * 32), 0);
        mbarrier[ko].arrive_and_expect_tx(64);
        tl::tma_load(X_desc, mbarrier[ko], (&(((half_t*)Xs)[(ko * 32)])), (ko * 32));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int ko_1 = 0; ko_1 < 2; ++ko_1) {
      mbarrier[ko_1].wait(0);
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        *(uint4*)(As_local_cast + 0) = *(uint4*)(((half_t*)As) + ((((ko_1 * 2048) + (i_1 * 256)) + (((int)threadIdx.x) * 8)) - 1024));
        *(uint4*)(Xs_local_cast_1 + 0) = *(uint4*)(((half_t*)Xs) + ((ko_1 * 32) + ((((int)threadIdx.x) & 3) * 8)));
        for (int vec = 0; vec < 2; ++vec) {
          float4 __1;
            float4 __2;
            uint2 v_ = *(uint2*)(As_local_cast + (vec * 4));
            ((float2*)(&__2))[0] = __half22float2(((half2*)(&v_))[0]);
            ((float2*)(&__2))[1] = __half22float2(((half2*)(&v_))[1]);
            float4 __3;
            uint2 v__1 = *(uint2*)(Xs_local_cast_1 + (vec * 4));
            ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__1))[0]);
            ((float2*)(&__3))[1] = __half22float2(((half2*)(&v__1))[1]);
            __1.x = (__2.x*__3.x);
            __1.y = (__2.y*__3.y);
            __1.z = (__2.z*__3.z);
            __1.w = (__2.w*__3.w);
          *(float4*)(p + ((i_1 * 8) + (vec * 4))) = __1;
        }
      }
      mbarrier[(ko_1 + 2)].arrive();
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        acc_clear[i_2] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv = 0; rv < 8; ++rv) {
          acc_clear[i_2] = (acc_clear[i_2] + p[((i_2 * 8) + rv)]);
        }
        acc_clear[i_2] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<32>>::run(acc_clear[i_2]);
        acc[i_2] = (acc[i_2] + acc_clear[i_2]);
      }
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        Y[(((i_3 * 8) + (((int)threadIdx.x) >> 2)) - 32)] = acc[i_3];
      }
    }
  }
}

