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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, float* __restrict__ Out);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, float* __restrict__ Out) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* As = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 65536));
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float acc[16];
  float As_frag[64];
  float tmp[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int ko = 0; ko < 16; ++ko) {
      mbarrier[((ko & 1) + 2)].wait((((ko & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(ko & 1)].arrive_and_expect_tx(32768);
        tl::tma_load(A_desc, mbarrier[(ko & 1)], (&(((float*)As)[((ko & 1) * 8192)])), (ko * 256), (((int)blockIdx.x) * 32));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int ko_1 = 0; ko_1 < 16; ++ko_1) {
      mbarrier[(ko_1 & 1)].wait(((ko_1 & 3) >> 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        *(float4*)(As_frag + (i_1 * 4)) = *(float4*)(((float*)As) + (((((ko_1 & 1) * 8192) + (i_1 * 512)) + (((int)threadIdx.x) * 4)) - 512));
      }
      mbarrier[((ko_1 & 1) + 2)].arrive();
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_2 = 0; i_2 < 16; ++i_2) {
        tmp[i_2] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv = 0; rv < 4; ++rv) {
          tmp[i_2] = (tmp[i_2] + As_frag[((i_2 * 4) + rv)]);
        }
        tmp[i_2] = tl::AllReduce<tl::SumOp, 64, 1, 128, tl::NamedBarrier<128>>::run(tmp[i_2], (&(((float*)workspace)[0])));
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        acc[i_3] = (acc[i_3] + tmp[i_3]);
      }
    }
    if ((((int)threadIdx.x) % 64) == 0) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 16; ++i_4) {
        Out[((((((int)blockIdx.x) * 32) + (i_4 * 2)) + (((int)threadIdx.x) >> 6)) - 2)] = acc[i_4];
      }
    }
  }
}

