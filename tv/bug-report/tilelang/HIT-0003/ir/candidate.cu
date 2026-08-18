#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void matmul_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(256, 1) matmul_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* C_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_local[128];
  float C_local_accum[128];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(C_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 128; ++k) {
      mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k % 3)].expect_transaction(8192);
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((fp8_e5_t*)A_shared)[((k % 3) * 8192)])), (k * 64), (((int)blockIdx.y) * 128));
        mbarrier[(k % 3)].arrive_and_expect_tx(8192);
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((fp8_e5_t*)B_shared)[((k % 3) * 8192)])), (k * 64), (((int)blockIdx.x) * 128));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local_accum + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    for (int k_1 = 0; k_1 < 128; ++k_1) {
      mbarrier[(k_1 % 3)].wait(((k_1 % 6) / 3));
      {
        fp8_e5_t A_local[64];
        fp8_e5_t B_local[64];
        for (int ki = 0; ki < 2; ++ki) {
          for (int i_2 = 0; i_2 < 4; ++i_2) {
            tl::ptx_ldmatrix_x4((&(((fp8_e5_t*)A_shared)[(((((((k_1 % 3) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (i_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(A_local[(i_2 * 16)])));
          }
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4((&(((fp8_e5_t*)B_shared)[((((((((k_1 % 3) * 8192) + (((((int)threadIdx.x) & 127) >> 6) * 4096)) + (i_3 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(B_local[(i_3 * 16)])));
          }
          for (int i_4 = 0; i_4 < 4; ++i_4) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat8_e5m2, tl::DataType::kFloat8_e5m2, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_local + ((i_4 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_4 * 16)), reinterpret_cast<const unsigned*>(B_local + (j * 16)));
              tl::mma_sync<tl::DataType::kFloat8_e5m2, tl::DataType::kFloat8_e5m2, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_local + (((i_4 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_4 * 16)), reinterpret_cast<const unsigned*>(B_local + ((j * 16) + 8)));
            }
          }
        }
      }
      mbarrier[((k_1 % 3) + 3)].arrive();
      if (((k_1 + 1) % 2) == 0) {
        #pragma unroll
        for (int i_5 = 0; i_5 < 128; ++i_5) {
          C_local_accum[i_5] = (C_local_accum[i_5] + C_local[i_5]);
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(C_local + (i_6 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
        }
      }
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_7 = 0; i_7 < 64; ++i_7) {
      *(float2*)(((float*)C_shared) + ((((((((((((((int)threadIdx.x) >> 6) * 8192) + (((i_7 & 15) >> 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + ((i_7 >> 4) * 512)) + ((i_7 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_7 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_7 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4)) + ((((int)threadIdx.x) & 1) * 2)) - 16384)) = *(float2*)(C_local_accum + (i_7 * 2));
    }
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store(C_desc, (&(((float*)C_shared)[0])), (((int)blockIdx.x) * 128), (((int)blockIdx.y) * 128));
      tl::tma_store(C_desc, (&(((float*)C_shared)[4096])), ((((int)blockIdx.x) * 128) + 32), (((int)blockIdx.y) * 128));
      tl::tma_store(C_desc, (&(((float*)C_shared)[8192])), ((((int)blockIdx.x) * 128) + 64), (((int)blockIdx.y) * 128));
      tl::tma_store(C_desc, (&(((float*)C_shared)[12288])), ((((int)blockIdx.x) * 128) + 96), (((int)blockIdx.y) * 128));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

