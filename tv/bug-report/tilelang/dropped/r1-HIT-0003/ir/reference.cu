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
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a, (&(((fp8_e5_t*)A_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_a, ((k_1 % 3) * 8192));
        tl::initialize_wgmma_descriptor<2, 1, 32>(desc_b, (&(((fp8_e5_t*)B_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b, ((k_1 % 3) * 8192));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          #pragma unroll
          for (int ki = 0; ki < 2; ++ki) {
            tl::wgmma_ss<tl::DataType::kFloat8_e5m2, tl::DataType::kFloat8_e5m2, tl::DataType::kFloat32, 64, 128, 32, false, false, 1, 1>(uint64_t(desc_a + (((i_2 * 4096) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 32) >> 4)), ((uint32_t*)(C_local + (i_2 * 64))), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
      }
      mbarrier[((k_1 % 3) + 3)].arrive();
      if (((k_1 + 1) % 2) == 0) {
        #pragma unroll
        for (int i_3 = 0; i_3 < 128; ++i_3) {
          C_local_accum[i_3] = (C_local_accum[i_3] + C_local[i_3]);
        }
        #pragma unroll
        for (int i_4 = 0; i_4 < 32; ++i_4) {
          float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(C_local + (i_4 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
        }
      }
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_5 = 0; i_5 < 64; ++i_5) {
      *(float2*)(((float*)C_shared) + ((((((((((((i_5 & 31) >> 3) * 4096) + ((i_5 >> 5) * 2048)) + ((((int)threadIdx.x) >> 5) * 512)) + ((i_5 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_5 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4)) + ((((int)threadIdx.x) & 1) * 2)) - 2048)) = *(float2*)(C_local_accum + (i_5 * 2));
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

