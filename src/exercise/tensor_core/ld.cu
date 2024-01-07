#include "cuda_runtime.h"
#include "gflags/gflags.h"
#include "utils.cuh"
#include "utils.h"
#include "gtest/gtest.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

__global__ void ldmatrix(const half *input) {
  __shared__ half smem[64];
  int tid = threadIdx.x;
  smem[2 * tid] = input[2 * tid];
  smem[2 * tid + 1] = input[2 * tid + 1];
  __syncthreads();

  union F {
    half2 fp16_data;
    float fp32_data;
  };
  F f;

  auto smem_addr = __cvta_generic_to_shared(&smem) + tid * 16;

  __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0},[%1];"
                       : "=f"(f.fp32_data)
                       : "l"(smem_addr)
                       : "memory");

  printf("Tid: %d, Value: %f, %f\n", tid, (float)(f.fp16_data.x),
         (float)(f.fp16_data.y));
}

__global__ void m8nnk4(const half *a, const half *b, half *c) {
  __shared__ half a_smem[64];
  __shared__ half b_smem[64];
  int tid = threadIdx.x;
  a_smem[2 * tid] = a[2 * tid];
  a_smem[2 * tid + 1] = a[2 * tid + 1];

  b_smem[2 * tid] = b[2 * tid];
  b_smem[2 * tid + 1] = b[2 * tid + 1];
  __syncthreads();

  union F {
    half2 fp16_data;
    ushort2 u16_data;
    float fp32_data;
  };
  F fa, fb;

  auto a_smem_addr = __cvta_generic_to_shared(&a_smem) + tid * 16;
  auto b_smem_addr = __cvta_generic_to_shared(&b_smem) + tid * 16;

  __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0},[%1];"
                       : "=f"(fa.fp32_data)
                       : "l"(a_smem_addr)
                       : "memory");

  __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0},[%1];"
                       : "=f"(fb.fp32_data)
                       : "l"(b_smem_addr)
                       : "memory");

  unsigned c0(0), c1(0), c2(0), c3(0);
  __asm__ __volatile__(
      R"(    
        mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 {%0,%1,%2,%3},{%4,%5},{%6,%7},{%0,%1,%2,%3};)"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"((unsigned)fa.u16_data.x), "r"((unsigned)fa.u16_data.y),
        "r"((unsigned)fb.u16_data.x), "r"((unsigned)fb.u16_data.y)
      : "memory");

  printf("Tid: %d, Value: %f, %f, %f,  %f\n", tid, (float)(c0), (float)(c1),
         (float)(c2), (float)(c3));
}
// threadFrags: 32x16
__global__ void WMMAFrag(const half *input, float *threadFrags) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::load_matrix_sync(a_frag, input, 16);
  int tid = threadIdx.x;
  for (int i = 0; i < a_frag.num_elements; i++) {
    float t = static_cast<float>(a_frag.x[i]);
    threadFrags[tid * 16 + i] = t;
    if (threadIdx.x == 2) {
      printf("Idex: %d, ThreadId :%d, Value: %f \n", i, threadIdx.x, t);
    }
  }
}

TEST(TensorCore, ld) {
  int size = 16 * 16;
  int bytes = size * sizeof(half);
  std::vector<float> h_fp32_input = CreateData<float>(size, 0, size);
  std::vector<float> h_frags(32 * 16, 0);
  std::vector<half> h_fp16_input = Convert<float, half>(h_fp32_input);
  half *d_input;
  float *d_output;
  dim3 dimBlock(32, 1);
  dim3 dimGrid(1, 1);
  CUDA_ERROR_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_ERROR_CHECK(cudaMalloc(&d_output, 32 * 16 * 4));
  // Copy memory from host to device
  CUDA_ERROR_CHECK(
      cudaMemcpy(d_input, h_fp16_input.data(), bytes, cudaMemcpyHostToDevice));
  // ldmatrix<<<dimGrid, dimBlock>>>(d_input);
  WMMAFrag<<<dimGrid, dimBlock>>>(d_input, d_output);
  CUDA_ERROR_CHECK(cudaMemcpy(h_frags.data(), d_output, 32 * 16 * 4,
                              cudaMemcpyDeviceToHost));

  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 16; j++) {
      std::cout << h_frags[i * 16 + j] << "  ";
    }
    std::cout << std::endl;
  }
  cudaFree(d_input);
  cudaFree(d_output);
}
