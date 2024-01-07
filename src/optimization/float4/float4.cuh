#pragma once
#include "cuda_runtime.h"
#include "utils.cuh"
#include "utils.h"
static __global__ void LD1Impl(const float *__restrict__ input, float *output,
                               int len) {
  float x;
  float y = 0;
  int idx = threadIdx.x;
  int thread_num = blockDim.x;
#pragma unroll(1)
  for (size_t i = 0; i < len; i += thread_num) {
    const auto *ptr = input + i + idx;
    __asm__ __volatile__("ld.global.f32 %0,[%1];" : "=f"(x) : "l"(ptr));
    y += x;
  }
  output[idx] = y;
}

static __global__ void LD4Impl(const float *__restrict__ input, float *output,
                               int len) {
  float x0, x1, x2, x3;
  float y0 = 0, y1 = 0, y2 = 0, y3 = 0;
  int idx = threadIdx.x;
  int thread_num = blockDim.x;
#pragma unroll(1)
  for (size_t i = 0; i < len; i += thread_num * 4) {
    const auto *ptr = input + i + idx * 4;
    __asm__ __volatile__("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
                         : "=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3)
                         : "l"(ptr));
    y0 += x0;
    y1 += x1;
    y2 += x2;
    y3 += x3;
  }

  output[idx] = (y0 + y1 + y2 + y3);
}

static __global__ void LD4SMemToRegImpl(const float *__restrict__ input,
                                        float *output, int len) {
  float x0 = 0, x1 = 0, x2 = 0, x3 = 0;
  float y0 = 0, y1 = 0, y2 = 0, y3 = 0;
  int tid = threadIdx.x;
  int thread_num = blockDim.x;

  extern __shared__ float smem_buffer[];
  auto *smem_ptr = smem_buffer + tid * 4;

#pragma unroll(1)
  for (size_t i = 0; i < len; i += thread_num * 4) {
    const auto *ptr = input + i + tid * 4;
    auto smem_value = *FP4_PTR(smem_ptr);
    auto gmem_value = *CONST_FP4_PTR(ptr);
    smem_value.x += gmem_value.x;
    smem_value.y += gmem_value.y;
    smem_value.z += gmem_value.z;
    smem_value.w += gmem_value.w;

    *FP4_PTR(smem_ptr) = smem_value;
  }
  __syncthreads();

#pragma unroll(1)
  for (size_t i = 0; i < len; i += thread_num * 4) {
    // Illegal
    // __asm__ __volatile__("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
    //                      : "=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3)
    //                      : "l"(smem_ptr));

    x0 = smem_ptr[0];
    x1 = smem_ptr[1];
    x2 = smem_ptr[2];
    x3 = smem_ptr[3];

    y0 += x0;
    y1 += x1;
    y2 += x2;
    y3 += x3;
  }

  output[tid] = (y0 + y1 + y2 + y3);
}

static __global__ void LD1SMemToRegImpl(const float *__restrict__ input,
                                        float *output, int len) {
  float x0 = 0, x1 = 0;
  float y0 = 0, y1 = 0;
  int tid = threadIdx.x;
  int thread_num = blockDim.x;

  // Size: threadNum * 4
  extern __shared__ float smem_buffer[];
  auto *smem_ptr = smem_buffer + tid * 4;

#pragma unroll(1)
  for (size_t i = 0; i < len; i += thread_num * 4) {
    const auto *ptr = input + i + tid * 4;
    auto smem_value = *FP4_PTR(smem_ptr);
    auto gmem_value = *CONST_FP4_PTR(ptr);
    smem_value.x += gmem_value.x;
    smem_value.y += gmem_value.y;
    smem_value.z += gmem_value.z;
    smem_value.w += gmem_value.w;

    *FP4_PTR(smem_ptr) = smem_value;
  }
  __syncthreads();

#pragma unroll(1)
  for (size_t i = 0; i < len * 2; i += thread_num * 4) {

    x0 = smem_ptr[0];
    x1 = smem_ptr[1];

    y0 += x0;
    y1 += x1;
  }

  output[tid] = (y0 + y1);
}

static void LD1(const float *__restrict__ input, float *output, int len) {
  dim3 block_dim(WARP_SIZE * 16, 1);
  dim3 grid_dim(1, 1);
  LD1Impl<<<grid_dim, block_dim>>>(input, output, len);
}

static void LD4(const float *__restrict__ input, float *output, int len) {
  dim3 block_dim(WARP_SIZE * 16, 1);
  dim3 grid_dim(1, 1);
  LD4Impl<<<grid_dim, block_dim>>>(input, output, len);
}
static void LD4SMemToReg(const float *__restrict__ input, float *output,
                         int len) {
  const int THREAD_NUM = WARP_SIZE * 16;
  dim3 block_dim(THREAD_NUM, 1);
  dim3 grid_dim(1, 1);
  LD4SMemToRegImpl<<<grid_dim, block_dim, THREAD_NUM * sizeof(float) * 4>>>(
      input, output, len);
}
static void LD1SMemToReg(const float *__restrict__ input, float *output,
                         int len) {
  const int THREAD_NUM = WARP_SIZE * 16;
  dim3 block_dim(THREAD_NUM, 1);
  dim3 grid_dim(1, 1);
  LD1SMemToRegImpl<<<grid_dim, block_dim, THREAD_NUM * sizeof(float) * 4>>>(
      input, output, len);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}