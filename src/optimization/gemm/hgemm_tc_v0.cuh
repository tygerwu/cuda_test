
#pragma once
#include "utils.cuh"
#include "utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

template <size_t MC, size_t KC, size_t NC, size_t MR, size_t NR, size_t WY,
          size_t WX>
static __global__ void CudaHGemmTCV0Impl(const half *__restrict__ A,
                                         const half *__restrict__ B,
                                         half *__restrict__ C, int M, int N,
                                         int K, int ldk, int ldn) {}

template <size_t MC, size_t KC, size_t NC, size_t MR, size_t NR, size_t WY,
          size_t WX>
static void CudaHGemmTCV0(const half *A, const half *B, half *C, int M, int N,
                          int K) {
  dim3 dimBlock(NC / NR, MC / MR);
  dim3 dimGrid(N / NC, M / MC);
  CudaHGemmTCV0Impl<MC, KC, NC, MR, NR, WY, WX>
      <<<dimGrid, dimBlock>>>(A, B, C, M, N, K, K, N);
}