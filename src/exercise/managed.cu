
#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
static __global__ void sum(float *A, float *B, float *C, const int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

static void init_data(float *data, int len, float x) {
  for (int i = 0; i < len; i++) {
    data[i] = x;
  }
}

TEST(manage, test0) {
  constexpr int len = 32;
  int bytes = len * sizeof(float);

  // allocate managed memory
  float *A, *B, *C;
  cudaMallocManaged((void **)&A, bytes);
  cudaMallocManaged((void **)&B, bytes);
  cudaMallocManaged((void **)&C, bytes);

  init_data(A, len, 1);
  init_data(B, len, 2);

  sum<<<1, len>>>(A, B, C, len);
  cudaDeviceSynchronize();

  for (int i = 0; i < len; i++) {
    std::cout << C[i] << " ";
  }
  std::cout << std::endl;

  // free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}