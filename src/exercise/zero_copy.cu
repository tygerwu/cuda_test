
#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
__global__ void sum(float *A, float *B, float *C, const int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

void init_data(float *data, int len, float x) {
  for (int i = 0; i < len; i++) {
    data[i] = x;
  }
}

TEST(zero_copy, test0) {
  constexpr int len = 32;
  int bytes = len * sizeof(float);

  std::vector<float> hC(len, 0);

  float *h_A, *h_B;
  // allocate global device memory
  float *d_C;
  cudaMalloc((float **)&d_C, bytes);

  // allocate zerocpy host memory
  unsigned int flags = cudaHostAllocMapped;
  cudaHostAlloc((void **)&h_A, bytes, flags);
  cudaHostAlloc((void **)&h_B, bytes, flags);

  // initialize data at host side
  init_data(h_A, len, 1);
  init_data(h_B, len, 2);

  float *d_A, *d_B;
  // pass the pointer to device
  cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0);
  cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0);

  // execute kernel with zero copy memory
  sum<<<1, len>>>(d_A, d_B, d_C, len);

  // copy kernel result back to host side
  cudaMemcpy(hC.data(), d_C, bytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < len; i++) {
    std::cout << hC[i] << " ";
  }
  std::cout << std::endl;

  // free memory
  cudaFree(d_C);
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
}