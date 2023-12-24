#include "cuda_runtime.h"
#include "gflags/gflags.h"
#include "naive.cuh"
#include "sgemm_v3.cuh"
#include "utils.cuh"
#include "utils.h"
#include "gtest/gtest.h"

// block(0,0,0)thread(2,0,0)

__global__ void __attribute__((noinline))
WarpOuterProduct(const float *__restrict__ dA, const float *__restrict__ dB,
                 float *out) {
  constexpr size_t MR = 1;
  constexpr size_t NR = 1;
  constexpr size_t WY = 4;
  constexpr size_t WX = 8;
  float a_regs[MR];
  float b_regs[NR];
  float c_regs[MR][NR] = {0};
  int tid = threadIdx.x;
  int tid_y = tid / WX;
  int tid_x = tid % WX;

  for (size_t i = 0; i < MR; i++) {
    a_regs[i] = *(dA + tid_y * MR + i);
  }
  for (size_t i = 0; i < NR; i++) {
    b_regs[i] = *(dB + tid_x * NR + i);
  }

  for (size_t i = 0; i < MR; i++) {
    for (size_t j = 0; j < NR; j++) {
      c_regs[i][j] += (a_regs[i] * b_regs[j]);
    }
  }

  for (size_t i = 0; i < MR; i++) {
    for (size_t j = 0; j < NR; j++) {
      out[(tid_y * MR + i) * (NR * WX) + (tid_x * NR + j)] = c_regs[i][j];
    }
  }
}

TEST(warp, test) {
  constexpr size_t MR = 1;
  constexpr size_t NR = 1;
  constexpr size_t WY = 4;
  constexpr size_t WX = 8;
  constexpr size_t ABYTES = MR * WY * sizeof(float) * 2;
  constexpr size_t BBYTES = NR * WX * sizeof(float) * 2;
  constexpr size_t CBYTES = MR * NR * WY * WX * sizeof(float);

  std::vector<float> hA(MR * WY, 1);
  std::vector<float> hB(NR * WX, 2);
  std::vector<float> hC(MR * NR * WY * WX, 0);

  float *dA, *dB, *dC;
  CUDA_ERROR_CHECK(cudaMalloc(&dA, ABYTES));
  CUDA_ERROR_CHECK(cudaMalloc(&dB, BBYTES));
  CUDA_ERROR_CHECK(cudaMalloc(&dC, CBYTES));
  // Copy memory from host to device
  CUDA_ERROR_CHECK(cudaMemcpy(dA, hA.data(), ABYTES, cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMemcpy(dB, hB.data(), BBYTES, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  dim3 gridDim(1, 1, 1);
  dim3 blockDim(32, 1, 1);
  WarpOuterProduct<<<gridDim, blockDim>>>(dA, dB, dC);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  // Copy memory from devie to host
  CUDA_ERROR_CHECK(cudaMemcpy(hC.data(), dC, CBYTES, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  for (int i = 0; i < hC.size(); i++) {
    std::cout << hC[i] << " ";
  }
  std::cout << std::endl;

  // Free device memory
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}