#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <vector>

template <size_t MR, size_t NR, size_t WY, size_t WX>
__global__ void WarpOuterProduct(const float *dA, const float *dB, float *out) {

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

void WarpOuterProduct4x4(const float *a, const float *b, float *c) {
  dim3 gridDim(1, 1);
  dim3 blockDim(32, 1);
  WarpOuterProduct<4, 4, 4, 8><<<gridDim, blockDim>>>(a, b, c);
}
