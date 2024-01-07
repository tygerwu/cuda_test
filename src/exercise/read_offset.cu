#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

__global__ void read_offset_unroll4(float *A, float *B, float *C, const int n,
                                    int offset) {
  unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
}