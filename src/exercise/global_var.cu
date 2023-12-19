#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <stdio.h>

__device__ float global_var;

__global__ void check_global_var() {
  printf("Original value %f\n", global_var);
  // alther the value
  global_var += 2.0;
}

TEST(global_test, test0) {
  // initialize the global value
  float value = 1.0;
  cudaMemcpyToSymbol(global_var, &value, sizeof(float));

  // invoke the kernel
  check_global_var<<<1, 1>>>();

  // dump the global_var to host
  cudaMemcpyFromSymbol(&value, global_var, sizeof(float));

  printf("Altered value %f\n", value);
}