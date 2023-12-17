#include "jitify.hpp"
#include "utils.cuh"
#include "gtest/gtest.h"
#include <cuda.h>
#include <iostream>
#include <string>

template <typename T> void simple() {
  const char *program_source = R"(my_program
template <int N, typename T> __global__ void my_kernel(T *data) {
  T data0 = data[0];
  for (int i = 0; i < N - 1; ++i) {
    data[0] *= data0;
  }
}
)";

  static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(program_source, 0);
  T h_data = 5;
  T *d_data;
  cudaMalloc((void **)&d_data, sizeof(T));
  cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);
  dim3 grid(1);
  dim3 block(1);
  using jitify::reflection::type_of;
  CUDA_RESULT_CHECK(program.kernel("my_kernel")
                        .instantiate(3, type_of(*d_data))
                        .configure(grid, block)
                        .launch(d_data));
  cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  std::cout << h_data << std::endl;
}

TEST(jitify, simple) { simple<int>(); }