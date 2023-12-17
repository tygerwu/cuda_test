
#include "jitify.hpp"
#include "utils.cuh"
#include "gtest/gtest.h"
#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <string>

TEST(jitify, consts) {
  using jitify::reflection::Type;
  thread_local static jitify::JitCache kernel_cache;

  constexpr int n_const = 3;
  int *outdata;
  cudaMalloc((void **)&outdata, n_const * sizeof(int));

  bool test = true;

  dim3 grid(1);
  dim3 block(1);
  { // test __constant__ look up in kernel string using diffrent namespaces
    const char *const_program = R"(const_program
    #pragma once

    __constant__ int a;
    namespace b { __constant__ int a; }
    namespace c { namespace b { __constant__ int a; } }

    __global__ void constant_test(int *x) {
      x[0] = a;
      x[1] = b::a;
      x[2] = c::b::a;
    }
    )";
    jitify::Program program =
        kernel_cache.program(const_program, 0, {"--use_fast_math"});
    auto instance = program.kernel("constant_test").instantiate();
    int inval[] = {2, 4, 8};
    cuMemcpyHtoD(instance.get_constant_ptr("a"), &inval[0], sizeof(int));
    cuMemcpyHtoD(instance.get_constant_ptr("b::a"), &inval[1], sizeof(int));
    cuMemcpyHtoD(instance.get_constant_ptr("c::b::a"), &inval[2], sizeof(int));
    CUDA_RESULT_CHECK(instance.configure(grid, block).launch(outdata));
    cudaDeviceSynchronize();
    int outval[n_const];
    cudaMemcpy(outval, outdata, sizeof(outval), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_const; i++)
      if (inval[i] != outval[i])
        test = false;
  }
}
