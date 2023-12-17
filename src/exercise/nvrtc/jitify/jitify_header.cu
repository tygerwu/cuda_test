
#include "jitify.hpp"
#include "utils.cuh"
#include "gtest/gtest.h"
#include <cuda.h>
#include <iostream>
#include <string>

static std::istream *file_callback(std::string filename,
                                   std::iostream &tmp_stream) {
  // User returns NULL or pointer to stream containing file source
  // Note: tmp_stream is provided for convenience
  if (filename == "jit_header/my_header4.cuh") {
    tmp_stream << R"(
                    #pragma once
                    template<typename T>
                    T pointless_func(T x) {
                        return x;
                    }
                )";
    return &tmp_stream;
  } else {
    // Find this file through other mechanisms
    return 0;
  }
}

template <typename T> void test_kernels() {
  // Note: The name is specified first, followed by a newline, then the code
  const char *program1 = R"(my_program1
        #include "jit_header/my_header1.cuh"
        #include "jit_header/my_header2.cuh"
        #include "jit_header/my_header3.cuh"
        #include "jit_header/my_header4.cuh"

        __global__ void my_kernel1(float const *indata, float *outdata) {
            outdata[0] = indata[0] + 1;
            outdata[0] -= 1;
        }

        template <int C, typename T>
        __global__ void my_kernel2(float const *indata, float *outdata) {
            for (int i = 0; i < C; ++i) {
                outdata[0] = pointless_func(identity(sqrt(square(negate(indata[0])))));
            }
        }
    )";

  using jitify::reflection::instance_of;
  using jitify::reflection::NonType;
  using jitify::reflection::reflect;
  using jitify::reflection::Type;
  using jitify::reflection::type_of;

  thread_local static jitify::JitCache kernel_cache;
  const char *const example_headers_my_header1_cuh =
      R"(jit_header/my_header1.cuh
        #pragma once
        template <typename T> __device__ inline T square(T x) { return x * x; }
    )";
  jitify::Program program = kernel_cache.program(
      program1,                         // CodeString
      {example_headers_my_header1_cuh}, // Load header from source
      {"--use_fast_math", "-I/media/tyger/linux_ssd/codes/cxx_test/cuda_lab/"
                          "src/exercise/nvrtc/jitify"},
      file_callback // Load header from callback
  );

  T *indata;
  T *outdata;
  cudaMalloc((void **)&indata, sizeof(T));
  cudaMalloc((void **)&outdata, sizeof(T));
  T inval = 3.14159f;
  cudaMemcpy(indata, &inval, sizeof(T), cudaMemcpyHostToDevice);

  dim3 grid(1);
  dim3 block(1);
  CUDA_RESULT_CHECK(program.kernel("my_kernel1")
                        .instantiate()
                        .configure(grid, block)
                        .launch(indata, outdata));
  enum { C = 123 };
  // These invocations are all equivalent and will come from cache after the 1st
  CUDA_RESULT_CHECK((program.kernel("my_kernel2")
                         .instantiate<NonType<int, C>, T>()
                         .configure(grid, block)
                         .launch(indata, outdata)));
  CUDA_RESULT_CHECK(program.kernel("my_kernel2")
                        .instantiate({reflect((int)C), reflect<T>()})
                        .configure(grid, block)
                        .launch(indata, outdata));
  // Recommended versions
  CUDA_RESULT_CHECK(program.kernel("my_kernel2")
                        .instantiate((int)C, Type<T>())
                        .configure(grid, block)
                        .launch(indata, outdata));
  CUDA_RESULT_CHECK(program.kernel("my_kernel2")
                        .instantiate((int)C, type_of(*indata))
                        .configure(grid, block)
                        .launch(indata, outdata));
  CUDA_RESULT_CHECK(program.kernel("my_kernel2")
                        .instantiate((int)C, instance_of(*indata))
                        .configure(grid, block)
                        .launch(indata, outdata));

  T outval = 0;
  cudaMemcpy(&outval, outdata, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(outdata);
  cudaFree(indata);

  std::cout << inval << " -> " << outval << std::endl;
}

TEST(jitify, header) { test_kernels<int>(); }