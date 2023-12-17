#include "cuda_runtime.h"
#include "float4.cuh"
#include "float4_jit.cuh"
#include "gflags/gflags.h"
#include "utils.cuh"
#include "utils.h"
#include "gtest/gtest.h"

using FloatVector = std::vector<float>;

using MemcpyFunc =
    std::function<void(const float *input, float *output, int len)>;

class Float4Bench : public ::testing::Test {

public:
  void BenchLD4JIT() { Bench(LD4Jit); }
  void BenchLD4() { Bench(LD4); }
  void BenchLD4SMemToReg() { Bench(LD4SMemToReg); }
  void BenchLD1SMemToReg() { Bench(LD1SMemToReg); }

private:
  void Bench(MemcpyFunc func) {

    std::vector<float> times;
    for (int i = 0; i < loops; i++) {
      // Allocate host memory
      FloatVector h_input(len, 1);
      FloatVector h_output(len, 0);
      size_t bytes = len * sizeof(float);

      // Allocate device memory
      float *d_input, *d_output;
      CUDA_ERROR_CHECK(cudaMalloc(&d_input, bytes));
      CUDA_ERROR_CHECK(cudaMalloc(&d_output, bytes));
      // Copy memory from host to device
      CUDA_ERROR_CHECK(
          cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

      float time_ms = 0;
      cudaEvent_t start, stop;
      CUDA_ERROR_CHECK(cudaEventCreate(&start));
      CUDA_ERROR_CHECK(cudaEventCreate(&stop));

      CUDA_ERROR_CHECK(cudaEventRecord(start));
      func(d_input, d_output, len);
      CUDA_ERROR_CHECK(cudaEventRecord(stop));
      CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
      CUDA_ERROR_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

      // Copy memory from devie to host
      CUDA_ERROR_CHECK(
          cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

      //   for (int i = 0; i < 32; i++) {
      //     std::cout << h_output[i] << " ";
      //   }
      // std::cout << std::endl;

      // Free device memory
      cudaFree(d_input);
      cudaFree(d_output);

      times.push_back(time_ms);
    }

    std::cout << "Average Time: " << Average(times) << std::endl;
  }

public:
  int len;
  int loops = 5;
};

TEST_F(Float4Bench, test) {
  len = 256 * 1024 * 2;
  //   BenchLD4();
  BenchLD4SMemToReg();
  BenchLD1SMemToReg();
}
