#include "cuda_runtime.h"
#include "gflags/gflags.h"
#include "utils.cuh"
#include "utils.h"
#include "gtest/gtest.h"

__global__ void CopyImpl(const float *__restrict__ input, float *output) {
  int offset = threadIdx.x + blockIdx.x * blockDim.x;
  output[offset] = input[offset];
}

__global__ void Copy4Impl(const float *__restrict__ input, float *output) {
  int offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
  *FP4_PTR((output + offset)) = *CONST_FP4_PTR(input + offset);
}

__global__ void OneWarpCopyImpl(const float *__restrict__ input, float *output,
                                int len) {
  int offset = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < len; i += 32) {
    float x = input[i + offset];
  }
}
__global__ void OneWarpCopy4Impl(const float *__restrict__ input, float *output,
                                 int len) {
  int offset = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < len; i += 128) {
    float4 x = *CONST_FP4_PTR(input + i + offset * 4);
  }
}

__global__ void LD1Impl(const float *__restrict__ input, float *output,
                        int len) {
  float x;
  float y = 0;
  int idx = threadIdx.x;
  int thread_num = blockDim.x;
#pragma unroll(1)
  for (size_t i = 0; i < len; i += thread_num) {
    const auto *ptr = input + i + idx;
    __asm__ __volatile__("ld.global.f32 %0,[%1];" : "=f"(x) : "l"(ptr));
    y += x;
  }
  output[idx] = y;
}

__global__ void LD4Impl(const float *__restrict__ input, float *output,
                        int len) {
  float x0, x1, x2, x3;
  float y0 = 0, y1 = 0, y2 = 0, y3 = 0;
  int idx = threadIdx.x;
  int thread_num = blockDim.x;
#pragma unroll(1)
  for (size_t i = 0; i < len; i += thread_num * 4) {
    const auto *ptr = input + i + idx * 4;
    __asm__ __volatile__("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
                         : "=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3)
                         : "l"(ptr));
    y0 += x0;
    y1 += x1;
    y2 += x2;
    y3 += x3;
  }

  output[idx] = (y0 + y1 + y2 + y3);
}

void Copy(const float *__restrict__ input, float *output, int len) {
  dim3 block_dim(256, 1);
  dim3 grid_dim(len / 256, 1);
  CopyImpl<<<grid_dim, block_dim>>>(input, output);
}

void Copy4(const float *__restrict__ input, float *output, int len) {
  dim3 block_dim(256, 1);
  dim3 grid_dim(len / (256 * 4), 1);
  Copy4Impl<<<grid_dim, block_dim>>>(input, output);
}

void OneWarpCopy(const float *__restrict__ input, float *output, int len) {
  dim3 block_dim(1, 1);
  dim3 grid_dim(1, 1);
  OneWarpCopyImpl<<<grid_dim, block_dim>>>(input, output, len);
}
void OneWarpCopy4(const float *__restrict__ input, float *output, int len) {
  dim3 block_dim(1, 1);
  dim3 grid_dim(1, 1);
  OneWarpCopy4Impl<<<grid_dim, block_dim>>>(input, output, len);
}

void LD1(const float *__restrict__ input, float *output, int len) {
  dim3 block_dim(WARP_SIZE * 16, 1);
  dim3 grid_dim(1, 1);
  LD1Impl<<<grid_dim, block_dim>>>(input, output, len);
}

void LD4(const float *__restrict__ input, float *output, int len) {
  dim3 block_dim(WARP_SIZE * 16, 1);
  dim3 grid_dim(1, 1);
  LD4Impl<<<grid_dim, block_dim>>>(input, output, len);
}

using FloatVector = std::vector<float>;

using MemcpyFunc =
    std::function<void(const float *input, float *output, int len)>;

class CUMemcpyBench : public ::testing::Test {

public:
  void Bench1() { Bench(Copy); }
  void Bench4() { Bench(Copy4); }
  void BenchOneWarpCopy() { Bench(OneWarpCopy); }
  void BenchOneWarpCopy4() { Bench(OneWarpCopy4); }
  void BenchLD1() { Bench(LD1); }
  void BenchLD4() { Bench(LD4); }

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

      // for (int i = 0; i < 32; i++) {
      //   std::cout << h_output[i] << " ";
      // }
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

TEST_F(CUMemcpyBench, test) {
  len = 256 * 1024 * 2;
  BenchLD1();
  BenchLD4();
}
