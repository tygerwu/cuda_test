#include "cuda_runtime.h"
#include "gflags/gflags.h"
#include "hgemm_tc_v0.cuh"
#include "naive.cuh"
#include "utils.cuh"
#include "utils.h"
#include "gtest/gtest.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

using HGemmFunc = std::function<void(const half *A, const half *B, half *C,
                                     int M, int N, int K)>;

template <cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP>
static void CublasF16F16Gemm(const half *a, const half *b, half *c, int M,
                             int N, int K) {

  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha = 1.0;
  half beta = 0.0;
  CUBLAS_ERROR_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, CUDA_R_16F, N, a,
      CUDA_R_16F, K, &beta, c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, algo));
  cublasDestroy(handle);
}

class CUHGemmBench : public ::testing::Test {

public:
  void Verify(HGemmFunc func) {
    loops = 1;
    BenchFunc(func, true);
  }
  void BenchBlas() { BenchFunc(nullptr, false, true); }
  double Bench(HGemmFunc func) { return BenchFunc(func, false, false); }

protected:
  double BenchFunc(HGemmFunc func, bool verify = false, bool blas = false) {
    int ASIZE = m * k, BSIZE = k * n, CSIZE = m * n;
    int ABYTES = ASIZE * sizeof(half);
    int BBYTES = BSIZE * sizeof(half);
    int CBYTES = CSIZE * sizeof(half);

    std::vector<float> times;
    for (int i = 0; i < loops; i++) {
      // Allocate fp32 host memory
      FloatVector fp32_hA = CreateData<float>(ASIZE, 0, 6);
      FloatVector fp32_hB = CreateData<float>(BSIZE, 0, 6);
      FloatVector fp32_hC(CSIZE, 0);

      // Fp16 host memory
      std::vector<half> hA = Convert<float, half>(fp32_hA);
      std::vector<half> hB = Convert<float, half>(fp32_hB);
      std::vector<half> hC(CSIZE, 0);

      // Allocate device memory
      half *dA, *dB, *dC;
      CUDA_ERROR_CHECK(cudaMalloc(&dA, ABYTES));
      CUDA_ERROR_CHECK(cudaMalloc(&dB, BBYTES));
      CUDA_ERROR_CHECK(cudaMalloc(&dC, CBYTES));

      // Copy memory from host to device
      CUDA_ERROR_CHECK(
          cudaMemcpy(dA, hA.data(), ABYTES, cudaMemcpyHostToDevice));
      CUDA_ERROR_CHECK(
          cudaMemcpy(dB, hB.data(), BBYTES, cudaMemcpyHostToDevice));

      float time_ms = 0;
      cudaEvent_t start, stop;
      CUDA_ERROR_CHECK(cudaEventCreate(&start));
      CUDA_ERROR_CHECK(cudaEventCreate(&stop));
      CUDA_ERROR_CHECK(cudaEventRecord(start));

      func(dA, dB, dC, m, n, k);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
      }

      CUDA_ERROR_CHECK(cudaEventRecord(stop));
      CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
      CUDA_ERROR_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

      // Copy memory from devie to host
      CUDA_ERROR_CHECK(
          cudaMemcpy(hC.data(), dC, CBYTES, cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();

      // Fp16 to Fp 32
      fp32_hC = Convert<half, float>(hC);

      if (verify) {
        FloatVector groundTruth(CSIZE, 0);
        RawMatmul(fp32_hA.data(), fp32_hB.data(), groundTruth.data(), m, n, k);
        FloatsCompare(fp32_hC.data(), groundTruth.data(), m * n);
      }

      // Free device memory
      cudaFree(dA);
      cudaFree(dB);
      cudaFree(dC);

      times.push_back(time_ms);
    }
    double gflops = (double)2 * m * n * k / (Average(times) * 1e6);
    std::cout << "Average Time: " << Average(times) << " gflops:" << gflops
              << std::endl;
    return gflops;
  }
  int m, n, k;
  int loops = 12;
};
