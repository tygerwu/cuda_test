#include "cuda_runtime.h"
#include "gflags/gflags.h"
#include "naive.cuh"
#include "sgemm.cuh"
#include "sgemm_cublas.cuh"
#include "sgemm_v3.cuh"
#include "utils.cuh"
#include "utils.h"
#include "gtest/gtest.h"

using FloatVector = std::vector<float>;

using SGemmFunc = std::function<void(const float *A, const float *B, float *C,
                                     int M, int N, int K)>;

class CUSgemmBench : public ::testing::Test {

public:
  void Verify(SGemmFunc func) {
    loops = 1;
    Bench(func, true);
  }
  double Bench(SGemmFunc func, bool verify = false) {
    int ASIZE = m * k, BSIZE = k * n, CSIZE = m * n;
    int ABYTES = ASIZE * sizeof(float);
    int BBYTES = BSIZE * sizeof(float);
    int CBYTES = CSIZE * sizeof(float);

    std::vector<float> times;
    for (int i = 0; i < loops; i++) {
      // Allocate host memory
      FloatVector hA = CreateData<float>(ASIZE, 0, 4);
      FloatVector hB = CreateData<float>(BSIZE, 0, 4);
      FloatVector hC(CSIZE, 0);

      // Allocate device memory
      float *dA, *dB, *dC;
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

      checkCudaErrors(cudaEventRecord(start));
      func(dA, dB, dC, m, n, k);
      CUDA_ERROR_CHECK(cudaEventRecord(stop));
      CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
      CUDA_ERROR_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

      // Copy memory from devie to host
      CUDA_ERROR_CHECK(
          cudaMemcpy(hC.data(), dC, CBYTES, cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();

      if (verify) {
        FloatVector groundTruth(CSIZE, 0);
        RawMatmul(hA.data(), hB.data(), groundTruth.data(), m, n, k);
        FloatsCompare(hC.data(), groundTruth.data(), m * n);
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

class CUGemmMNK : public ::CUSgemmBench {
public:
  void Bench(SGemmFunc func) {
    table.SetHeads({"MNK", "GFlops"});
    for (int i = start; i < end; i += stride) {
      this->m = i;
      this->n = i;
      this->k = i;
      double glops = CUSgemmBench::Bench(func);
      table.AddRow({(float)i, (float)(glops)});
    }
    table.Print();
    table.ExportToCSV(
        "/media/tyger/linux_ssd/codes/cxx_test/cuda_lab/data/sgemm.csv");
  }

public:
  int start = 256, stride = 256, end = 4352;

private:
  Table2D table;
};

TEST_F(CUSgemmBench, v3) {
  m = 512, n = 512, k = 512;
  constexpr int BLOCK_SIZE_M = 128;
  constexpr int BLOCK_SIZE_K = 8;
  constexpr int BLOCK_SIZE_N = 128;
  constexpr int THREAD_SIZE_X = 8;
  constexpr int THREAD_SIZE_Y = 8;
  constexpr bool ENABLE_DOUBLE_BUFFER = false;

  Bench(SGemmV3<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X,
                THREAD_SIZE_Y, ENABLE_DOUBLE_BUFFER>);
}

TEST_F(CUSgemmBench, v0) {
  m = 256, n = 256, k = 16;

  constexpr int WY = 4;
  constexpr int WX = 8;

  constexpr int NR = 8;
  constexpr int MR = 8;

  constexpr int BX = WX * 2;
  constexpr int BY = WY * 4;

  constexpr int MC = MR * BY;
  constexpr int NC = NR * BX;

  constexpr int KC = 16;

  constexpr int BLOCK_SIZE_M = 128;
  constexpr int BLOCK_SIZE_K = 8;
  constexpr int BLOCK_SIZE_N = 128;
  constexpr int THREAD_SIZE_X = 8;
  constexpr int THREAD_SIZE_Y = 8;
  constexpr bool ENABLE_DOUBLE_BUFFER = false;

  Verify(SGemmV3<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X,
                 THREAD_SIZE_Y, ENABLE_DOUBLE_BUFFER>);

  Verify(CudaSGemm<MC, KC, NC, MR, NR, WY, WX>);
  // Bench(CublasSgemm);
}

TEST_F(CUGemmMNK, naive) { Bench(CublasSgemm); }