#include "utils.cuh"
#include "utils.h"
#include "gtest/gtest.h"
#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <string>

TEST(nvdriver, vector_add) {
  // Init cuda driver API
  CUdevice cuDevice;
  CUcontext context;
  CUDA_RESULT_CHECK(cuInit(0));
  CUDA_RESULT_CHECK(cuDeviceGet(&cuDevice, 0));
  CUDA_RESULT_CHECK(cuCtxCreate(&context, 0, cuDevice));

  // Load the generated PTX and get a handle to kernel.
  CUmodule module;
  CUfunction kernel;
  CUDA_RESULT_CHECK(
      cuModuleLoad(&module, "/media/tyger/linux_ssd/codes/cxx_test/cuda_lab/"
                            "src/exercise/driver/vector_add_kernel.ptx"));
  CUDA_RESULT_CHECK(cuModuleGetFunction(&kernel, module, "vecadd_kernel"));

  // Generate input for execution, and create output buffers.
  int N = 5000;
  int bytes = N * sizeof(float);
  auto hA = CreateFloats(N, -10, 10);
  auto hB = CreateFloats(N, 10, -10, -0.5);
  std::vector<float> hC(N);

  // Allocate vectors in device memory
  CUdeviceptr dA;
  CUdeviceptr dB;
  CUdeviceptr dC;
  CUDA_RESULT_CHECK(cuMemAlloc(&dA, bytes));
  CUDA_RESULT_CHECK(cuMemAlloc(&dB, bytes));
  CUDA_RESULT_CHECK(cuMemAlloc(&dC, bytes));

  // Copy data from host memory to device memory
  CUDA_RESULT_CHECK(cuMemcpyHtoD(dA, hA.data(), bytes));
  CUDA_RESULT_CHECK(cuMemcpyHtoD(dB, hB.data(), bytes));

  if (1) {
    // Grid/Block configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    void *args[] = {&dA, &dB, &dC, &N};

    // Launch the CUDA kernel
    CUDA_RESULT_CHECK(cuLaunchKernel(kernel, blocksPerGrid, 1, 1,
                                     threadsPerBlock, 1, 1, 0, NULL, args,
                                     NULL));
  } else {
    // This is the new CUDA 4.0 API for Kernel Parameter Passing and Kernel
    // Launch (advanced method)
    int offset = 0;
    void *argBuffer[16];
    *((CUdeviceptr *)&argBuffer[offset]) = dA;
    offset += sizeof(dA);
    *((CUdeviceptr *)&argBuffer[offset]) = dB;
    offset += sizeof(dB);
    *((CUdeviceptr *)&argBuffer[offset]) = dC;
    offset += sizeof(dC);
    *((int *)&argBuffer[offset]) = N;
    offset += sizeof(N);

    // Grid/Block configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    CUDA_RESULT_CHECK(cuLaunchKernel(kernel, blocksPerGrid, 1, 1,
                                     threadsPerBlock, 1, 1, 0, NULL, NULL,
                                     argBuffer));
  }
  // Copy result from device memory to host memory
  CUDA_RESULT_CHECK(cuMemcpyDtoH(hC.data(), dC, bytes));

  PrintVector(hC);

  // Release resource
  CUDA_RESULT_CHECK(cuMemFree(dA));
  CUDA_RESULT_CHECK(cuMemFree(dB));
  CUDA_RESULT_CHECK(cuMemFree(dC));
  CUDA_RESULT_CHECK(cuModuleUnload(module));
  CUDA_RESULT_CHECK(cuCtxDestroy(context));
}
