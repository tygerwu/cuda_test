#include "utils.cuh"
#include "gtest/gtest.h"
#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <string>
#define NUM_THREADS 128
#define NUM_BLOCKS 32

const char *saxpy = "\n\
extern \"C\" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n) { \n\
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n\
  if (tid < n) { \n\
    out[tid] = a * x[tid] + y[tid]; \n\
  } \n\
} \n";

const char *saxpy_r = R"(
extern "C" __global__ void saxpy_r(float a, float *x, float *y, float *out,
                                   size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = a * x[tid] + y[tid];
  }
};
)";

TEST(nvrtc, saxpy) {
  // Create an instance of nvrtcProgram
  nvrtcProgram prog;
  NVRTC_RESULT_CHECK(nvrtcCreateProgram(&prog,      // prog
                                        saxpy,      // code string
                                        "saxpy.cu", // name
                                        0,          // numHeaders
                                        NULL,       // headers
                                        NULL        // includeNames
                                        ));

  // Compile the program with fmad disabled.
  // Note: Can specify GPU target architecture explicitly with '-arch' flag.
  const char *opts[] = {"--fmad=false"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                  1,     // numOptions
                                                  opts); // options

  // Obtain compilation log from the program.
  size_t logSize;
  NVRTC_RESULT_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  std::string log(logSize, ' ');
  NVRTC_RESULT_CHECK(nvrtcGetProgramLog(prog, log.data()));
  std::cout << log << '\n';
  if (compileResult != NVRTC_SUCCESS) {
    exit(1);
  }

  // Obtain PTX code from the program.
  size_t ptxSize;
  NVRTC_RESULT_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
  std::string ptx(ptxSize, ' ');
  NVRTC_RESULT_CHECK(nvrtcGetPTX(prog, ptx.data()));
  // std::cout << ptx << '\n';

  // Init cuda driver API
  CUdevice cuDevice;
  CUcontext context;
  CUDA_RESULT_CHECK(cuInit(0));
  CUDA_RESULT_CHECK(cuDeviceGet(&cuDevice, 0));
  CUDA_RESULT_CHECK(cuCtxCreate(&context, 0, cuDevice));

  // Load the generated PTX and get a handle to kernel.
  CUmodule module;
  CUfunction kernel;
  CUDA_RESULT_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
  CUDA_RESULT_CHECK(cuModuleGetFunction(&kernel, module, "saxpy"));

  // Generate input for execution, and create output buffers.
  size_t n = NUM_THREADS * NUM_BLOCKS;
  size_t bufferSize = n * sizeof(float);
  float a = 5.1f;
  float *hX = new float[n], *hY = new float[n], *hOut = new float[n];
  for (size_t i = 0; i < n; ++i) {
    hX[i] = static_cast<float>(i);
    hY[i] = static_cast<float>(i * 2);
  }
  CUdeviceptr dX, dY, dOut;
  CUDA_RESULT_CHECK(cuMemAlloc(&dX, bufferSize));
  CUDA_RESULT_CHECK(cuMemAlloc(&dY, bufferSize));
  CUDA_RESULT_CHECK(cuMemAlloc(&dOut, bufferSize));
  CUDA_RESULT_CHECK(cuMemcpyHtoD(dX, hX, bufferSize));
  CUDA_RESULT_CHECK(cuMemcpyHtoD(dY, hY, bufferSize));

  // Execute SAXPY.
  void *args[] = {&a, &dX, &dY, &dOut, &n};
  CUDA_RESULT_CHECK(cuLaunchKernel(kernel, NUM_BLOCKS, 1, 1, // grid dim
                                   NUM_THREADS, 1, 1,        // block dim
                                   0, NULL, // shared mem and stream
                                   args, 0));

  // arguments
  CUDA_RESULT_CHECK(cuCtxSynchronize());
  // Retrieve and print output.
  CUDA_RESULT_CHECK(cuMemcpyDtoH(hOut, dOut, bufferSize));

  for (size_t i = 0; i < n; ++i) {
    std::cout << a << " * " << hX[i] << " + " << hY[i] << " = " << hOut[i]
              << '\n';
  }

  // Release resources.
  CUDA_RESULT_CHECK(cuMemFree(dX));
  CUDA_RESULT_CHECK(cuMemFree(dY));
  CUDA_RESULT_CHECK(cuMemFree(dOut));
  CUDA_RESULT_CHECK(cuModuleUnload(module));
  CUDA_RESULT_CHECK(cuCtxDestroy(context));
  delete[] hX;
  delete[] hY;
  delete[] hOut;
  // Destroy the program.
  NVRTC_RESULT_CHECK(nvrtcDestroyProgram(&prog));
}