
#include "utils.cuh"
#include "gtest/gtest.h"
#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <stdio.h>
#include <string>

// We want to init variable V1 and V2 and call cxx kernel function f2 and f1
// from host code
const char *codeStr = "                                         \n\
__device__ int V1;                                              \n\
static __global__ void f1(int *result) {                        \n\
   *result = V1 + 10;           \n\
}   \n\
namespace N1 {                                                  \n\
namespace N2 {                                                  \n\
__constant__ int V2;                                            \n\
__global__ void f2(int *result) { *result = V2 + 20; }          \n\
} // namespace N2                                               \n\
} // namespace N1                                               \n\
template <typename T> __global__ void f3(int *result) { *result = sizeof(T); }\n\  \n";

TEST(nvrtc, lowered_names) {
  nvrtcProgram prog;
  NVRTC_RESULT_CHECK(
      nvrtcCreateProgram(&prog, codeStr, "lower_name.cu", 0, NULL, NULL));

  // Add kernel name expressions to NVRTC.
  std::vector<std::string> kernel_name_exprs{"f1", "N1::N2::f2", "f3<int>",
                                             "f3<double>"};
  for (size_t i = 0; i < kernel_name_exprs.size(); ++i) {
    NVRTC_RESULT_CHECK(
        nvrtcAddNameExpression(prog, kernel_name_exprs[i].c_str()));
  }

  // Add __device__ / __constant__ variables name expressions to NVRTC
  std::vector<std::string> variable_name_exprs{"&V1", "&N1::N2::V2"};
  for (size_t i = 0; i < variable_name_exprs.size(); ++i) {
    NVRTC_RESULT_CHECK(
        nvrtcAddNameExpression(prog, variable_name_exprs[i].c_str()));
  }

  // Compile
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                  0,     // numOptions
                                                  NULL); // options

  // Obtain compilation log from the program.
  CHECK_NVRTC_LOG(prog, compileResult);

  // Obtain PTX code from the program.
  size_t ptxSize;
  NVRTC_RESULT_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
  std::string ptx(ptxSize, ' ');
  NVRTC_RESULT_CHECK(nvrtcGetPTX(prog, ptx.data()));

  // Init cuda driver API
  CUdevice cuDevice;
  CUcontext context;
  CUDA_RESULT_CHECK(cuInit(0));
  CUDA_RESULT_CHECK(cuDeviceGet(&cuDevice, 0));
  CUDA_RESULT_CHECK(cuCtxCreate(&context, 0, cuDevice));

  // Load the generated PTX
  CUmodule module;
  CUDA_RESULT_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));

  // For each of the __device__/__constant__ variable address
  // expressions provided to NVRTC, extract the real name of the
  // corresponding variable, and set its value
  std::vector<int> variable_initial_values{100, 200};
  for (size_t i = 0; i < variable_name_exprs.size(); ++i) {
    // Get the real name
    const char *name;
    NVRTC_RESULT_CHECK(
        nvrtcGetLoweredName(prog, variable_name_exprs[i].c_str(), &name));

    // Look up the pointer to the variable by name in module
    CUdeviceptr variable_addr;
    CUDA_RESULT_CHECK(cuModuleGetGlobal(&variable_addr, NULL, module, name));

    // Set initial_value
    int initial_value = variable_initial_values[i];
    CUDA_RESULT_CHECK(
        cuMemcpyHtoD(variable_addr, &initial_value, sizeof(initial_value)));
  }

  CUdeviceptr dResult;
  int hResult = 0;
  CUDA_RESULT_CHECK(cuMemAlloc(&dResult, sizeof(hResult)));
  CUDA_RESULT_CHECK(cuMemcpyHtoD(dResult, &hResult, sizeof(hResult)));

  // For each of the kernel name expressions previously provided to NVRTC,
  // extract the real name of the corresponding __global__ function, and launch
  // it.
  std::vector<int> expected_result = {10 + 100, 20 + 200, sizeof(int),
                                      sizeof(double)};
  for (size_t i = 0; i < kernel_name_exprs.size(); ++i) {
    const char *name;
    NVRTC_RESULT_CHECK(
        nvrtcGetLoweredName(prog, kernel_name_exprs[i].c_str(), &name));

    // Get pointer to kernel from loaded PTX
    CUfunction kernel;
    CUDA_RESULT_CHECK(cuModuleGetFunction(&kernel, module, name));

    // Launch the kernel
    std::cout << "\nlaunching " << name << " (" << kernel_name_exprs[i] << ")"
              << std::endl;
    void *args[] = {&dResult};
    CUDA_RESULT_CHECK(cuLaunchKernel(kernel, 1, 1, 1, // grid dim
                                     1, 1, 1,         // block dim
                                     0, NULL,         // shared mem and stream
                                     args, 0));
    CUDA_RESULT_CHECK(cuCtxSynchronize());

    // Retrieve the result
    CUDA_RESULT_CHECK(cuMemcpyDtoH(&hResult, dResult, sizeof(hResult)));
    // check against expected value
    if (expected_result[i] != hResult) {
      std::cout << "\n Error: expected result = " << expected_result[i]
                << " , actual result = " << hResult << std::endl;
      exit(1);
    }
  }

  // Release resources.
  CUDA_RESULT_CHECK(cuMemFree(dResult));
  CUDA_RESULT_CHECK(cuModuleUnload(module));
  CUDA_RESULT_CHECK(cuCtxDestroy(context));
  // Destroy the program.
  NVRTC_RESULT_CHECK(nvrtcDestroyProgram(&prog));
}
