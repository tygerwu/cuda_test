
#include "utils.cuh"
#include "gtest/gtest.h"
#include <cuda.h>
#include <iostream>
#include <limits.h>
#include <nvrtc.h>
#include <stdio.h>
#include <string>

const char *codeStrs = R"(
#include "headerA.h"
#include "headerB.h"
__global__ void print_const(){
  printf("myVarA : %d \n",myVarA); 
  printf("myVarB : %d \n",myVarB);
})";
const char *headerA = R"(__constant__ double myVarA;)";

TEST(nvrtc, header) {

  nvrtcProgram prog;
  const char *headerSources[] = {headerA};
  const char *headerNames[] = {"headerA.h"};
  NVRTC_RESULT_CHECK(nvrtcCreateProgram(&prog, codeStrs, "header_test.cu", 1,
                                        headerSources, headerNames));

  // Add kernel name expressions to NVRTC.
  std::vector<std::string> kernel_name_exprs{"print_const"};
  for (size_t i = 0; i < kernel_name_exprs.size(); ++i) {
    NVRTC_RESULT_CHECK(
        nvrtcAddNameExpression(prog, kernel_name_exprs[i].c_str()));
  }

  // Add __device__ / __constant__ variables name expressions to NVRTC
  std::vector<std::string> variable_name_exprs{"&myVarA", "&myVarB"};
  for (size_t i = 0; i < variable_name_exprs.size(); ++i) {
    NVRTC_RESULT_CHECK(
        nvrtcAddNameExpression(prog, variable_name_exprs[i].c_str()));
  }

  // Compile
  const char *options[] = {
      "-I /media/tyger/linux_ssd/codes/cxx_test/cuda_lab/src/exercise/nvrtc/"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,     // prog
                                                  1,        // numOptions
                                                  options); // options

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
  // expressions provided to NVRTC, extract the mangled name of the
  // corresponding variable, and set its value
  std::vector<int> variable_initial_values{100, 200};
  for (size_t i = 0; i < variable_name_exprs.size(); ++i) {
    // Get the mangled name
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

  // For each of the kernel name expressions previously provided to NVRTC,
  // extract the mangled name of the corresponding __global__ function, and
  // launch it.
  for (size_t i = 0; i < kernel_name_exprs.size(); ++i) {
    const char *name;
    NVRTC_RESULT_CHECK(
        nvrtcGetLoweredName(prog, kernel_name_exprs[i].c_str(), &name));

    // Get pointer to kernel from loaded PTX
    CUfunction kernel;
    CUDA_RESULT_CHECK(cuModuleGetFunction(&kernel, module, name));

    CUDA_RESULT_CHECK(cuLaunchKernel(kernel, 1, 1, 1, // grid dim
                                     1, 1, 1,         // block dim
                                     0, NULL,         // shared mem and stream
                                     NULL, 0));
    CUDA_RESULT_CHECK(cuCtxSynchronize());
  }

  // Release resources.
  CUDA_RESULT_CHECK(cuModuleUnload(module));
  CUDA_RESULT_CHECK(cuCtxDestroy(context));
  // Destroy the program.
  NVRTC_RESULT_CHECK(nvrtcDestroyProgram(&prog));
}
