
#pragma once
#include <cublas_v2.h>
#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <stdio.h>
#define WARP_SIZE 32
#define BANK_ROW_SIZE 128

#define FP4_PTR(pointer) reinterpret_cast<float4 *>(pointer)
#define CONST_FP4_PTR(pointer) reinterpret_cast<const float4 *>(pointer)

#define LD_FP4(pointer) CONST_FP4_PTR(pointer)[0]
#define ST_FP4(pointer, value) FP4_PTR(pointer)[0] = value

#define COPY_FP4(src_ptr, dst_ptr)                                             \
  {                                                                            \
    float4 value = LD_FP4(src_ptr);                                            \
    ST_FP4(dst_ptr, value);                                                    \
  }

#define CUDA_ERROR_CHECK(call)                                                 \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      printf("CUDA Error:\n");                                                 \
      printf("    File:   %s\n", __FILE__);                                    \
      printf("    Line:   %d\n", __LINE__);                                    \
      printf("    Error:  %s\n", cudaGetErrorString(error_code));              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define NVRTC_RESULT_CHECK(call)                                               \
  do {                                                                         \
    nvrtcResult result = call;                                                 \
    if (result != NVRTC_SUCCESS) {                                             \
      printf("NVRTC Error:\n");                                                \
      printf("    File:   %s\n", __FILE__);                                    \
      printf("    Line:   %d\n", __LINE__);                                    \
      printf("    Error:  %s\n", nvrtcGetErrorString(result));                 \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_RESULT_CHECK(call)                                                \
  do {                                                                         \
    CUresult result = call;                                                    \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      printf("NVRTC Error:\n");                                                \
      printf("    File:   %s\n", __FILE__);                                    \
      printf("    Line:   %d\n", __LINE__);                                    \
      printf("    Error:  %s\n", msg);                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_NVRTC_LOG(prog, compileResult)                                   \
  do {                                                                         \
    size_t logSize;                                                            \
    NVRTC_RESULT_CHECK(nvrtcGetProgramLogSize(prog, &logSize));                \
    char *log = new char[logSize];                                             \
    NVRTC_RESULT_CHECK(nvrtcGetProgramLog(prog, log));                         \
    std::cout << log << '\n';                                                  \
    delete[] log;                                                              \
    if (compileResult != NVRTC_SUCCESS) {                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const char *_cuBlasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  }
  return "<unknown>";
}

#define CHECK_NVRTC_LOG(prog, compileResult)                                   \
  do {                                                                         \
    size_t logSize;                                                            \
    NVRTC_RESULT_CHECK(nvrtcGetProgramLogSize(prog, &logSize));                \
    char *log = new char[logSize];                                             \
    NVRTC_RESULT_CHECK(nvrtcGetProgramLog(prog, log));                         \
    std::cout << log << '\n';                                                  \
    delete[] log;                                                              \
    if (compileResult != NVRTC_SUCCESS) {                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_ERROR_CHECK(status)                                             \
  do {                                                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      printf("%s %d CuBlas: %s\n", __FILE__, __LINE__,                         \
             _cuBlasGetErrorEnum(status));                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
