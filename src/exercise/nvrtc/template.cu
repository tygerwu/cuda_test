
#include "utils.cuh"
#include "gtest/gtest.h"
#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <stdio.h>
#include <string>

static const char *codeStr = R"(
namespace N1 {
  struct S1_t {
    int i;
    double d;
  };
  __global__ void f3(int *result) { *result = sizeof(T); 
};
)";

// This structure is also defined in GPU code string. Should ideally
// be in a header file included by both GPU code string and by CPU code.
namespace N1 {
struct S1_t {
  int i;
  double d;
};
}; // namespace N1

template <typename T> std::string GetTypeName(void) {

  // Look up the source level name string for the type "T" using
  // nvrtcGetTypeName()
  std::string type_name;
  NVRTC_RESULT_CHECK(nvrtcGetTypeName<T>(&type_name));
  return type_name;
}

TEST(nvrtc, template) {
  std::cout << GetTypeName<int>() << std::endl;
  std::cout << GetTypeName<N1::S1_t>() << std::endl;
}
