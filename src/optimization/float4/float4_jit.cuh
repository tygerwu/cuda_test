#include "cuda_runtime.h"
#include "jitify.hpp"

// SASS is optimized as well ....
static void LD4Jit(const float *__restrict__ input, float *output, int len) {
  const char *program_source = R"(my_program
__global__ void LD4Impl(const float *__restrict__ input, float *output,
                        int len) {
   float x0, x1, x2, x3;
  float y0 = 0, y1 = 0, y2 = 0, y3 = 0;
  int idx = threadIdx.x;
  int thread_num = blockDim.x;
#pragma unroll(1)
  for (size_t i = 0; i < len; i += thread_num * 4) {
    const auto *ptr = input + i + idx * 4;
    asm __volatile__("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
                         : "=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3)
                         : "l"(ptr));
    y0 += x0;
    y1 += x1;
    y2 += x2;
    y3 += x3;
  }

  output[idx] = (y0 + y1 + y2 + y3);
}
)";
  static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(program_source, 0);
  dim3 block(WARP_SIZE * 16, 1);
  dim3 grid(1, 1);
  CUDA_RESULT_CHECK(program.kernel("LD4Impl")
                        .instantiate()
                        .configure(grid, block)
                        .launch(input, output, len));
}
