// #include "gtest/gtest.h"
// #include <cuda_runtime.h>
// #include <stdio.h>
// #include <vector>

// struct ConstantTrait {
//   static const int value = 4;
// };
// template <int X, typename trait> __device__ void foo(int *p1, int *p2) {
// // no argument specified, loop will be completely unrolled
// #pragma unroll
//   for (int i = 0; i < 12; ++i)
//     p1[i] += p2[i] * 2;

// // unroll value = X+1
// #pragma unroll(X + 1)
//   for (int i = 0; i < 12; ++i)
//     p1[i] += p2[i] * 4;

// // unroll value = 1, loop unrolling disabled
// #pragma unroll 1
//   for (int i = 0; i < 12; ++i)
//     p1[i] += p2[i] * 8;

// // unroll value is determined by template expr
// #pragma unroll(trait::value)
//   for (int i = 0; i < 12; ++i)
//     p1[i] += p2[i] * 16;
// }

// __global__ void bar(int *p1, int *p2) { foo<7, ConstantTrait>(p1, p2); }

// #define THREADS_PER_BLOCK 256
// #if __CUDA_ARCH__ >= 200
// #define MY_KERNEL_MAX_THREADS (2 * THREADS_PER_BLOCK)
// #define MY_KERNEL_MIN_BLOCKS 3
// #else
// #define MY_KERNEL_MAX_THREADS THREADS_PER_BLOCK
// #define MY_KERNEL_MIN_BLOCKS 2
// #endif
// // Device code
// __global__ void __launch_bounds__(MY_KERNEL_MAX_THREADS,
// MY_KERNEL_MIN_BLOCKS)
//     MyKernel(...) {
//   ...
// }
