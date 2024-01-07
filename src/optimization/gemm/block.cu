// #include <cuda_runtime.h>
// #include <stdio.h>

// template <int BLOCK_SIZE>
// static __global__ void block_sgemm_kernel(const float *a, const float *b,
//                                           float *c, int k) {
//   __shared__ A_Block[BLOCK_SIZE][BLOCK_SIZE];
//   __shared__ B_Block[BLOCK_SIZE][BLOCK_SIZE];
//   // Obsolute position
//   int row = blockIdx.x * blockDim.x + threadIdx.x;
//   int col = blockIdx.y * blockDim.y + threadIdx.y;

//   // Relative position inside a block
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//   for (int p = 0; p < k; p += BLOCK_SIZE) {
//     // Each thread load two elements
//     A_Block[tx][ty] = A[i * k + p + tx];
//     B_Block[tx][ty] = B[p * n + j + ty];
//   }
// }

// template <int BlockSize>
// void block_sgemm(const float *a, const float *b, float *c, int m, int n,
//                  int k) {
//   constexpr int TB_SIZE = 32;
//   dim3 blockDim(TB_SIZE, TB_SIZE, 1);
//   dim3 gridDim(n / TB_SIZE, m / TB_SIZE, 1);
//   naive_sgemm_impl<<<gridDim, gridDim>>>(a, b, c, m, n, k);
// }
