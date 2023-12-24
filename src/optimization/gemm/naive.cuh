#include <cuda_runtime.h>
#include <stdio.h>

static __global__ void NaiveSgemmImpl(const float *a, const float *b, float *c,
                                      int m, int n, int k) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  for (int p = 0; p < k; p++) {
    c[i * n + j] += a[i * k + p] * b[p * n + j];
  }
}

static void NaiveSgemm(const float *a, const float *b, float *c, int m, int n,
                       int k) {
  constexpr int TB_SIZE = 32;
  dim3 blockDim(TB_SIZE, TB_SIZE, 1);
  dim3 gridDim(n / TB_SIZE, m / TB_SIZE, 1);
  NaiveSgemmImpl<<<gridDim, blockDim>>>(a, b, c, m, n, k);
}
