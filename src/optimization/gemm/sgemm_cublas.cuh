#include <cublas_v2.h>

static void CublasSgemm(const float *a, const float *b, float *c, int m, int n,
                        int k) {

  // cublas
  cublasHandle_t blas_handle;
  cublasCreate(&blas_handle);
  float alpha = 1.0;
  float beta = 0;
  cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, a, k, b,
              n, &beta, c, n);
}
