

__global__ void vec_add_v1(const float *__restrict__ a,
                           const float *__restrict__ b, float *c, int n) {
  int tx = threadIdx.x;
  int thread_num = blockdim.x;
  int i = 0;
  for (; i + thread_num < n + 1; i += thread_num) {
    c[i] = a[i] + b[i];
  }
}