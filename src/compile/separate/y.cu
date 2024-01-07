//---------- y.cu ----------
#define N 8
__device__ int g[N];

__device__ void bar(void) { g[threadIdx.x]++; }