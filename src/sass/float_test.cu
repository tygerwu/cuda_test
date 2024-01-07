

#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
using namespace nvcuda;

#define FP4_PTR(pointer) reinterpret_cast<float4 *>(pointer)
#define CONST_FP4_PTR(pointer) reinterpret_cast<const float4 *>(pointer)

__global__ void HAdd(const half *__restrict__ a, const half *__restrict__ b,
                     half *c) {
  int offset = threadIdx.x + blockIdx.x * blockDim.x;
  c[offset] = a[offset] + b[offset];
}

__global__ void SAdd(const float *__restrict__ a, const float *__restrict__ b,
                     float *c) {
  __shared__ float a_tmp[32];
  __shared__ float b_tmp[32];
  int offset = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x % 32;
  a_tmp[tid] = a[offset];
  b_tmp[tid] = b[offset];
  __syncthreads();

  c[offset] = a_tmp[tid] + b_tmp[tid];
}

// __global__ void SAdd4(const float *__restrict__ input, float *output) {
//   int offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

//   output[offset] = input[offset];

//   *FP4_PTR((output + offset)) = *CONST_FP4_PTR(input + offset);
// }

__global__ void m8nnk4(const half *a, const half *b, half *c) {
  __shared__ half a_smem[64];
  __shared__ half b_smem[64];
  int tid = threadIdx.x;
  a_smem[2 * tid] = a[2 * tid];
  a_smem[2 * tid + 1] = a[2 * tid + 1];

  b_smem[2 * tid] = b[2 * tid];
  b_smem[2 * tid + 1] = b[2 * tid + 1];
  __syncthreads();

  union F {
    half2 fp16_data;
    ushort2 u16_data;
    float fp32_data;
  };
  F fa, fb;

  auto a_smem_addr = __cvta_generic_to_shared(&a_smem) + tid * 16;
  auto b_smem_addr = __cvta_generic_to_shared(&b_smem) + tid * 16;

  __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0},[%1];"
                       : "=f"(fa.fp32_data)
                       : "l"(a_smem_addr)
                       : "memory");

  __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0},[%1];"
                       : "=f"(fb.fp32_data)
                       : "l"(b_smem_addr)
                       : "memory");

  unsigned c0(0), c1(0), c2(0), c3(0);
  __asm__ __volatile__(
      R"(    
        mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 {%0,%1,%2,%3},{%4,%5},{%6,%7},{%0,%1,%2,%3};)"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"((unsigned)fa.u16_data.x), "r"((unsigned)fa.u16_data.y),
        "r"((unsigned)fb.u16_data.x), "r"((unsigned)fb.u16_data.y)
      : "memory");

  printf("Tid: %d, Value: %f, %f, %f,  %f\n", tid, (float)(c0), (float)(c1),
         (float)(c2), (float)(c3));
}

__global__ void WMMA(const half *a, const half *b, float *c) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::load_matrix_sync(a_frag, a, 16);

  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::load_matrix_sync(b_frag, b, 16);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}