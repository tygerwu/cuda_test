#include "cute/tensor.hpp"
#include "utils.cuh"
#include "utils.h"
#include "gtest/gtest.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <stdlib.h>

template <typename T, int MC, int NC, int KC, typename TiledMMA>
__global__ void gemm_simple(T *dC, const T *dA, const T *dB, int m, int n,
                            int k) {

  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(dA), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(dB), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(dC), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<MC>{}, Int<KC>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<NC>{}, Int<KC>{}), make_coord(ix, _));
  Tensor gC =
      local_tile(C, make_tile(Int<MC>{}, Int<NC>{}), make_coord(iy, ix));
  //  gA(MC, KC, num_tile_k)
  //  gB(NC, KC, num_tile_k)
  //  gC(MC, NC)

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

  // auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M,
  // MMA_K) auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA,
  // MMA_N, MMA_K) auto tCrC = thr_mma.partition_fragment_C(gC(_, _));    //
  // (MMA, MMA_M, MMA_N)

  // clear(tCrC);

  //   int num_tile_k = size<2>(gA);
  // #pragma unroll 1
  //   for (int itile = 0; itile < num_tile_k; ++itile) {
  //     cute::copy(tAgA(_, _, _, itile), tArA);
  //     cute::copy(tBgB(_, _, _, itile), tBrB);

  //     cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  //   }

  //   cute::copy(tCrC, tCgC);
}

TEST(cute, simple_gemm) {
  constexpr int MC = 128;
  constexpr int NC = 128;
  constexpr int KC = 32;

  int m = 256;
  int n = 256;
  int k = 256;
  using T = cute::half_t;
  using namespace cute;
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using MMA =
      decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_2, _2, _1>{}),
                              make_layout(Shape<_1, _2, _1>{})));

  int ASIZE = m * k, BSIZE = k * n, CSIZE = m * n;
  int ABYTES = ASIZE * sizeof(T);
  int BBYTES = BSIZE * sizeof(T);
  int CBYTES = CSIZE * sizeof(T);

  T *dA, *dB, *dC;
  CUDA_ERROR_CHECK(cudaMalloc(&dA, ABYTES));
  CUDA_ERROR_CHECK(cudaMalloc(&dB, BBYTES));
  CUDA_ERROR_CHECK(cudaMalloc(&dC, CBYTES));

  // Allocate host memory
  FloatVector hA = CreateData<float>(ASIZE, 0, 6);
  FloatVector hB = CreateData<float>(BSIZE, 0, 6);
  FloatVector hC(CSIZE, 0);

  // Copy memory from host to device
  CUDA_ERROR_CHECK(cudaMemcpy(dA, hA.data(), ABYTES, cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMemcpy(dB, hB.data(), BBYTES, cudaMemcpyHostToDevice));

  auto blockThreadNum = size(MMA{});
  dim3 block(blockThreadNum);
  dim3 grid(n / NC, m / MC);
  gemm_simple<T, MC, NC, KC, MMA><<<grid, block>>>(dC, dA, dB, m, n, k);
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // Copy memory from devie to host
  CUDA_ERROR_CHECK(cudaMemcpy(hC.data(), dC, CBYTES, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
}
