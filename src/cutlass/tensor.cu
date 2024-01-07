#include "cute/tensor.hpp"
#include "utils.cuh"
#include "utils.h"
#include "gtest/gtest.h"
#include <cublas_v2.h>

TEST(cute, tensor) {
  using T = cute::half_t;
  using namespace cute;
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  constexpr size_t MC = 128;
  constexpr size_t KC = 32;

  int m = 256;
  int n = 256;
  int k = 256;
  int ASIZE = m * k, BSIZE = k * n, CSIZE = m * n;
  int ABYTES = ASIZE * sizeof(T);
  int BBYTES = BSIZE * sizeof(T);
  int CBYTES = CSIZE * sizeof(T);

  // Allocate host memory
  FloatVector hA = CreateData<float>(ASIZE, 0, 6);
  FloatVector hB = CreateData<float>(BSIZE, 0, 6);
  FloatVector hC(CSIZE, 0);

  using MMA =
      decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_2, _2, _1>{}),
                              make_layout(Shape<_1, _1, _1>{})));
  MMA tiled_mma;

  auto thr_mma = tiled_mma.get_slice(0);

  Tensor A = make_tensor(hA.data(), make_shape(m, k), make_stride(1, k));
  auto tile = make_tile(Int<MC>{}, Int<KC>{});
  Tensor gA = local_tile(A, tile, make_coord(0, 0));
  // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tAgA = thr_mma.partition_A(gA);

  print(tiled_mma);
  std::cout << std::endl;

  print(thr_mma);
  std::cout << std::endl;

  print(A);
  std::cout << std::endl;
  print(gA);
  std::cout << std::endl;
  print(tAgA);
}

TEST(cute, tensor2) {
  using T = cute::half_t;
  using namespace cute;
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  int m = 256;
  int n = 32;
  int k = 64;
  int ASIZE = m * k, BSIZE = k * n, CSIZE = m * n;
  int ABYTES = ASIZE * sizeof(T);
  int BBYTES = BSIZE * sizeof(T);
  int CBYTES = CSIZE * sizeof(T);

  // Allocate host memory
  FloatVector hA = CreateData<float>(ASIZE, 0, 6);
  FloatVector hB = CreateData<float>(BSIZE, 0, 6);
  FloatVector hC(CSIZE, 0);

  using MMA =
      decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_1, _1, _1>{}),
                              make_layout(Shape<_1, _2, _2>{})));
  MMA tiled_mma;
  // ColMajor
  auto b_tensor = make_tensor(hB.data(), make_shape(k, n), make_stride(n, 1));
  auto thrb_tensor = tiled_mma.thrfrg_B(b_tensor);
  auto tid_tensor = tiled_mma.tidfrg_B(b_tensor);
  auto tbGb = tiled_mma.get_slice(1).partition_B(b_tensor);

  print(tbGb);
  printf("\n");

  print(tid_tensor);
  printf("\n");
  print(thrb_tensor);
  printf("\n");
  print(tiled_mma);
  printf("\n");
}
