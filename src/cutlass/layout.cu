#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/util/host_tensor.h"
#include "gtest/gtest.h"
using namespace cute;

template <class Shape, class Stride>
void Print2D(Layout<Shape, Stride> const &layout) {
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m, n));
    }
    printf("\n");
  }
}

template <class Shape, class Stride>
void Print1D(Layout<Shape, Stride> const &layout) {
  for (int m = 0; m < size<0>(layout); ++m) {
    printf("%3d  ", layout(m));
  }
  printf("\n");
}

TEST(cutlass, layout) {

  // cute::make_shape(cute::Int<2>{}, Int<3>{});
  // cute::make_stride();
  // auto layout_8s = cute::make_layout(Int<8>{});
  // cute::make_tensor

  // Single compile-time integer
  auto s0 = cute::make_shape(cute::Int<8>());

  // Single runtime integer
  auto s1 = cute::make_shape(8);
  // Single integer + IntTuple
  auto s2 = cute::make_shape(1, s0);
  std::cout << cute::depth(s1) << " " << cute::depth(s2) << std::endl;

  // Shape(2,(2,2))
  // Stride(4,(2,1))
  // Row2Major:
  auto layout =
      cute::make_layout(cute::make_shape(2, cute::make_shape(2, 2)),
                        cute::make_stride(2, cute::make_stride(1, 4)));

  std::cout << cute::get<0>(layout) << " " << cute::get<1>(layout) << std::endl;
  // std::cout << layout(1, 2) << std::endl;
  //   std::cout << layout(1) << std::endl;
  //   std::cout << layout(1, (0, 1)) << std::endl; // 3
  //   std::cout << layout(1, (1, 0)) << std::endl; // 2
  //   std::cout << layout(1, (1, 1)) << std::endl; // 3
  auto coord = cute::make_coord(1, cute::make_coord(1, 1));
  std::cout << layout(coord) << std::endl; // 6

  cute::print(layout);
  cute::print_layout(layout);

  // Print2D(layout);
  // std::cout << cute::rank(layout) << std::endl;
  // std::cout << cute::depth(layout) << std::endl;
  // std::cout << cute::shape(layout) << std::endl;
  // std::cout << cute::stride(layout) << std::endl;
  // std::cout << cute::size(layout) << std::endl;
}

TEST(cutlass, layout2) {
  auto layout = cute::make_layout(
      cute::make_shape(cute::make_shape(2, 2), cute::make_shape(3, 3)),
      cute::make_stride(cute::make_stride(3, 6), cute::make_stride(1, 12)));
  cute::print_layout(layout);

  // std::cout << layout(1) << std::endl;
  // std::cout << layout(1, 2) << std::endl;
  // std::cout << layout(cute::make_coord(_, _), cute::make_coord(1, 1))
  //           << std::endl;

  // cute::print_layout(layout(cute::make_coord(_, _), cute::make_coord(0, 0)));
  // cute::print_layout(layout(cute::make_coord(0, 0), cute::make_coord(_, _)));
  // cute::print_layout(
  //     layout(cute::make_coord(1, 1), cute::make_coord(cute::_, cute::_)));

  // auto slice_layout = cute::slice(
  //     cute::make_coord(cute::make_coord(_, 1), cute::make_coord(_, 1)),
  //     layout);
  // Print2D(slice_layout);
  // std::cout << slice_layout << std::endl;

  cute::print(cute::flatten(layout));
}

TEST(cutlass, layout3) {
  auto A = Layout<Shape<_2, _2>, Stride<_1, _2>>{};
  auto B = Layout<Shape<_3, _4>, Stride<_4, _1>>{};
  std::cout << size(A) << " " << cosize(B) << std::endl;

  auto t0 = complement(A, 48);
  auto t1 = composition(t0, B);
  auto C = make_layout(A, t1);
  print(t0);
  print("\n");
  print(t1);
  print("\n");
  print(C);
}

TEST(cute, layout4) {
  auto layout =
      make_layout(make_shape(make_shape(4, 8), make_shape(2, 2, 2)),
                  make_stride(make_stride(32, 1), make_stride(16, 8, 128)));

  print_layout(layout);
}

TEST(cute, layout5) {
  auto layout =
      make_layout(make_shape(make_shape(2, 2, 2), make_shape(2, 2, 2)),
                  make_stride(make_stride(1, 16, 4), make_stride(8, 2, 32)));

  print_layout(layout);
}