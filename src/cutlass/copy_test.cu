#include "cute/tensor.hpp"
#include "utils.cuh"
#include "utils.h"
#include "gtest/gtest.h"
#include <cublas_v2.h>

TEST(cute, copy) {
  using T = float;
  using namespace cute;
  using copy_op = UniversalCopy<T, T>;
  using copy_traits = Copy_Traits<copy_op>;

  int m = 256;
  int n = 32;
  int SIZE = m * n;
  int BYTES = SIZE * sizeof(T);

  // Allocate host memory
  FloatVector hV = CreateData<float>(SIZE, 0, 6);

  // Thread arrangement
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<2>{}));

  // Vector dimensions
  Layout vec_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));
  Copy_Atom<copy_traits, T> copy_atom;
  auto tiled_copy = make_tiled_copy(copy_atom,   // access size
                                    thr_layout,  // thread layout
                                    vec_layout); // vector layout (e.g. 4x1)

  //  constexpr int R = cute::max(rank_v(thr_layout), rank_v(vec_layout));

  auto thr_layout_mn = append<2>(thr_layout, Layout<_1>{});
  auto val_layout_mn = append<2>(vec_layout, Layout<_1>{});

  // Take the raked_products to compute the Layout_MN
  auto layout_mn = raked_product(thr_layout_mn, val_layout_mn);
  auto layout_tv = right_inverse(layout_mn).with_shape(
      make_shape(size(thr_layout), size(vec_layout)));
  //   return make_tiled_copy_impl(copy_atom, layout_tv,
  //                               product_each(shape(layout_mn)));
  print("thr_layout: ");
  print(thr_layout_mn);
  print("\n");

  print("thr_layout_mn: ");
  print(thr_layout_mn);
  print("\n");

  print("val_layout_mn: ");
  print(val_layout_mn);
  print("\n");

  print("val_layout: ");
  print(val_layout_mn);
  print("\n");
  print("layout_mn : ");
  print(layout_mn);
  print("\n");
  print("layout_tv : ");
  print(layout_tv);
  print("\n");

  print(tiled_copy);
}
