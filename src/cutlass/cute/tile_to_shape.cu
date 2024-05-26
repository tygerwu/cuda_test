
#include "cute/tensor.hpp"
#include "gtest/gtest.h"
#include "utils.h"

using namespace cute;


template <class Shape, class Stride,
          class TrgShape, class ModeOrder = LayoutLeft>
CUTE_HOST_DEVICE constexpr
auto
tile_to_shape_(Layout<Shape,Stride> const& block,
              TrgShape             const& trg_shape,
              ModeOrder            const& ord_shape = {}){
  CUTE_STATIC_ASSERT_V(rank(block) <= rank(trg_shape), "Rank of layout must be <= rank of target shape.");
  constexpr int R = rank_v<TrgShape>;

  auto padded_block = append<R>(block);

  auto block_shape  = product_each(shape(padded_block));
  auto target_shape = product_each(shape(trg_shape));

  // Assert proper division
  if constexpr (is_static<decltype(target_shape)>::value) {
    CUTE_STATIC_ASSERT_V(weakly_compatible(block_shape, target_shape),
                        "tile_to_shape: block shape does not divide the target shape.");
  }

  auto product_shape = ceil_div(target_shape, block_shape);
  Print("padded_block:",padded_block);
  Print("product_shape:",product_shape);
  Print("ordered_layout:", make_ordered_layout(product_shape, ord_shape));
  Print("blocked_product:", blocked_product(padded_block, make_ordered_layout(product_shape, ord_shape)));

  return coalesce(blocked_product(padded_block, make_ordered_layout(product_shape, ord_shape)), product_shape);
}

TEST(cute,tile_to_shape){
    // MMajor
    using BM = Int<64>;
    using BK = Int<32>;
    using BKStages = Int<3>;
    using AtomLayout = Layout<Shape<_32,_8>,Stride<_1,_32>>;
    using TrgShape = Shape<BM,BK,BKStages>;
    using Steps = Step<_2,_1,_3>;
    auto SMemLayout = tile_to_shape_(AtomLayout{},TrgShape{},Steps{});

    using Steps2 = Step<_1,_2,_3>;
    auto SMemLayout2 = tile_to_shape_(AtomLayout{},TrgShape{},Steps2{});
    
    Print("SMemLayout:",SMemLayout);
    Print("SMemLayout2:",SMemLayout2);
}
