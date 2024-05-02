#include "cute/tensor.hpp"
#include "gtest/gtest.h"
#include "utils.h"
using namespace cute;


TEST(cute,tma_1){
    using T = uint16_t;
    auto g_shape  = make_shape(make_shape(80,40),make_shape(32,12));
    auto g_layout = make_layout(g_shape);

    auto g_data   = std::vector<T>(size(g_layout));

    auto g_tensor = make_tensor(make_gmem_ptr(g_data.data()),g_layout);

    
    auto s_shape  = Shape<_64,_32>{};
    auto s_layout = make_layout(s_shape);

    auto cta_tile = Shape<Shape<_16,_4>,Shape<_8,_4>>{}; 

    auto tma = make_tma_copy(SM90_TMA_LOAD_MULTICAST{},g_tensor,s_layout,cta_tile,_2{});
}
