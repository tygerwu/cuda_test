#include "cute/tensor.hpp"
#include "gtest/gtest.h"
#include "utils.h"

using namespace cute;

TEST(cute, layout_1) { 
    auto a = Layout<Shape<_4, _8>, Stride<_1, _1>>{}; 
}

TEST(cute, with_shape) { 
    auto a = Layout<Shape<_4,_64>,Stride<_64,_1>>{};
    auto b = a.with_shape(make_shape(_32{},_2{}));
    auto d = a.with_shape(make_shape(_8{},_32{}));
    Print("b:",b);
    Print("d:",d);
}


TEST(cute, shape_div_0) { 
    auto TileShape = Shape<_64,_128,_64>{};
    auto ClusterShape = Shape<_2,_4,_1>{};
    Print("out:",shape_div(TileShape,ClusterShape));
}
 

 TEST(cute, ss) { 
    auto layout = GMMA::Layout_K_SW64_Atom<cute::half_t>{};
    auto layout1 = GMMA::Layout_K_SW64_Atom_Bits{};
    Print("ss:",layout);
    Print("ss1:",layout1);
}