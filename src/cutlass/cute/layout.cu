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

 