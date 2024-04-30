#include "cute/tensor.hpp"
#include "gtest/gtest.h"

using namespace cute;

TEST(cute, iden_0) { 
    auto a = Layout<Shape<_4, _8>, Stride<_1, _1>>{}; 
}