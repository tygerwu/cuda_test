#include "cute/tensor.hpp"
#include "gtest/gtest.h"
#include "utils.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/detail/layout.hpp"
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
    auto bit_layout = GMMA::Layout_MN_SW64_Atom_Bits{};
    auto half_layout = GMMA::Layout_MN_SW64_Atom<cute::half_t>{};
    Print("bit_layout:",bit_layout);
    Print("half_layout:",half_layout);
}



TEST(cute, order) { 
    auto shape = Shape<_2,_4,_6>{};
    auto step1 = Step<_1,_0,_2>{};
    auto step2 = Step<_2,_0,_1>{};
    auto layout0 = make_layout(shape);
    auto layout1 = make_ordered_layout(shape,step1);
    auto layout2 = make_ordered_layout(shape,step2);
    Print("layout0:",layout0);
    Print("layout1:",layout1);
    Print("layout2:",layout2);
}



TEST(cute, product) { 
    auto A = Layout<Shape<_2,_2,>,Stride<_1,_2>>{};
    auto B = Layout<Shape<_3,_4>,Stride<_4,_1>>{};
 
    auto c = complement(A, size(A)*cosize(B));
    auto d = composition(c,B);

    auto e = logical_product(A,B);

    // make_layout(A, 
    //         composition(complement(A, size(A)*cosize(B)),B));
    Print("c:",c);
    Print("d:",d);
    PrintValue("e:",e);
    PrintValue("d:",d);
}


TEST(cute, stride) { 
    using LayoutTag = cutlass::layout::RowMajor;
    using Stride = cutlass::detail::TagToStrideA_t<LayoutTag>;
    Print("Stride:",Stride{});
}



TEST(cute, local_tile) { 
    using BM = _32;
    using BK = _32;
    int GM = 64;
    int GK = 64;
    int Batch = 2;

    auto g_layout = make_layout(make_shape(GM,GK,Batch),make_stride(GK,1,GM*GK));

    auto tile = Shape<BM,BK>();

    Print("zip:",zipped_divide(g_layout,tile));
}