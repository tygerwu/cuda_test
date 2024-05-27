#include "cute/tensor.hpp"
#include "gtest/gtest.h"
#include "utils.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/detail/layout.hpp"
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

    Print("tma:",tma);
}

TEST(cute,tma_2){
    int GM = 150*3;
    int GK = 64*5;
    int Batch = 3;
    int cta_coord = 0;
    int cta_index = 1;
    int bm_idx = 2;
    int batch_idx = 1;
    constexpr bool KMajor = 1;
    using C = _2;

    using T = uint16_t;
    using BM = _128;
    using BK = _32;
    using Stages = _3;
    using StageStride = Int<int(BM{} * BK{})>;

    auto g_stride = make_stride(GK,1,GM*GK);          // K-Major
    if(!KMajor){
         g_stride = make_stride(1,GM,GM*GK);
    }
    auto g_layout = make_layout(make_shape(GM,GK,Batch), g_stride);

    auto g_data   = std::vector<T>(size(g_layout));
    auto g_tensor = make_tensor(make_gmem_ptr(g_data.data()),g_layout);


    using SStageStride = std::conditional_t<KMajor,Stride<BK,_1>,Stride<_1,BM>>;
    using SStride = std::conditional_t<KMajor,Stride<BK,_1,StageStride>,Stride<_1,BM,StageStride>>;

    auto s_layout_stage = make_layout(Shape<BM,BK>{},SStageStride{});
    auto s_layout       = make_layout(Shape<BM,BK,Stages>{},SStride{});
    auto s_data   = std::vector<T>(size(s_layout));
    auto s_tensor = make_tensor(make_gmem_ptr(s_data.data()),s_layout);

    auto cta_tile = Shape<BM,BK>{}; 
    auto tma = make_tma_copy(SM90_TMA_LOAD_MULTICAST{},g_tensor,s_layout_stage,cta_tile,C{});

    auto block_a = make_shape(BM{},GK);
    auto m_a = tma.get_tma_tensor(make_shape(GM,GK,Batch));               // (GM,GK,Batch) TMA coord tensor
    auto g_a = local_tile(m_a, block_a, make_coord(bm_idx,0,batch_idx)); // (BM,GK) TMA coord tensor for this CTA

    auto cta_tma = tma.get_slice(cta_index);             // Slice for multicast partitioning
    auto tma_src_a = cta_tma.partition_S(g_a);           // Partition for src
    auto tma_dst_a = cta_tma.partition_D(s_tensor);      // Partition for dst
    //copy(tma.with(barrier, mcast_mask), tma_src_a, tma_dst_a);          // copy with supporting TMA params


    Print("m_a:",m_a);
    Print("g_a:",g_a);
    Print("div:",zipped_divide(m_a,cta_tile));
    Print("tidfrg_S:",tma.tidfrg_S(g_a));
    Print("tidfrg_D:",tma.tidfrg_D(s_tensor));
    Print("tma_src_a:",tma_src_a);
    Print("tma_dst_a:",tma_dst_a);

    auto tiledtma_src_a = tma.tidfrg_S(g_a);

    Print("pos0:",layout(tma_src_a)(make_coord(0,0,2)));
    Print("pos1:",tma_src_a(make_coord(_,_,2)));
}




TEST(cute,tma_3){
    using T = half_t;

    // ColMajor
    using LayoutA = cutlass::layout::ColumnMajor;
    using AtomLayout = Layout<Shape<_32,_8>,Stride<_1,_32>>;
    using BM = Int<128>;
    using BK = Int<32>;
    using BKStages = Int<3>;
    using Steps = Step<_2,_1,_3>;
    using SMemLayout = decltype(tile_to_shape(AtomLayout{},
                                              make_shape(BM{},BK{},BKStages{}),
                                              Steps{}));
    using Steps2 = Step<_1,_2,_3>;
    using SMemLayout2 = decltype(tile_to_shape(AtomLayout{},
                                              make_shape(BM{},BK{},BKStages{}),
                                                 Steps2{}));
    
    using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;

    using GmemTiledCopyA = SM90_TMA_LOAD_MULTICAST;


    Print("SMemLayout2:",SMemLayout2{});
    Print("SMemLayout:",SMemLayout{});
    
    using ClusterShape = Shape<_2>;

    int GM = 256;
    int GK = 128;
    int Batch = 3;
    auto g_layout = make_layout(make_shape(GM,GK,Batch), make_stride(1,GM,GM*GK));
    auto g_data   = std::vector<T>(size(g_layout));
    auto g_tensor = make_tensor(make_gmem_ptr(g_data.data()),g_layout);

    auto tma_a  = make_tma_copy(GmemTiledCopyA{},g_tensor,SMemLayout{}(_,_,_0{}),make_shape(BM{},BK{}),ClusterShape{});
    auto tma_a2 = make_tma_copy(GmemTiledCopyA{},g_tensor,SMemLayout2{}(_,_,_0{}),make_shape(BM{},BK{}),ClusterShape{});

    Print("tma_a:",tma_a);   // boxDim         (32,16,1,1,1)
    Print("tma_a2:",tma_a2); // boxDim         (32,4,1,1,1)
    Print("tma_a2:",tma_a2.get_slice(2).partition_S(g_tensor(_,_,0))); 
}



TEST(cute,tma_4){
    using T = half_t;

    // ColMajor
    using LayoutA = cutlass::layout::ColumnMajor;
    using AtomLayout = Layout<Shape<_32,_8>,Stride<_1,_32>>;
    using BN = Int<128>;
    using BK = Int<32>;
    using BKStages = Int<3>;
    using Steps = Step<_2,_1,_3>;
    using SMemLayout = decltype(tile_to_shape(AtomLayout{},
                                              make_shape(BN{},BK{},BKStages{}),
                                              Steps{}));

    using GmemTiledCopyA = SM90_TMA_LOAD_MULTICAST;


    Print("SMemLayout:",SMemLayout{});
    
    int GN = 256;
    int GK = 256;
    int HN = 2;
    int Batch = 3;

    auto g_layout = make_layout(make_shape(GN,GK,HN,Batch), 
                                make_stride(1,GN,GN*GK,GN*GK*HN));

    auto g_data   = std::vector<T>(size(g_layout));
    auto g_tensor = make_tensor(make_gmem_ptr(g_data.data()),g_layout);

    auto tma_a  = make_tma_copy(GmemTiledCopyA{},g_tensor,SMemLayout{}(_,_,_0{}),make_shape(BN{},BK{}),_2{});

    Print("tma_a:",tma_a);   
}