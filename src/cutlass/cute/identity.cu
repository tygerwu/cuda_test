#include "cute/tensor.hpp"
#include "gtest/gtest.h"
#include "utils.h"

using namespace cute;

TEST(cute,iden){
    using BM = _8;
    using BK = _32; 
    using TK = Int<int(BK{} / 8)>;
    using TO = Int<int(32 / TK{})>;
    using M  = Int<15>;
    using K  = _32;
    // Padded M
    using PM = _16; 
    using PK = _32;

    using T = uint16_t;

    auto thr_layout = Layout<Shape<TO,TK>,Stride<TK,_1>>{};
    auto val_layout = Layout<Shape<_1,_8>,Stride<_1,_1>>{};

    using Vec = cutlass::AlignedArray<T,8>;

    auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<Vec>,T>{},thr_layout,val_layout);

    std::vector<T> gmem_data(PM{} * K{});
    auto packed_gmem_layout = Layout<Shape<PM,PK>,Stride<PK,_1>>{};
    auto gmem = make_tensor(gmem_data.data(),packed_gmem_layout);

    int tid = 28;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto g2s_src  = thr_copy.partition_S(gmem);


    auto identity = make_identity_tensor(Shape<PM,PK>{});
    auto g2s_iden = thr_copy.partition_S(identity); 

    int M_TiledCopy_ValTile = size<1>(g2s_src);
    int K_TiledCopy_ValTile = size<2>(g2s_src);
    auto pred_data = std::vector<int>(M_TiledCopy_ValTile*K_TiledCopy_ValTile,1);
    auto g2s_pred = make_tensor(pred_data.data(),
                                make_shape(M_TiledCopy_ValTile,K_TiledCopy_ValTile),
                                make_stride(1,0));

    for(int i=0;i<M_TiledCopy_ValTile;i++){
        if(get<0>(g2s_iden(0,i,0)) >= M{}){
            g2s_pred(i,0) = false;
        }
    }

    PrintIden("g2s_iden:",g2s_iden);
    Print("g2s_iden(8):",g2s_iden(8));
}
