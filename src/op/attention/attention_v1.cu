#include "attention_v1_config.cuh"
#include "attention_api.cuh"
#include "macros.h"

namespace Op{



template<typename T,int HDSize>
__global__ void AttentionV1Kernel(const T* q,const T* k,const T* v,T* o,
                                  int batch,int qo_seq_len,int kv_seq_len,int head_num,float log2_scale,
                                  int q_batch_stride,int kv_batch_stride,int o_batch_stride){

    using HD = Int<HDSize>;
    using CFG = typename AttentionV1ConfigTratis<HDSize>::CFG;                      
    using BM  = typename CFG::BM;
    using BN  = typename CFG::BN;
    using BK  = typename CFG::BK;
    using BK2 = typename CFG::BK2;
    using BN2 = typename CFG::BN2;

    using BKNum   = typename CFG::BKNum;
    using BK2Num  = typename CFG::BK2Num;
    using BN2Num  = typename CFG::BN2Num;
    using BKTiles = typename CFG::BKTiles;


    extern __shared__ char smem[];

    // Positions
    int tid = threadIdx.x;
    int bx  = blockIdx.x;             // qo_seq_len
    int head_id  = blockIdx.y;        // head_num
    int batch_id = blockIdx.z;        // batch

    // Instances
    auto tiled_g2s_q    = typename CFG::G2SCopyQ{};
    auto tiled_g2s_k    = typename CFG::G2SCopyK{};
    auto tiled_g2s_v    = typename CFG::G2SCopyV{};
    auto tiled_qk_mma   = typename CFG::QKMMA{};
    auto tiled_pv_mma   = typename CFG::PVMMA{};

    auto tiled_s2r_q    = typename CFG::S2RCopyQ{};
    auto tiled_s2r_k    = typename CFG::S2RCopyK{};
    auto tiled_s2r_v    = typename CFG::S2RCopyV{};

    auto tiled_r2s_o    = typename CFG::R2SCopyO{};
    auto tiled_s2g_o    = typename CFG::S2GCopyO{};

    auto g2s_q = tiled_g2s_q.get_slice(tid);
    auto g2s_k = tiled_g2s_k.get_slice(tid);
    auto g2s_v = tiled_g2s_v.get_slice(tid);

    auto s2r_q = tiled_s2r_q.get_slice(tid);
    auto s2r_k = tiled_s2r_k.get_slice(tid);
    auto s2r_v = tiled_s2r_v.get_slice(tid);

    auto r2s_o = tiled_r2s_o.get_slice(tid);
    auto s2g_o = tiled_s2g_o.get_slice(tid);

    int bn_num = kv_seq_len / BN{};

    int qo_head_stride  = qo_seq_len * HDSize;
    int kv_head_stride  = kv_seq_len * HDSize;
    int kv_head_offset  = batch_id * kv_batch_stride + head_id * kv_head_stride;
    int q_head_offset   = batch_id * q_batch_stride + head_id * qo_head_stride;
    int o_head_offset   = batch_id * o_batch_stride + head_id * qo_head_stride;
    int qo_block_offset = bx * BM{} * HDSize;


    // GMem Block Q
    auto gq_block_layout = Layout<Shape<BM,BK,BKNum>,Stride<HD,_1,BK>>{};
    auto gq_block = make_tensor(make_gmem_ptr(q+q_head_offset+qo_block_offset),gq_block_layout);
    // SMem Block Q
    auto sq = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem + typename CFG::SOffQ{})),typename CFG::SMemLayoutQ{});

    auto g2s_src_q = g2s_q.partition_S(gq_block);   //(8,1),G2S_ValTile_BM,G2S_ValTile_BK,BKNum
    auto g2s_dst_q = g2s_q.partition_D(sq);         // ~

   
    // Load Q into SMem as early as possible
    copy(tiled_g2s_q,g2s_src_q,g2s_dst_q);
    cp_async_fence();


    // GMem K,V,O
    auto gk_head_layout  = make_layout(make_shape( BN{},BK{},BKNum{},bn_num),
                                      make_stride(HD{},_1{},BK{},   HD{}*BN{}));                          // KMajor
    auto gv_head_layout  = make_layout(make_shape( BN2{},BK2{},BN2Num{},BK2Num{},  bn_num),
                                      make_stride(_1{}, HD{}, BN2{},   BK2{}*HD{},BK2{}*HD{}*BK2Num{}));  // MMjaor
    auto go_block_layout = make_layout(make_shape( BM{},BN2{},BN2Num{}),
                                      make_stride(HD{},_1{}, BN2{}));                                     // KMjaor


    auto gk_head  = make_tensor(make_gmem_ptr(k+kv_head_offset),gk_head_layout);
    auto gv_head  = make_tensor(make_gmem_ptr(v+kv_head_offset),gv_head_layout);
    auto go_block = make_tensor(make_gmem_ptr(v+o_head_offset+qo_block_offset),go_block_layout);

    // SMem K,V,O
    auto sk = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem + typename CFG::SOffK{})),typename CFG::SMemLayoutK{});
    auto sv = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem + typename CFG::SOffV{})),typename CFG::SMemLayoutV{});
    auto so = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem + typename CFG::SOffO{})),typename CFG::SMemLayoutO{});

    // G2S K,V
    auto g2s_src_k = g2s_k.partition_S(gk_head);     // (8,1),G2S_ValTile_BN,G2S_ValTile_BK,BKNum,bn_num
    auto g2s_dst_k = g2s_k.partition_D(sk);          // (8,1),G2S_ValTile_BN,G2S_ValTile_BK,BKNum

    auto g2s_src_v = g2s_k.partition_S(gv_head);     // (8,1),G2S_ValTile_BN2,G2S_ValTile_BK2,BN2Num,BK2Num,bn_num
    auto g2s_dst_v = g2s_k.partition_D(sv);          // (8,1),G2S_ValTile_BN2,G2S_ValTile_BK2,BN2Num,BK2Num,

    // Reg Q,K,V
    auto rq = make_tensor<T>(typename CFG::RShapeQ{});   // (2,2,2),QKMMA_ValTile_BM,2
    auto rk = make_tensor<T>(typename CFG::RShapeK{});   // (2,2),  QKMMA_ValTile_BN,2
    auto rv = make_tensor<T>(typename CFG::RShapeV{});


    // S2R Q,K,V
    auto s2r_src_q_bks = s2r_q.partition_S(sq);         // (8,1),S2R_ValTile_BM,S2R_ValTile_BK,BKNum
    auto s2r_dst_q     = s2r_q.retile_D(rq);            // (8,1),S2R_ValTile_BM,2

    auto s2r_src_k_bks = s2r_k.partition_S(sk);         // (8,1),S2R_ValTile_BN,S2R_ValTile_BK,BKNum
    auto s2r_dst_k     = s2r_k.retile_D(rk);            // (8,1),S2R_ValTile_BN,2

    auto s2r_src_v = s2r_v.partition_S(sv);             // (8,1),(S2R_ValTile_BN2),S2R_ValTile_BK2,BN2Num,BK2Num
    auto s2r_dst_v = s2r_v.retile_D(rv);                // (8,1),(S2R_ValTile_BN2),2


    // Reg HO
    auto rho = make_tensor<T>(typename CFG::RShapeO{});
    // R2S
    //  s2r_tile_mn : (PVMMA_M,PVMMA_N), ex:64x16
    auto r2s_src_ho = group_diff<1,0>(flatten(r2s_o.retile_S(rho)));   // ((2),S2RAtom_ValTile_PVMMA_M,S2RAtom_ValTile_PVMMA_N,S2R_ValTile_BM,S2R_ValTile_BN2,BN2Num
    auto r2s_dst_ho = group_diff<1,0>(flatten(r2s_o.partition_D(so))); // ~
    // S2G
    auto s2g_src_ho = s2g_o.partition_S(so);            // (8,1),S2G_ValeTile_BM,S2G_ValeTile_BN2,BN2Num
    auto s2g_dst_ho = s2g_o.partition_S(go_block);      // ~


    // Reg accumulators
    auto rfx = make_tensor<float>(typename CFG::RShapeX{});

    // QKMMA
    auto qk_mm = [&](int bk){
        // Double buffer
        int ld_id = 0;
        int st_id = 0;

        auto s2r_src_q = s2r_src_q_bks(_,_,_,bk);
        auto s2r_src_k = s2r_src_k_bks(_,_,_,bk);

        // Prefetch for 1st mma
        copy(tiled_s2r_q,s2r_src_q(_,_,0),s2r_dst_q(_,_,st_id));
        copy(tiled_s2r_k,s2r_src_k(_,_,0),s2r_dst_k(_,_,st_id));
        st_id ^= 1;
        for(int i=0; i<BKTiles{}; i++){
            if(i+1<BKTiles{}){
                // Prefetch for next round
                copy(tiled_s2r_q,s2r_src_q(_,_,i+1),s2r_dst_q(_,_,st_id));
                copy(tiled_s2r_k,s2r_src_k(_,_,i+1),s2r_dst_k(_,_,st_id));
                st_id ^= 1;
            }

            gemm(tiled_qk_mma,rfx,rq(_,_,ld_id),rk(_,_,ld_id),rfx);
            ld_id ^= 1;
        }     
    };

    for(int bn=0; bn<bn_num; bn++){

    }

    if(thread0()){
        Print("r2s_src_ho:",r2s_src_ho);
        Print("r2s_dst_ho:",r2s_dst_ho);

        Print("s2g_src_ho:",s2g_src_ho);
        Print("s2g_dst_ho:",s2g_dst_ho);
    }




}

void PrintAttentionV1Info(int HD){
    if(HD == 64){
        using CFG = typename AttentionV1ConfigTratis<64>::CFG;
        CFG{}.print();
    }
}


void AttentionV1(const __half* q,const __half* k,const __half* v,__half* o,
                 int batch,int head_num,int qo_seq_len,int kv_seq_len,int head_dim,float softmax_scale,
                 int q_batch_stride,int kv_batch_stride,int o_batch_stride,
                 cudaStream_t stream){
    
    using T = __half;
    
    int BM;
    int threads;
    int smem_bytes;
    auto func = AttentionV1Kernel<T,64>;
    if(head_dim == 64){
        using CFG = typename AttentionV1ConfigTratis<64>::CFG;
        BM = typename CFG::BM{};
        threads = typename CFG::Threads{};
        smem_bytes = CFG::SBytes;
    }
    if(smem_bytes >= (48 << 10)){
        CUDA_ERROR_CHECK(cudaFuncSetAttribute(func,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_bytes));
    }
    dim3 grid(UP_DIV(qo_seq_len,BM),head_num,batch);
    dim3 block(threads);

    func<<<grid,block,smem_bytes,stream>>>(
        q,k,v,o,
        batch,qo_seq_len,kv_seq_len,head_num,softmax_scale,
        q_batch_stride,kv_batch_stride,o_batch_stride
    );
}


}