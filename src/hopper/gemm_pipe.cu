
#include "utils.h"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/cutlass.h"


using namespace cute;
using namespace cutlass::gemm::collective::detail;

// Size of BKTile is fixed: 16

template<int BMSize,int BNSize,int BKSize,int BKStageNum,int ClusterMSize,int ClusterNSize,int PipeStages>
struct GemmConfig{
    using T = half_t;
    static constexpr GMMA::Major GmmaMajorA = GMMA::Major::K;
    static constexpr GMMA::Major GmmaMajorB = GMMA::Major::K;

    using BM = Int<BMSize>;
    using BN = Int<BNSize>;
    using BK = Int<BKSize>;
    using BKStages = Int<BKStageNum>;
    using TileShapeMNK = Tile<BM,BN,BK>;

    using ClusterM = Int<ClusterMSize>;
    using ClusterN = Int<ClusterNSize>;
    using ClusterShape = Shape<ClusterM,ClusterN>;

    // Tiled MMA
    using MMAOp = decltype(GMMA::ss_op_selector<T,T,float,TileShapeMNK,GmmaMajorA,GmmaMajorB>());
    using MMAThreadTile = Layout<Shape<_1,_1,_1>>;
    using TiledGMMA = decltype(make_tiled_mma(MMAOp{},MMAThreadTile{}));

    // SMemLayout
    using SMemLayoutAtomA = decltype(ss_smem_selector<GmmaMajorA, T, BM, BK>());
    using SMemLayoutAtomB = decltype(ss_smem_selector<GmmaMajorB, T, BN, BK>()); 
    using SMemLayoutA     = decltype(tile_to_shape(SMemLayoutAtomA{},Shape<BM,BK,BKStages>{}));
    using SMemLayoutB     = decltype(tile_to_shape(SMemLayoutAtomB{},Shape<BN,BK,BKStages>{}));

    // TMA
    using TmaCopyAOp      = std::conditional_t<ClusterNSize==1, SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST>;
    using TmaCopyBOp      = std::conditional_t<ClusterMSize==1, SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST>;
    // Structures of strides
    using DummyStrideA    = Stride<_0,_0,_0>;
    using DummyStrideB    = Stride<_0,_0,_0>;

    // TMA Type
    using TMA_A = decltype(make_tma_copy(
        TmaCopyAOp{},
        // Dummy Tensor
        make_tensor(static_cast<half_t const*>(nullptr), 
                    repeat_like(DummyStrideA{}, int32_t(0)), DummyStrideA{}),
        SMemLayoutA{}(_,_,_0{}),
        Shape<BM,BK>{},
        ClusterN{})
    );
    using TMA_B = decltype(make_tma_copy(
        TmaCopyBOp{},
        // Dummy Tensor
        make_tensor(static_cast<half_t const*>(nullptr), 
                    repeat_like(DummyStrideB{}, int32_t(0)), DummyStrideB{}),
        SMemLayoutB{}(_,_,_0{}),
        Shape<BN,BK>{},
        ClusterM{})
    );

    // G2SPiepline
    using G2SPiepline = cutlass::PipelineTmaAsync<PipeStages>;
    using G2SPieplineParams = typename G2SPiepline::Params;

    // SharedMemory Storage
    struct SharedStorage{
        
        struct TensorStorage : aligned_struct<128>{
            array_aligned<half_t,cosize_v<SMemLayoutA>> smem_a;
            array_aligned<half_t,cosize_v<SMemLayoutB>> smem_b;
        } tensors;
        
        using G2SPieplineStorage = typename G2SPiepline::SharedStorage;
        G2SPieplineStorage g2s_pipe;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

};

constexpr int BM = 128;
constexpr int BK = 64;
constexpr int BN = 128; 
constexpr int ClusterM = 2;
constexpr int ClusterN = 2;
constexpr int BKStageNum = 3;
constexpr int G2SPieplineStages = 2;

using CFG = GemmConfig<BM,BK,BN,BKStageNum,ClusterM,ClusterN,G2SPieplineStages>;


__global__ void SM90GemmSSWSKernel(const half_t* a,const half_t* b,half_t* c,
                                   int L,int M,int N,int K,
                                   CFG::TMA_A tma_a,     // TMA muse be created at host
                                   CFG::TMA_B tma_b){


    constexpr int NumWarpsPerWarpGroup = 4;
    constexpr int NumThreadsPerWarpGroup = 128;
    constexpr int NumThreadsPerWarp = 32;

    using TmaCopyAOp = CFG::TmaCopyAOp;
    using TmaCopyBOp = CFG::TmaCopyBOp;
    using SMemLayoutA = CFG::SMemLayoutA;
    using SMemLayoutB = CFG::SMemLayoutB;
    using G2SPiepline = CFG::G2SPiepline;
    using G2SPieplineStage = CFG::G2SPiepline::PipelineState;
    using G2SPieplineParams = CFG::G2SPieplineParams;


    using BM = CFG::BM;
    using BN = CFG::BN;
    using BK = CFG::BK;
    using ClusterM = CFG::ClusterM;
    using ClusterN = CFG::ClusterN;
    using ClusterShape = CFG::ClusterShape;
    // SharedMemory Storage
    using SharedStorage = CFG::SharedStorage;
    using TensorStorage = CFG::SharedStorage::TensorStorage;
    using G2SPieplineStorage = CFG::SharedStorage::G2SPieplineStorage;


    // Positions
    int tid = threadIdx.x;
    // tid within warp
    int lane_id = tid % NumThreadsPerWarp;
    int lane_pred = elect_one_sync();
    int tid_wg  = tid % NumThreadsPerWarpGroup;
    int wid     = tid / NumThreadsPerWarp;
    int bx = blockIdx.x;                // M dimension
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int m_coord = bx / BM{};
    int n_coord = by / BN{};
    int l_coord = bz;
    int bid_cluster = block_rank_in_cluster();
    int bx_cluster = bid_cluster % ClusterM{};
    int by_cluster = bid_cluster / ClusterM{};

    // Size
    int bk_num = K / BK{};

    // Locate block tma within cluster tma
    auto block_tma_a = tma_a.get_slice(by_cluster);
    auto block_tma_b = tma_a.get_slice(bx_cluster);

    // Prefetch TMA Desc
    if(wid == 0 && lane_pred){
        prefetch_tma_descriptor(tma_a.get_tma_descriptor());
        prefetch_tma_descriptor(tma_b.get_tma_descriptor());
    }
    

    extern __shared__ char smem[];
    SharedStorage& smem_storage = *reinterpret_cast<SharedStorage*>(smem);
  

    // Init load pipeline
    G2SPieplineParams g2s_pipe_params;
    // Two WarpGroups in total:
    //  0 : Producer
    //  1 : Consumer
    int wg_role = tid / NumThreadsPerWarpGroup;
    if(wg_role == 0){
        g2s_pipe_params.role = G2SPiepline::ThreadCategory::Producer;
    }else if(wg_role == 1){
        g2s_pipe_params.role = G2SPiepline::ThreadCategory::Consumer;
    }
    g2s_pipe_params.is_leader = tid_wg == 0;
    g2s_pipe_params.transaction_bytes = 128;
    g2s_pipe_params.num_consumers = NumThreadsPerWarpGroup;
    G2SPiepline g2s_pipe(smem_storage.g2s_pipe,g2s_pipe_params,ClusterShape{});
    
    // Init pipeline state
    //  Producer
    G2SPieplineStage g2s_producer_pipe_state = cutlass::make_producer_start_state<G2SPiepline>();
    // Consumer
    G2SPieplineStage g2s_consumer_pipe_state;


    // GMem Tensor
    Tensor ga_tma = tma_a.get_tma_tensor(make_shape(M,K,L));
    Tensor gb_tma = tma_a.get_tma_tensor(make_shape(M,K,L));
    // GMem Block
    Tensor ga_block_tma = local_tile(ga_tma,Shape<BM,BK>{},make_coord(m_coord,_,l_coord));      // (BM,BK,BKNum)
    Tensor gb_block_tma = local_tile(gb_tma,Shape<BN,BK>{},make_coord(n_coord,_,l_coord));

    // SMem Tensor
    Tensor sa = make_tensor(make_smem_ptr(smem_storage.tensors.smem_a.data()), SMemLayoutA{});  // (BM,BK,BKStages)
    Tensor sb = make_tensor(make_smem_ptr(smem_storage.tensors.smem_b.data()), SMemLayoutB{});  // (BN,BK,BKStages)


    // G2S 
    auto g2s_src_a = block_tma_a.partition_S(ga_block_tma);           // (ABlock,G2S_ValTile_BM,G2S_ValTile_BK,BKNum)
    auto g2s_dst_a = block_tma_a.partition_D(sa);                     // (ABlock,G2S_ValTile_BM,G2S_ValTile_BK,BKStages)

    auto g2s_src_b = block_tma_b.partition_S(gb_block_tma);          
    auto g2s_dst_b = block_tma_b.partition_D(sb); 

    // G2S MulticastMask
    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;

    if constexpr(cute::is_same_v<TmaCopyAOp,SM90_TMA_LOAD_MULTICAST>){
        // Add the id of the CTA in the same row to mask
        auto cta_layout = Layout<ClusterShape>{};       // (CTAX,CTAY) -> CTA_ID_Cluster
        for(int j=0; j<size<1>(cta_layout); ++j){
            mcast_mask_a |= (uint16_t(1) << cta_layout(bx_cluster,j));
        }
    }
    

    if constexpr(cute::is_same_v<TmaCopyAOp,SM90_TMA_LOAD_MULTICAST>){
        // Add the id of the CTA in the same col to mask
        auto cta_layout = Layout<ClusterShape>{};       
        for(int i=0; i<size<0>(cta_layout); ++i){
            mcast_mask_b |= (uint16_t(1) << cta_layout(i,by_cluster));
        }
    }
    

    // Make sure the init of pipeline is visible to all all other producers and consumers before using pipeline
    if constexpr (size(ClusterShape{}) > 1) {
        cluster_arrive_relaxed();
        cluster_wait();
    }else {
        __syncthreads();
    }

    if(wg_role == 0){
        auto* tma_barrier = g2s_pipe.producer_get_barrier();
        // Producer
        for(int i=0; i<bk_num; i++){    //Load a BKStage each time
            // Acquire 
            g2s_pipe.producer_acquire(g2s_producer_pipe_state);

            int smem_st_id = g2s_producer_pipe_state.index();

            
            for

        }

    }else{
        // Consumer


        // R2S
 
        // S2G
    }

    

    

}


void SM90GemmSSW(const half_t* a,const half_t* b,half_t* c,
                                   int batch,int m,int n,int k){
    
    using TmaCopyAOp = CFG::TmaCopyAOp;
    using TmaCopyBOp = CFG::TmaCopyBOp;
    using SMemLayoutA = CFG::SMemLayoutA;
    using SMemLayoutB = CFG::SMemLayoutB;

    using BM = CFG::BM;
    using BN = CFG::BN;
    using BK = CFG::BK;
    using ClusterM = CFG::ClusterM;
    using ClusterN = CFG::ClusterN;

    // Global Tensors
    auto ga = make_tensor(a,make_layout(make_shape(m,k,batch),make_stride(k,_1{},m*k)));
    auto gb = make_tensor(b,make_layout(make_shape(n,k,batch),make_stride(k,_1{},n*k)));
    auto gc = make_tensor(c,make_layout(make_shape(m,n,batch),make_stride(n,_1{},m*n))); 

    //Init TMA
    // auto tma_a = make_tma_copy(TmaCopyAOp{},ga,SMemLayoutA{}(_,_,_0{}),Shape<BM,BK>{},ClusterN{});
    // auto tma_b = make_tma_copy(TmaCopyBOp{},gb,SMemLayoutB{}(_,_,_0{}),Shape<BN,BK>{},ClusterM{});
}
