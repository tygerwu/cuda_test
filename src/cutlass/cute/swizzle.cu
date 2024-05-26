#include "cute/tensor.hpp"
#include "gtest/gtest.h"
#include "utils.h"

using namespace cute;
using T = __half;


namespace{
    constexpr int BM = 64;
    constexpr int BK = 32;
    constexpr int BKStages = 3;
    constexpr int Threads = 32;
    constexpr int GK = BK * BKStages;
    constexpr int Size = BM * BK * BKStages;
    constexpr int Bytes = Size * sizeof(T);
};


__global__ void SwizzleTestKernel(const T* input,T* output){
    using namespace cute;

    using CFG = HalfSMem::KMajorConfig<BK,BM,Threads,T,tuple<Int<BKStages>>>;

    int tid = threadIdx.x;
    auto tiled_copy = typename CFG::Copy{};
    auto thr_copy   = tiled_copy.get_slice(tid);

    __shared__ T smem[BM * BK * BKStages];

    auto gx = make_tensor(make_gmem_ptr(input),make_layout(make_shape(BM,BK,BKStages),make_stride(GK,1,BK)));
    auto sx = make_tensor(make_smem_ptr(smem),typename CFG::SMemLayout{});
    
    auto g2s_src_x = thr_copy.partition_S(gx);
    auto g2s_dst_x = thr_copy.partition_D(sx);  

    if(thread(8)){
        Print("tid_D:",tiled_copy.tidfrg_D(sx.layout()));
        Print("tid_D:",tiled_copy.tidfrg_D(sx.layout())(8,0,0));
        Print("tid_D:",tiled_copy.tidfrg_D(sx.layout()).offset());
    }

    for(int i=0; i<Threads; i++){
        if(thread(i)){
            Print("Thread:",i);
            Print("sx:",g2s_dst_x.layout());
        }
    }
    //copy(tiled_copy,g2s_src_x,g2s_dst_x);
}

TEST(cute,swizzle){
    auto hIn = CreateData<T>(Size,0,10);
    std::vector<T> hOut(Size);

    T* dIn,*dOut;
    CUDA_ERROR_CHECK(cudaMalloc(&dIn,Bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&dOut,Bytes));

    SwizzleTestKernel<<<dim3(1),dim3(Threads)>>>(dIn,dOut);

    CUDA_ERROR_CHECK(cudaFree(dIn));
    CUDA_ERROR_CHECK(cudaFree(dOut));

}


TEST(cute, swizzle_2) { 
    using T = __half;
    using BK = _32;
    using BM = Int<48>;
    using LogicalLayoutAtom = Layout<Shape<_8,_32>,Stride<_32,_1>>;
    using SMemLayoutAtom = decltype(composition(Swizzle<2,3,3>{},LogicalLayoutAtom{}));
    using SMemLayout = decltype(tile_to_shape(SMemLayoutAtom{},Shape<BM,BK>{}));

    using TiledMMa = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<_1,_1,_1>>>;
    using S2RCopy_A = decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,T>{},TiledMMa{}));

    auto tiled_s2r = S2RCopy_A{};
    std::vector<T> smem(BM{} * BK{});
    auto smem_tensor = make_tensor(make_smem_ptr(smem.data()),SMemLayout{});
    auto tiled_tv = tiled_s2r.tidfrg_S(smem_tensor);

    int tid = 9;
    auto tv = tiled_tv(tid,_,_);

    for(int i=0; i<32; i++){
        auto coord = make_coord(i,_,_);
        auto const& [sliced_layout,offset] = slice_and_offset(coord, tiled_tv.layout());
        print("thread:");
        print(i);
        print(" offset:");
        print(offset);
        print(" sliced_layout:");
        print(sliced_layout);
        print("\n");
    }


    Print("SMemLayout:",SMemLayout{});
    Print("tiled_tv:",tiled_tv);   
    Print("tv:",tv);   
}



TEST(cute, swizzle_3){
    auto layout = cute::composition(Swizzle<2,0,-2>{},
                                    Layout<Shape<_4,_4>,Stride<_4,_1>>{});
    PrintValue("layout:",layout);
    Print("layout:",layout(0));
}

TEST(cute, swizzle_4){
    for(int i=0; i<256; i++){
        int thread_idx = i % 128;
        bool is_signalling_thread_ = (thread_idx % (128 / 16)) == 0;
        auto layout = cute::composition(Swizzle<2,0,-2>{},
                                      Layout<Shape<_4,_4>,Stride<_4,_1>>{});
        int warp_idx = i / 32;
        uint32_t thread_row = warp_idx % 4;
        uint32_t thread_col = (thread_idx / 8) % 4;
        int dst_blockid_ = layout(thread_row, thread_col);

        std::cout << i << " " << thread_row << " " << thread_col << " " << dst_blockid_ << std::endl;
    }
   
}

