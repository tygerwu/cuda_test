
#include "cute/tensor.hpp"
#include "gtest/gtest.h"
#include "utils.h"


using namespace cute;


TEST(cute,copy1){
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                            Layout<Shape<_2, _2, _1>>, // 2x2x1 thread
                            Tile<_32, Int<32>, _16>>;
    auto tiled_mma = TiledMma{};
    auto tiled_copy =
          make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, tiled_mma);
    
    auto layout = Layout<Shape<Int<96>,Int<32>>,Stride<_32,_1>>{};

    auto ref_S = Layout<Shape<Shape<_32,_16,_1>>,Stride<Stride<_1,_32>,_0>>{};

    Print("tiled_copy:",tiled_copy);
    Print("tidfrg_S:",tiled_copy.tidfrg_S(layout));
    Print("tidfrg_S:",tiled_copy.get_layoutS_TV());
    Print("tidfrg_MN:",tiled_copy.get_layoutS_MN());

    Print("tidfrg_D:",tiled_copy.tidfrg_D(layout));
    Print("tidfrg_D:",tiled_copy.get_layoutD_TV());
}

