#include "utils.cuh"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>

// transfer float4

template <typename T>
__forceinline__ __device__ void transpose2d(const T __restrict__ *in, T *out,
                                            int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      out[j * cols + i] = in[i * cols + j];
    }
  }
}

// Requirements:
//  Cols is a multiple of threadNum*4
//  Rows is a multiple of 4
template <size_t MC, size_t KC, size_t MR, size_t TMN>
__device__ inline void SGemmPackA(const float *__restrict__ in, float *out) {
  // static_assert(MC % 4 == 0, "Invalid MC");
  // static_assert(KC % 4 == 0, "Invalid KC");
  // static_assert(MR % 4 == 0, "Invalid MR");

  // const size_t pack_block_size = TMN * 4;
  // const size_t pack_block_num = rows * cols / pack_block_size;
  // const size_t pack_block_cols = UP_DIV(cols, pack_block_size);
  // const size_t pack_block_rows = pack_block_num / pack_block_cols;
  // const size_t row_stride = pack_block_cols * pack_block_num;

  // constexpr size_t PACK_BLOCK_SIZE = TMN * 4;
  // constexpr size_t PACK_BLOCK_NUM = MC * KC / PACK_BLOCK_SIZE;
}

template <size_t MC, size_t KC, size_t NC, size_t MR, size_t NR>
__global__ void SGemm(const float *__restrict__ A, const float *__restrict__ B,
                      float *__restrict__ C, int M, int N, int K) {
  static_assert(MC % 4 == 0, "Invalid MC");
  static_assert(KC % 4 == 0, "Invalid KC");
  static_assert(NC % 4 == 0, "Invalid NC");

  static_assert(MR % 4 == 0, "Invalid MR");
  static_assert(NR % 4 == 0, "Invalid NR");

  // ThreadNum along m in ThreadBlcok
  constexpr size_t TM = MC / MR;
  constexpr size_t TN = NC / NR;
  constexpr size_t TMN = TM * TN;

  // Shared Memory
  //  MacroBlock + Double Buffer
  __shared__ float a_shares[2][MC][KC];
  __shared__ float b_shares[2][KC][NC];

  // Reg
  //  Tile of A(B)MicroBlock + Double Buffer
  float a_regs[2][MR];
  float b_regs[2][NR];
  //  CMicroBlock + Double Buffer
  float c_regs[2][MR][NR];

  // Locate the MacroBlock
  const int by = blockIdx.y;
  const int bx = blockIdx.x;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int tOffset = ty * TN + tx;

  const float *a_block_start = A + by * MC * K;
  const float *b_block_start = B + bx * NC;
  float *c_blcok_start = C + by * MC * N + bx * NC;

  const size_t a_pack_block_size = TMN * 4;
  const size_t a_pack_block_num = MC * KC / a_pack_block_size;
  const auto *a_pack_block_in = a_block_start + tOffset * 4;
  float a_pack_regs[4];

  SGemmPackA<MC, KC, MR, TMN>(a_block_start, a_shares);

  // SGemmPackB<MC, KC, MR, TMN>(a_block_start, a_shares);
}

// // K: ldA
// // N: ldB
// template <const int BM, const int BK, const int BN, const int RM, const
// int RN,
//           const bool ENABLE_DOUBLE_BUFFER>
// __global__ void Sgemm(const float *__restrict__ A, const float
// *__restrict__ B,
//                       float *__restrict__ C, int M, int N, int K) {
//   constexpr int THREADS_ALONG_BLOCK_X = BN / RN;
//   constexpr int THREADS_ALONG_BLOCK_Y = BM / RM;
//   constexpr int THREADS_PER_BLOCK = THREAD_ALONG_BLOCK_N *
//   THREAD_ALONG_BLOCK_M;

//   const int tid = threadIdx.y * THREADS_ALONG_BLOCK_X + threadIdx.x;

//   // declare resouces:
//   //  1.shared memory for macro block
//   //  2.double buffer
//   __shared__ float a_shares[2][BM][BK];
//   __shared__ float c_shares[2][BK][BN];

//   //  2.registers for micro block
//   float a_regs[2][RM];
//   float b_regs[2][RN];
//   float c_regs[RM][RN] = {0};

//   //  3.registers for stashing global memory when transfering macro block
//   float a_regs_stash[4];

//   // Fill first buffer
//   //  1. Load A macro block from global memory to shared memory
//   //     RM x BK floats in total need to be fetched and at every time a
//   //  threa fetches 4 floats
//   constexpr int a_threads_x = BK / 4;
//   const int a_x = tid % a_threads_x;
//   const int a_y = tid / a_threads_x;
//   const float *a_global = nullptr;
//   float *a_share = nullptr;
//   const int a_size = (BM * BK / (4 * THREADS_PER_BLOCK);
//   for (int i = 0; i <a_size ; i++) {
//     // load into regs
//     ST_FP4(&a_regs_stash[0], LD_FP4((a_global + OFFSET(a_y + i, a_x,
//     K))));
//     // transpose and load
//     a_share[i] = a_regs_stash[0];
//     a_share[a_size + i] = a_regs_stash[1];
//     a_share[2 * a_size + i] = a_regs_stash[2];
//     a_share[3 * a_size + i] = a_regs_stash[3];
//   }

//   // 2. Load B macro block
//   constexpr int b_threads_x = BN / 4;
//   const int b_x = tid % b_threads_x;
//   const int b_y = tid / b_threads_x;
//   const float *b_global = nullptr;
//   float *b_share = nullptr;
//   const int b_size = (BK * BN / (4 * THREADS_PER_BLOCK);
//   for (int i = 0; i <b_size ; i++) {
//     // B is not transposed
//     ST_FP4(&b_share[i], LD_FP4((b_global + OFFSET(b_y + i, b_x, N))));
//   }
//    __syncthreads();

//   // 2. Load A micro block
//   for(int i=0;i<a_size;a+=4){

//   }
// }