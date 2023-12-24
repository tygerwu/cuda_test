
#pragma once
#include "utils.cuh"
#include "utils.h"
#include <cuda_runtime.h>

#include <stdio.h>

template <size_t MC, size_t KC, size_t NC, size_t MR, size_t NR, size_t WY,
          size_t WX>
static __global__ void CudaSGemmMacroMicroBufferImplV2(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int M, int N, int K, int ldk, int ldn) {

  static_assert(MR % 4 == 0, "Invalid MR");
  static_assert(NR % 4 == 0, "Invalid NR");
  static_assert(NC % (WX * NR) == 0, "Invalid NC");
  static_assert(MC % (WY * MR) == 0, "Invalid MC");
  static_assert(KC % 4 == 0, "Invalid KC");

  constexpr size_t TX = NC / NR;
  constexpr size_t TY = MC / MR;
  constexpr size_t TXY = TX * TY;
  static_assert(TXY % WARP_SIZE == 0, "Invalid ThreadBlock");

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int tid = tid_y * blockDim.x + tid_x;

  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int wid_x = tid_x / WX;
  int wid_y = tid_y / WY;
  int tid_x_in_w = tid_x % WX;
  int tid_y_in_w = tid_y % WY;

  // SMem
  constexpr size_t A_PAD = 4;
  constexpr size_t A_MICRO_SIZE = UP_ROUND(MR * KC, 32) + A_PAD;
  constexpr size_t A_WARP_MICRO_SIZE = WY * A_MICRO_SIZE;
  constexpr size_t A_MICRO_NUM = MC / MR;
  constexpr size_t A_MACRO_SMEM_SIZE = A_MICRO_NUM * A_MICRO_SIZE;

  constexpr size_t B_PAD = 4;
  constexpr size_t B_MICRO_SIZE = UP_ROUND(KC * NR, 32) + B_PAD;
  constexpr size_t B_WARP_MICRO_SIZE = WX * B_MICRO_SIZE;
  constexpr size_t B_MICRO_NUM = NC / NR;
  constexpr size_t B_MACRO_SMEM_SIZE = B_MICRO_NUM * B_MICRO_SIZE;

  __shared__ float a_macro_smem[2 * A_MACRO_SMEM_SIZE];
  __shared__ float b_macro_smem[2 * B_MACRO_SMEM_SIZE];

  // Regs for MicroKernel
  float a_regs[2 * MR];
  float b_regs[2 * NR];
  float c_acc_regs[MR * NR] = {0};

  // Regs for MacroKernel
  constexpr size_t A_MACRO_F4_NUM = MC * KC / 4;
  constexpr size_t KC_F4_NUM = KC / 4;
  //  Regs required to store AMacroBlock per thread
  static_assert(A_MACRO_F4_NUM % TXY == 0, "Invalid AMacroBlockSize");
  constexpr size_t A_MACRO_REG_F4_NUM = UP_DIV(A_MACRO_F4_NUM, TXY);
  float a_macro_regs[A_MACRO_REG_F4_NUM * 4];

  constexpr size_t B_MACRO_F4_NUM = KC * NC / 4;
  constexpr size_t NC_F4_NUM = NC / 4;
  constexpr size_t NR_F4_NUM = NR / 4;
  static_assert(B_MACRO_F4_NUM % TXY == 0, "Invalid BMacroBlockSize");
  constexpr size_t B_MACRO_REG_F4_NUM = UP_DIV(B_MACRO_F4_NUM, TXY);
  float b_macro_regs[B_MACRO_REG_F4_NUM * 4];

  size_t a_macro_f4_x0 = tid % KC_F4_NUM;
  size_t a_macro_f4_y0 = tid / KC_F4_NUM;
  constexpr size_t A_MACRO_F4_Y_STRIDE = TXY / KC_F4_NUM;

  size_t b_macro_f4_x0 = tid % NC_F4_NUM;
  size_t b_macro_f4_y0 = tid / NC_F4_NUM;
  constexpr size_t B_MACRO_F4_Y_STRIDE = TXY / NC_F4_NUM;

  // Helper lambdas
  auto SMemToReg = [&a_regs, &b_regs, wid_x, wid_y, tid_x_in_w,
                    tid_y_in_w](int reg_buf_id, int smem_buf_id, int p) {
#pragma unroll
    for (int i = 0; i < NR; i += 4) {
      *FP4_PTR(b_regs + reg_buf_id * NR + i) = *CONST_FP4_PTR(
          b_macro_smem + smem_buf_id * B_MACRO_SMEM_SIZE +
          wid_x * B_WARP_MICRO_SIZE + tid_x_in_w * B_MICRO_SIZE + p * NR + i);
    }
#pragma unroll
    for (int i = 0; i < MR; i += 4) {
      *FP4_PTR(a_regs + reg_buf_id * MR + i) = *CONST_FP4_PTR(
          a_macro_smem + smem_buf_id * A_MACRO_SMEM_SIZE +
          wid_y * A_WARP_MICRO_SIZE + tid_y_in_w * A_MICRO_SIZE + p * MR + i);
    }
  };

  auto OuterProduct = [&a_regs, &b_regs, &c_acc_regs](int buf_id) {

#pragma unroll
    for (int i = 0; i < MR; i++) {
#pragma unroll
      for (int j = 0; j < NR; j++) {
        c_acc_regs[i * NR + j] +=
            a_regs[buf_id * MR + i] * b_regs[buf_id * NR + j];
      }
    }
  };

  auto MicroKernel = [&a_regs, &b_regs, &c_acc_regs, &SMemToReg,
                      &OuterProduct](int smem_buf_id) {
    // Prefetch
    int st_buf_id = 0;
    int ld_buf_id = 0;
    SMemToReg(st_buf_id, smem_buf_id, 0);
    st_buf_id ^= 1; // XOR
// MicroKernel
#pragma unroll
    for (int p = 1; p < KC; p++) {
      // Prefetch
      SMemToReg(st_buf_id, smem_buf_id, p);
      st_buf_id ^= 1;

      OuterProduct(ld_buf_id);
      ld_buf_id ^= 1;
    }
    OuterProduct(ld_buf_id);
  };

  // Load MacroBlock from GMem to Reg
  auto GMemToReg = [&a_macro_regs, &b_macro_regs, A, B, tid, bid_y, bid_x, ldn,
                    ldk, a_macro_f4_x0, a_macro_f4_y0, b_macro_f4_x0,
                    b_macro_f4_y0](int pc) {
    // Load A
    const float *a_macro_gmem = A + bid_y * MC * ldk + pc;
#pragma unroll
    for (int i = 0; i < A_MACRO_REG_F4_NUM; i++) {
      auto x = a_macro_f4_x0;
      auto y = a_macro_f4_y0 + i * A_MACRO_F4_Y_STRIDE;
      *FP4_PTR(a_macro_regs + i * 4) =
          *CONST_FP4_PTR(a_macro_gmem + y * ldk + x * 4);
    }

    // Load B
    const float *b_macro_gmem = B + pc * ldn + bid_x * NC;
#pragma unroll
    for (int i = 0; i < B_MACRO_REG_F4_NUM; i++) {
      auto x = b_macro_f4_x0;
      auto y = b_macro_f4_y0 + i * B_MACRO_F4_Y_STRIDE;
      *FP4_PTR(b_macro_regs + i * 4) =
          *CONST_FP4_PTR(b_macro_gmem + y * ldn + x * 4);
    }
  };
  // Load MacroBlock from Reg to SMem
  auto RegToSMem = [&a_macro_regs, &b_macro_regs, tid, a_macro_f4_x0,
                    a_macro_f4_y0, b_macro_f4_x0, b_macro_f4_y0](int buf_id) {
// Load A
#pragma unroll
    for (int i = 0; i < A_MACRO_REG_F4_NUM; i++) {
      auto x = a_macro_f4_x0;
      auto y = a_macro_f4_y0 + i * A_MACRO_F4_Y_STRIDE;
      int mr_id = y / MR;
      int mr_offset = y % MR;

      float4 tmp = *CONST_FP4_PTR(a_macro_regs + i * 4);

      float *ptr = a_macro_smem + buf_id * A_MACRO_SMEM_SIZE +
                   mr_id * A_MICRO_SIZE + (x * 4) * MR + mr_offset;
      *(ptr) = tmp.x;
      *(ptr + MR) = tmp.y;
      *(ptr + 2 * MR) = tmp.z;
      *(ptr + 3 * MR) = tmp.w;
    }

// Load B
#pragma unroll
    for (int i = 0; i < B_MACRO_REG_F4_NUM; i++) {
      auto x = b_macro_f4_x0;
      auto y = b_macro_f4_y0 + i * B_MACRO_F4_Y_STRIDE;
      int nr_id = x / NR_F4_NUM;
      int nr_offset = x % NR_F4_NUM;

      *FP4_PTR(b_macro_smem + buf_id * B_MACRO_SMEM_SIZE +
               nr_id * B_MICRO_SIZE + y * NR + nr_offset * 4) =
          *CONST_FP4_PTR(b_macro_regs + i * 4);
    }
  };

  int smem_buf_st_id = 0;
  int smem_buf_ld_id = 0;
  GMemToReg(0);
  RegToSMem(smem_buf_st_id);
  smem_buf_st_id ^= 1;

  __syncthreads();
  size_t pc = 0;
  do {
    // Look ahead
    //  Prefetch
    if (pc + KC < K) {
      GMemToReg(pc + KC);
    }
    // MicroKernel for last prefetched lata
    // This step is alaways performed
    MicroKernel(smem_buf_ld_id);
    smem_buf_ld_id ^= 1;

    if (pc + KC < K) {
      RegToSMem(smem_buf_st_id);
      smem_buf_st_id ^= 1;
      __syncthreads();
    }
    pc += KC;
  } while (pc < K);

  // Store C
  float *c_macro_gmem = C + bid_y * MC * ldn + bid_x * NC;
#pragma unroll
  for (int i = 0; i < MR; i++) {
#pragma unroll
    for (int j = 0; j < NR; j++) {
      // py: tid_y * MR + i
      // px: tid_x * NR + j
      c_macro_gmem[(tid_y * MR + i) * ldn + tid_x * NR + j] =
          c_acc_regs[i * NR + j];
    }
  }
}

template <size_t MC, size_t KC, size_t NC, size_t MR, size_t NR, size_t WY,
          size_t WX>
static void CudaSGemmMacroMicroBufferV2(const float *A, const float *B,
                                        float *C, int M, int N, int K) {
  dim3 dimBlock(NC / NR, MC / MR);
  dim3 dimGrid(N / NC, M / MC);
  CudaSGemmMacroMicroBufferImplV2<MC, KC, NC, MR, NR, WY, WX>
      <<<dimGrid, dimBlock>>>(A, B, C, M, N, K, K, N);
}