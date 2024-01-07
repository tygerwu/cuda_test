#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void broadcast_within_warp(int arg) {
  // init : 0,1,2,3....31
  int value = threadIdx.x & 0x1F;
  // 'steal' varible from lane0
  value = __shfl_sync(0xFFFFFFFF, value, 0, 8);
  printf("Thread id : %d. Value : %d \n", threadIdx.x, value);
}

__global__ void accumulate() {
  int lane_id = (threadIdx.x & 0x1F & 7);
  int value = lane_id;
  for (int i = 1; i <= 4; i *= 2) {
    int n = __shfl_up_sync(0xFFFFFFFF, value, i, 8);
    if (lane_id >= i) {
      value += n;
      if (threadIdx.x <= 7) {
        printf("Thread id : %d, Lane id: %d, N: %d, Value : %d \nm ",
               threadIdx.x, lane_id, n, value);
      }
    }
  }
}

__global__ void warp_reduction() {
  int lane_id = threadIdx.x & 0x1F;
  int value = lane_id;
  // Butterfly reduction
  for (int i = 16; i >= 1; i /= 2) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, i, 32);
  }
  printf("Thread id : %d. N:% d, Value : %d \n", threadIdx.x, value);
}

TEST(warp_shuffle, test0) {
  broadcast_within_warp<<<1, 32>>>(123);
  cudaDeviceSynchronize();
}

TEST(warp_shuffle, test1) {
  accumulate<<<1, 32>>>();
  cudaDeviceSynchronize();
}