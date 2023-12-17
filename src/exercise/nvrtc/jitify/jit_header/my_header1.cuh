#pragma once

template <typename T>
__device__ inline T square(T x) {
  return x * x;
}
