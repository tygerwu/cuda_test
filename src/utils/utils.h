#pragma once

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "data_utils.h"
#include "utils.cuh"
#include "cute_utils.cuh"
#include "macros.h"
#include "print_utils.h"
#include "statistics_utils.h"
#include "float_utils.h"

using FloatVector = std::vector<float>;


template <typename T>
void RawMatmul(const T *a, const T *b, T *c, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < k; p++) {
        c[i * n + j] += a[i * k + p] * b[p * n + j];
      }
    }
  }
}


