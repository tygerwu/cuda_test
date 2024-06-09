#pragma once
#include <algorithm>
#include <iostream>
#include <random>


inline static bool FloatCompare(float a, float b,
                                float maxRelativeError = 10e-3,
                                float maxAbsoluteError = 10e-5) {
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  if (std::isinf(a) && std::isinf(b) && (a * b > 0)) {
    return true;
  }
  if (fabs(a - b) < maxAbsoluteError)
    return true;
  float relativeError = fabs((a - b) / (std::max(fabs(a), fabs(b))));
  return relativeError <= maxRelativeError;
}

static bool FloatsCompare(const float *A, const float *B, int len,
                          float maxRelativeError = 10e-3,
                          float maxAbsoluteError = 10e-5) {
  for (int i = 0; i < len; i++) {
    if (!FloatCompare(A[i], B[i], maxRelativeError, maxAbsoluteError)) {
      std::cout << "--------------------------" << std::endl;
      std::cout << "Fail to compare floats :"
                << "(" << A[i] << "!=" << B[i] << ")"
                << "( index = " << i << " )" << std::endl;
      return false;
    }
  }
  return true;
}