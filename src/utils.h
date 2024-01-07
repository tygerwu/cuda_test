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

#define UP_DIV(x, y) (((x) + (y)-1) / (y))
#define UP_ROUND(x, y) ((((x) + (y)-1) / (y)) * (y))
#define DOWN_ROUND(x, y) (((x) / (y)) * (y))
using FloatVector = std::vector<float>;

static std::vector<float> CreateFloats(int num, float beg, float end,
                                       float stride = 0.5) {
  std::vector<float> res(num);
  int endIdx = 0;
  for (int i = 0; i < num; i++) {
    float value = beg + (i - endIdx) * stride;
    if (value >= end || value <= end) {
      endIdx = i;
      value = beg + (i - endIdx) * stride;
    }
    res[i] = value;
  }
  return res;
}

template <typename T> static std::vector<T> CreateData(int num, T beg, T end) {
  int len = end - beg;
  std::vector<T> res(num);
  for (int i = 0; i < num; i++) {
    res[i] = beg + i % len;
  }
  return res;
}

template <typename From, typename To>
static std::vector<To> Convert(const std::vector<From> &src) {
  std::vector<To> dst(src.size(), 0);
  for (int i = 0; i < src.size(); i++) {
    dst[i] = static_cast<To>(src[i]);
  }
  return dst;
}

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

inline static bool FloatCompare(float a, float b,
                                float maxRelativeError = 10e-3,
                                float maxAbsoluteError = 10e-5) {
  auto nan = std::numeric_limits<float>::quiet_NaN();
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

template <typename T>
void PrintVector(const std::vector<T> &vec, const std::string &message = "") {
  std::cout << message << std::endl;
  for (const auto &item : vec) {
    std::cout << item << " ";
  }
  std::cout << std::endl;
}

template <typename T> static double Average(const std::vector<T> &times) {
  if (times.empty()) {
    return 0;
  }
  T max = *std::max_element(std::begin(times), std::end(times));
  T min = *std::min_element(std::begin(times), std::end(times));
  if (times.size() <= 2) {
    return (max + min) / 2;
  }
  T sum =
      std::accumulate(std::begin(times), std::end(times), T(0), std::plus<T>());
  return (sum - max - min) / double(times.size() - 2);
}

class Table2D {
  using ROW = std::vector<float>;

public:
  Table2D() = default;
  Table2D(const std::vector<std::string> &heads) : heads_(heads) {}
  void SetHeads(const std::vector<std::string> &heads) { heads_ = heads; }
  void AddRow(const std::vector<float> &row) { rows_.push_back(row); }
  void Print(int eleLen = 5, int columnSpace = 5) {
    std::stringstream ss;
    std::string columnSpaceStr(columnSpace, ' ');
    for (const auto &head : heads_) {
      ss << head << columnSpaceStr;
    }
    ss << "\n";
    for (const auto &row : rows_) {
      for (int i = 0; i < row.size(); i++) {
        std::string eleStr = std::to_string(row[i]);
        eleStr.resize(eleLen, ' ');
        eleStr.resize(heads_[i].length());
        ss << eleStr << columnSpaceStr;
      }
      ss << "\n";
    }
    std::cout << ss.str() << std::endl;
  }
  void ExportToCSV(const std::string &filePath) {
    std::stringstream ss;
    for (int i = 0; i < heads_.size(); i++) {
      ss << heads_[i];
      if (i != heads_.size() - 1) {
        ss << ",";
      }
    }
    ss << "\n";
    for (const auto &row : rows_) {
      for (int i = 0; i < row.size(); i++) {
        ss << std::to_string(row[i]);
        if (i != row.size() - 1) {
          ss << ",";
        }
      }
      ss << "\n";
    }
    std::fstream f(filePath, std::ios::out);
    assert(f.good());
    f << ss.str();
    f.close();
  }

public:
  std::vector<std::string> heads_;
  // Y,X
  std::vector<ROW> rows_;
};
