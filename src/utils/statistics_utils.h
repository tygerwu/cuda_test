
#pragma once
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>


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