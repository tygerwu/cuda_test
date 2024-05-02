#pragma once 
#include <vector>
#include <string>
#include <iostream>

template <typename T>
void PrintVector(const std::vector<T> &vec, const std::string &message = "") {
  std::cout << message << std::endl;
  for (const auto &item : vec) {
    std::cout << item << " ";
  }
  std::cout << std::endl;
}

