#pragma once
#include <vector>
#include <random>
#include <algorithm>

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



template <typename T> 
static std::vector<T> CreateData(int num, T beg, T end) {
  int len = end - beg;
  std::vector<T> res(num);
  T val = beg;
  for (int i = 0; i < num; i++) {
    res[i] = val;
    ++val; 
    if(val >= end){
        val = end;
    }
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



template<typename T>
static std::vector<T> RandomFloats(int size, float min=-10,float max=10) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto rg = std::uniform_real_distribution<>(min,max);
  std::vector<float> tmp(size);
  std::generate(tmp.begin(),tmp.end(),[&]{return rg(rng);});
  return Convert<float,T>(tmp);
}