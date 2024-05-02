#pragma once
#include <vector>
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