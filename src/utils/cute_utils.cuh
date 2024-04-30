#pragma once
#include "cute/tensor.hpp" 
#include <cuda.h>




template<typename T>
CUTE_HOST_DEVICE 
void Print(const char* msg,const T& obj){
    cute::print(msg);
    cute::print(obj);
    cute::print("\n");
}


template<typename T>
CUTE_HOST_DEVICE
void PrintIden(const char* msg,const T& obj){
    cute::print(msg);
    cute::print(obj);
    cute::print("\n");

    for(int i=0; i<size(obj); i++){
        cute::print(obj(i));
        cute::print(" ");
    }
    cute::print("\n");
}


template<typename T>
CUTE_HOST_DEVICE
void PrintValue(const char* msg,const T& obj){
    cute::print(msg);
    cute::print(obj);
    cute::print("\n");

    for(int i=0; i<size(obj); i++){
       printf("%6.1f",static_cast<float>(obj(i)));
    }
    cute::print("\n");
}