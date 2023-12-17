cmake_minimum_required(VERSION 3.0.0)
project(cuda_lab VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake) 
include(googletest)
include(gflags)

add_definitions(-DNVRTC_GET_TYPE_NAME=1 -DJITIFY_PRINT_PTX=1)

file(GLOB_RECURSE srcs 
    ${PROJECT_SOURCE_DIR}/src/exercise/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/exercise/*.cu
    ${PROJECT_SOURCE_DIR}/src/optimization/gemm/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/optimization/gemm/*.cu 
    ${PROJECT_SOURCE_DIR}/src/optimization/float4/*.cu 
) 
add_executable(cuda_lab_test main.cpp ${srcs} src/exercise/nvrtc/jitify/jitify_header.cu)
target_include_directories(cuda_lab_test PRIVATE 
    ${gmock_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}/include
    ${gflags_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
) 

target_link_libraries(cuda_lab_test 
    gmock
    gtest
    ${gflags_LIBRARIES}
    /usr/local/cuda/lib64/stubs/libcuda.so
    /usr/local/lib64/libnvrtc.so
    cublas
    curand
)
set_target_properties(cuda_lab_test PROPERTIES CUDA_ARCHITECTURES "86")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_lab_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
