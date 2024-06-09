cmake_minimum_required(VERSION 3.0.0)
project(cuda_lab VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake) 
include(googletest)
include(gflags)


# Settings
set(CUTLASS_ROOT ${PROJECT_SOURCE_DIR}/third_lib/cutlass CACHE STRING "Cutlass root dir")
set(CUDA_HOME /usr/local/cuda CACHE STRING "Cuda home")
set(ENABLE_HOPEER OFF CACHE BOOL "Hopper")

# NVRTC Settings
add_definitions(-DNVRTC_GET_TYPE_NAME=1 -DJITIFY_PRINT_PTX=1)

# Cutlass Settings
set(CUTLASS_INCLUDE_DIRS 
    ${CUTLASS_ROOT}/include
    ${CUTLASS_ROOT}/tools/util/include
)


# /torch_install_dir/share/cmake/Torch
set(Torch_DIR ${Torch_DIR})
if(NOT Torch_DIR)
    message(FATAL_ERROR "Torch_DIR is required")
endif()
find_package(Torch REQUIRED)

if(${ENABLE_HOPEER})
    set(CMAKE_CUDA_ARCHITECTURES 86 90a)
else()
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()


file(GLOB_RECURSE srcs 
    ${PROJECT_SOURCE_DIR}/src/exercise/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/exercise/*.cu
    ${PROJECT_SOURCE_DIR}/src/optimization/gemm/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/optimization/gemm/*.cu 
    ${PROJECT_SOURCE_DIR}/src/optimization/float4/*.cu 
    ${PROJECT_SOURCE_DIR}/src/cutlass/*.cu
    ${PROJECT_SOURCE_DIR}/src/op/attention/*.cu
)

if(ENABLE_HOPEER)
    file(GLOB_RECURSE hopper_srcs 
        ${PROJECT_SOURCE_DIR}/src/hopper/*.cu 
    )
    set(srcs ${srcs} ${hopper_srcs})
endif()


add_executable(cuda_lab_test main.cpp ${srcs} src/exercise/nvrtc/jitify/jitify_header.cu)
target_include_directories(cuda_lab_test PRIVATE 
    ${gmock_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}/include
    ${gflags_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/utils
    ${CUTLASS_INCLUDE_DIRS}
    ${CUDA_HOME}/include
    ${TORCH_INCLUDE_DIRS}
) 

target_link_directories(
    cuda_lab_test PRIVATE 
    ${CUDA_HOME}/lib64
    ${CUDA_HOME}/lib64/stubs
)

target_link_libraries(cuda_lab_test 
    gmock
    gtest
    ${gflags_LIBRARIES}
    libcuda.so
    libnvrtc.so
    cublas
    cublasLt
    curand
    ${TORCH_LIBRARIES}
)
#set_target_properties(cuda_lab_test PROPERTIES CUDA_ARCHITECTURES "86")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_lab_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

target_compile_options(cuda_lab_test PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>
)

