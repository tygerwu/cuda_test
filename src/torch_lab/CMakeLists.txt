cmake_minimum_required(VERSION 3.0.0)
project(torch_lab VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)


# /torch_install_dir/share/cmake/Torch
set(Torch_DIR ${Torch_DIR})
if(NOT Torch_DIR)
    message(FATAL_ERROR "Torch_DIR is required")
endif()

find_package(Torch REQUIRED)


add_executable(torch_lab main.cpp)

target_include_directories(torch_lab PRIVATE 
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(torch_lab 
    ${TORCH_LIBRARIES}
)

add_subdirectory(custom_op)
