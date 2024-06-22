

file(GLOB_RECURSE op_srcs 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_library(
    custom_op SHARED
    ${op_srcs}
)

target_include_directories(custom_op PRIVATE 
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(custom_op 
    ${TORCH_LIBRARIES}
)