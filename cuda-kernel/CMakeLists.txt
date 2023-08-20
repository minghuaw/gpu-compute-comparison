cmake_minimum_required(VERSION 3.16)
project(cuda_kernel LANGUAGES CXX CUDA)

set(CMAKE_C_COMPILER nvcc)
set(CMAKE_CXX_COMPILER nvcc)
set(CMAKE_CUDA_STANDARD 11)

add_executable(cuda_kernel main.cu
        kernels/matmul.cuh
)

set_target_properties(cuda_kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_kernel -lcublas)