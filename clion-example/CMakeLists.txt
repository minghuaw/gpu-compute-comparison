cmake_minimum_required(VERSION 3.26)
project(clion_example CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(clion_example main.cu)

set_target_properties(clion_example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
