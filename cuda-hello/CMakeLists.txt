cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(hello LANGUAGES CXX CUDA)

enable_language(CUDA)

add_executable(hello src/hello.cu)