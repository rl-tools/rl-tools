cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
project(test_layer_in_c_nn_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

add_executable(
        test_nn_cuda_benchmark
        ../benchmark.cpp
)
target_link_libraries(
        test_nn_cuda_benchmark
        PRIVATE
        layer_in_c
        GTest::gtest_main
        Eigen3::Eigen
)
gtest_discover_tests(test_nn_cuda_benchmark)
