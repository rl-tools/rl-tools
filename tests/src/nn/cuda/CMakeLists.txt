cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
project(test_layer_in_c_nn_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_compile_options(-g -G)
add_executable(test_nn_cuda main.cu)
target_link_libraries(test_nn_cuda
        layer_in_c
#        GTest::gtest_main
        HighFive
        )
#gtest_discover_tests(test_nn_cuda)

## set maximum optimization level for cpu