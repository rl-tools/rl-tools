cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
project(test_layer_in_c_nn_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -G")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 ")

option(LAYER_IN_C_TESTS_NN_CUDA_ENABLE_CUTLASS "Enable CUTLASS" OFF)

if(LAYER_IN_C_TESTS_NN_CUDA_ENABLE_CUTLASS)
    set(CUTLASS_ENABLE_TESTS OFF)
    add_subdirectory(lib/cutlass EXCLUDE_FROM_ALL)
endif()

add_compile_options(-g -G)
add_executable(test_nn_cuda main.cu)
target_link_libraries(test_nn_cuda
        layer_in_c
#        GTest::gtest_main
        HighFive
        )
#gtest_discover_tests(test_nn_cuda)

if(LAYER_IN_C_TESTS_NN_CUDA_ENABLE_CUTLASS)
    target_link_libraries(test_nn_cuda CUTLASS)
    target_compile_definitions(test_nn_cuda PRIVATE LAYER_IN_C_TESTS_NN_CUDA_ENABLE_CUTLASS)
endif()
