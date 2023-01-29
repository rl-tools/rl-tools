cmake_minimum_required(VERSION 3.16)
project(layer_in_c)

set(CMAKE_CXX_STANDARD 17)

option(LAYER_IN_C_ENABLE_TESTS "Build the unit tests when enabled." off)

# To obtain MKL: sudo apt install intel-oneapi-mkl-devel
option(LAYER_IN_C_BACKEND_ENABLE_MKL "Enable the MKL gemm library, only available on Intel CPUs" OFF)
option(LAYER_IN_C_BACKEND_ENABLE_ACCELERATE "Enable the Accelerate gemm library, only available on Apple CPUs" OFF)
option(LAYER_IN_C_BACKEND_ENABLE_CUDA "Enable Nvidia CUDA" OFF)

add_library(layer_in_c INTERFACE)
target_include_directories(layer_in_c INTERFACE include)


# dependencies
if(LAYER_IN_C_BACKEND_ENABLE_MKL)
    find_package(MKL REQUIRED)
    add_compile_definitions(LAYER_IN_C_BACKEND_ENABLE_MKL)
endif()

if(LAYER_IN_C_BACKEND_ENABLE_ACCELERATE)
    find_package(BLAS REQUIRED)
    add_compile_definitions(LAYER_IN_C_BACKEND_ENABLE_ACCELERATE)
endif()

if (LAYER_IN_C_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()


add_subdirectory(examples)