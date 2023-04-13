cmake_minimum_required(VERSION 3.16)
project(layer_in_c)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
message("C++ Compiler: ${CMAKE_CXX_COMPILER}")

option(LAYER_IN_C_ENABLE_TESTS "Build the unit tests when enabled." off)

# To obtain MKL: sudo apt install intel-oneapi-mkl-devel
option(LAYER_IN_C_BACKEND_ENABLE_MKL "Enable the MKL gemm library, only available on Intel CPUs" OFF)
option(LAYER_IN_C_BACKEND_ENABLE_ACCELERATE "Enable the Accelerate gemm library, only available on Apple CPUs" OFF)
option(LAYER_IN_C_BACKEND_ENABLE_CUDA "Enable Nvidia CUDA" OFF)
option(LAYER_IN_C_BACKEND_ENABLE_MUJOCO "Enable MuJoCo" OFF)


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

function(LAYER_IN_C_TARGET_LINK_BLAS target)
    if(LAYER_IN_C_BACKEND_ENABLE_MKL)
        message(STATUS "Linking MKL to ${target}")
        target_link_libraries(${target} PRIVATE MKL::MKL)
    endif()
    if(LAYER_IN_C_BACKEND_ENABLE_ACCELERATE)
        message(STATUS "Linking ACCELERATE to ${target}")
        target_link_libraries(${target} PRIVATE BLAS)
    endif()
endfunction()

if(LAYER_IN_C_BACKEND_ENABLE_CUDA)
    set(LAYER_IN_C_BACKEND_CUDA_ARCHITECTURES "all")
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA version: ${CUDA_VERSION}")
    message(STATUS "CUDA include path: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA cublas library: ${CUDA_CUBLAS_LIBRARIES}")
    add_compile_definitions(LAYER_IN_C_BACKEND_ENABLE_CUDA)
endif()

if(LAYER_IN_C_RL_ENVIRONMENTS_ENABLE_MUJOCO)
    set(MUJOCO_BUILD_EXAMPLES OFF)
    set(MUJOCO_BUILD_SIMULATE OFF)
    set(MUJOCO_BUILD_TESTS OFF)
    set(MUJOCO_TEST_PYTHON_UTIL OFF)
    add_subdirectory(lib/mujoco)
endif()


if (LAYER_IN_C_ENABLE_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
