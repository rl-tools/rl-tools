cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
project(example_td3_pendulum_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

add_compile_options(-g -G)
add_executable(example_td3_pendulum_cuda main.cu)
target_include_directories(example_td3_pendulum_cuda PRIVATE ../../include)
target_include_directories(example_td3_pendulum_cuda PRIVATE ${PROJECT_SOURCE_DIR})

## set maximum optimization level for cpu
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -O0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0")

#add_subdirectory(lib/highfive)

#set_target_properties(example_td3_pendulum_cuda PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

#add_executable(multirotor_cuda_test test.cu)

#target_link_libraries(multirotor_cuda_test HighFive)
