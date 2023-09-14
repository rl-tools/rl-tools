cmake_minimum_required(VERSION 3.23)
project(multirotor_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

set(CMAKE_CUDA_ARCHITECTURES "native")

add_executable(rl_environments_multirotor_cuda_benchmark benchmark.cu)

target_link_libraries(rl_environments_multirotor_cuda_benchmark backprop_tools)
BACKPROP_TOOLS_TAG_IS_CUDA(rl_environments_multirotor_cuda_benchmark)
