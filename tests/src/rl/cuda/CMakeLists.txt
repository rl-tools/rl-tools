cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_ARCHITECTURES 75)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
project(test_layer_in_c_rl_cuda CUDA)

find_package(CUDAToolkit REQUIRED)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

function(LAYER_IN_C_TESTS_RL_CUDA_ADD_DEBUG_DEFINITIONS TARGET)
    #    if(LAYER_IN_C_DEBUG)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_CONTAINER_CHECK_BOUNDS)
#        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC)
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_CONTAINER_MALLOC_INIT_NAN)
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_DEVICE_CUDA_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_DEVICE_CUDA_SYNCHRONIZE_STATUS_CHECK)
#        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_DEVICE_CUDA_CHECK_BOUNDS)
    endif()
endfunction()

add_executable(
        test_rl_cuda
        rl.cu
)
target_link_libraries(
        test_rl_cuda
        PRIVATE
        layer_in_c
        GTest::gtest_main
        CUDA::cublas
        HighFive
)
LAYER_IN_C_TESTS_RL_CUDA_ADD_DEBUG_DEFINITIONS(test_rl_cuda)
LAYER_IN_C_TARGET_LINK_BLAS(test_rl_cuda)
gtest_discover_tests(test_rl_cuda)

add_executable(
        test_rl_cuda_td3_full_training
        td3_full_training.cu
)
target_link_libraries(
        test_rl_cuda_td3_full_training
        PRIVATE
        layer_in_c
        GTest::gtest_main
        CUDA::cublas
        HighFive
        tensorboard_logger
)
LAYER_IN_C_TESTS_RL_CUDA_ADD_DEBUG_DEFINITIONS(test_rl_cuda_td3_full_training)
LAYER_IN_C_TARGET_LINK_BLAS(test_rl_cuda_td3_full_training)
gtest_discover_tests(test_rl_cuda_td3_full_training)

add_executable(
        test_rl_cuda_environments_multirotor
        environments/multirotor.cu
)
target_link_libraries(
        test_rl_cuda_environments_multirotor
        PRIVATE
        layer_in_c
        GTest::gtest_main
        CUDA::cublas
        HighFive
        tensorboard_logger
)
LAYER_IN_C_TESTS_RL_CUDA_ADD_DEBUG_DEFINITIONS(test_rl_cuda_environments_multirotor)
LAYER_IN_C_TARGET_LINK_BLAS(test_rl_cuda_environments_multirotor)
gtest_discover_tests(test_rl_cuda_environments_multirotor)
