cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_ARCHITECTURES 75)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
project(test_layer_in_c_nn_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

function(LAYER_IN_C_TESTS_NN_CUDA_ADD_DEBUG_DEFINITIONS TARGET)
    #    if(LAYER_IN_C_DEBUG)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_CONTAINER_CHECK_BOUNDS)
#        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC)
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_CONTAINER_MALLOC_INIT_NAN)
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT)
    endif()
endfunction()

add_executable(
        test_nn_cuda_basics
        cuda_basics.cu
)
target_link_libraries(
        test_nn_cuda_basics
        PRIVATE
        layer_in_c
        GTest::gtest_main
        cublas
)
LAYER_IN_C_TESTS_NN_CUDA_ADD_DEBUG_DEFINITIONS(test_nn_cuda_basics)
gtest_discover_tests(test_nn_cuda_basics)

#add_executable(
#        test_nn_cuda_cutlass
#        cutlass_test.cu
#)
#target_link_libraries(
#        test_nn_cuda_cutlass
#        PRIVATE
#        CUTLASS
#)
