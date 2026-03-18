project(test_nn_layers_dynamic_conv2d_cuda CUDA)

if(RL_TOOLS_BACKEND_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CXX_STANDARD  17)

    add_executable(test_nn_layers_dynamic_conv2d_cuda dynamic_conv2d_cuda.cu)
    target_link_libraries(test_nn_layers_dynamic_conv2d_cuda PRIVATE rl_tools_full RLtools::Test GTest::gtest_main)
    gtest_discover_tests(test_nn_layers_dynamic_conv2d_cuda)
    target_compile_options(test_nn_layers_dynamic_conv2d_cuda PRIVATE --diag-suppress=186)

    add_executable(test_nn_layers_dynamic_conv2d_cuda_bf16 dynamic_conv2d_cuda_bf16.cu)
    target_link_libraries(test_nn_layers_dynamic_conv2d_cuda_bf16 PRIVATE rl_tools_full RLtools::Test GTest::gtest_main)
    gtest_discover_tests(test_nn_layers_dynamic_conv2d_cuda_bf16)
    target_compile_options(test_nn_layers_dynamic_conv2d_cuda_bf16 PRIVATE --diag-suppress=186)
endif()
