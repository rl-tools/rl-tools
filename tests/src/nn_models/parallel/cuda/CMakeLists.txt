project(test_rl_tools_nn_models_parallel_cuda CUDA)

if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_BACKEND_ENABLE_MKL)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CXX_STANDARD  17)

    add_executable(test_nn_models_parallel_cuda parallel_cuda.cu)
    target_link_libraries(test_nn_models_parallel_cuda PRIVATE rl_tools_full GTest::gtest_main)
    gtest_discover_tests(test_nn_models_parallel_cuda)
    target_compile_options(test_nn_models_parallel_cuda PRIVATE --diag-suppress=186)
endif()
