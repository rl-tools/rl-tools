project(test_nn_layers_dense_cuda CUDA)

if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_BACKEND_ENABLE_CUDNN AND RL_TOOLS_ENABLE_HDF5 AND RL_TOOLS_TEST_DATA_PATH)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CXX_STANDARD  17)

    add_executable(test_nn_layers_dense_cuda dense_cuda.cu)
    target_link_libraries(test_nn_layers_dense_cuda PRIVATE rl_tools_full RLtools::Test GTest::gtest_main)
    gtest_discover_tests(test_nn_layers_dense_cuda)
    target_compile_options(test_nn_layers_dense_cuda PRIVATE --diag-suppress=186)
endif()
