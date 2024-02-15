
if(RL_TOOLS_BACKEND_ENABLE_MKL)
add_executable(
        test_nn_layers_dense_mkl
        mkl.cpp
)
target_link_libraries(
        test_nn_layers_dense_mkl
        PRIVATE
        RLtools::Backend
        GTest::gtest_main
)
gtest_discover_tests(test_nn_layers_dense_mkl)
endif()