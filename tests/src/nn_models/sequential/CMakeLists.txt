
if(BACKPROP_TOOLS_ENABLE_HDF5)
add_executable(test_nn_models_sequential_persist persist.cpp)
target_link_libraries(test_nn_models_sequential_persist
    PRIVATE
    backprop_tools
    GTest::gtest_main
    HighFive
)
gtest_discover_tests(test_nn_models_sequential_persist)

endif()

add_executable(test_nn_models_sequential_persist_code persist_code.cpp)
target_link_libraries(test_nn_models_sequential_persist_code
        PRIVATE
        backprop_tools
        GTest::gtest_main
        )
gtest_discover_tests(test_nn_models_sequential_persist_code)

add_executable(test_nn_models_sequential_persist_code_compile persist_code_compile.cpp)
target_link_libraries(test_nn_models_sequential_persist_code_compile
        PRIVATE
        backprop_tools
        GTest::gtest_main
        )
gtest_discover_tests(test_nn_models_sequential_persist_code_compile)

add_executable(
        test_nn_models_sequential_d_input_benchmark
        d_input_benchmark.cpp
)
target_link_libraries(
        test_nn_models_sequential_d_input_benchmark
        PRIVATE
        backprop_tools
        GTest::gtest_main
)
gtest_discover_tests(test_nn_models_sequential_d_input_benchmark)
BACKPROP_TOOLS_TESTS_ADD_DEFINITIONS(test_nn_models_sequential_d_input_benchmark)
BACKPROP_TOOLS_ADD_DEFINITIONS(test_nn_models_sequential_d_input_benchmark)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_nn_models_sequential_d_input_benchmark)
