
if(RL_TOOLS_ENABLE_HDF5)
add_executable(test_nn_models_sequential_persist persist.cpp)
target_link_libraries(test_nn_models_sequential_persist
    PRIVATE
    rl_tools
    rl_tools_test
)
gtest_discover_tests(test_nn_models_sequential_persist)

endif()

add_executable(test_nn_models_sequential_persist_code persist_code.cpp)
target_link_libraries(test_nn_models_sequential_persist_code
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_nn_models_sequential_persist_code)

add_executable(test_nn_models_sequential_persist_code_compile persist_code_compile.cpp)
target_link_libraries(test_nn_models_sequential_persist_code_compile
        PRIVATE
        rl_tools
        rl_tools_test
        )
gtest_discover_tests(test_nn_models_sequential_persist_code_compile)

add_executable(
        test_nn_models_sequential_d_input_benchmark
        d_input_benchmark.cpp
)
target_link_libraries(
        test_nn_models_sequential_d_input_benchmark
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_nn_models_sequential_d_input_benchmark)
