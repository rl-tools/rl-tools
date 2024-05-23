
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

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code.h)
add_executable(test_nn_models_sequential_persist_code_compile persist_code_compile.cpp)
target_link_libraries(test_nn_models_sequential_persist_code_compile
        PRIVATE
        rl_tools
        rl_tools_test
        )
gtest_discover_tests(test_nn_models_sequential_persist_code_compile)
else()
message(WARNING "test_nn_models_sequential_persist_code_compile is disabled because the file tests/data/nn_models_sequential_persist_code.h is missing")
endif()

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

add_executable(test_nn_models_sequential_compose compose.cpp)
target_link_libraries(test_nn_models_sequential_compose
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_nn_models_sequential_compose)
