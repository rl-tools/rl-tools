add_executable(
        test_nn_layers_gru
        gru.cpp
)

target_link_libraries(
        test_nn_layers_gru
        PRIVATE
        RLtools::RLtools
        RLtools::Test
        GTest::gtest_main
        asan
)
gtest_discover_tests(test_nn_layers_gru)
RL_TOOLS_WARNINGS_ARE_ERRORS(test_nn_layers_gru)
target_compile_options(test_nn_layers_gru PRIVATE -fsanitize=address -fno-omit-frame-pointer)
target_compile_definitions(test_nn_layers_gru PRIVATE -DRL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS)