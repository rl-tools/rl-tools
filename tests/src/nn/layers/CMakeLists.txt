add_executable(
        test_nn_layers_dense
        dense.cpp
)
target_link_libraries(
        test_nn_layers_dense
        PRIVATE
        backprop_tools
        GTest::gtest_main
)
gtest_discover_tests(test_nn_layers_dense)
BACKPROP_TOOLS_TESTS_ADD_DEFINITIONS(test_nn_layers_dense)
BACKPROP_TOOLS_ADD_DEFINITIONS(test_nn_layers_dense)



add_subdirectory(concat_constant)