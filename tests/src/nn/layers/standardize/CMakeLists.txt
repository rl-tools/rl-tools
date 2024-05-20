add_executable(
        test_nn_layers_standardize
        standardize.cpp
)
target_link_libraries(
        test_nn_layers_standardize
        PRIVATE
        RLtools::RLtools
        GTest::gtest_main
)
gtest_discover_tests(test_nn_layers_standardize)

