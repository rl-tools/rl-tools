
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