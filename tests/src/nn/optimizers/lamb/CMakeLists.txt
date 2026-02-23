add_executable(test_nn_optimizers_lamb test_lamb.cpp)
target_link_libraries(test_nn_optimizers_lamb RLtools::Core GTest::gtest_main)
gtest_discover_tests(test_nn_optimizers_lamb)

if(RL_TOOLS_ENABLE_HDF5 AND RL_TOOLS_TEST_DATA_PATH)
    add_executable(test_nn_optimizers_lamb_hdf5 test_lamb_hdf5.cpp)
    target_link_libraries(test_nn_optimizers_lamb_hdf5 RLtools::RLtools GTest::gtest_main)
    gtest_discover_tests(test_nn_optimizers_lamb_hdf5)
endif()
