add_executable(test_nn_models_parallel_evaluate_step evaluate_step.cpp)
target_link_libraries(test_nn_models_parallel_evaluate_step PRIVATE RLtools::Core GTest::gtest_main)
gtest_discover_tests(test_nn_models_parallel_evaluate_step)

add_executable(test_nn_models_parallel_persist_code_store persist_code_store.cpp)
target_link_libraries(test_nn_models_parallel_persist_code_store PRIVATE RLtools::Core GTest::gtest_main)
gtest_discover_tests(test_nn_models_parallel_persist_code_store)

if(RL_TOOLS_TEST_DATA_PATH)
    add_executable(test_nn_models_parallel_persist_code_compile persist_code_compile.cpp)
    target_link_libraries(test_nn_models_parallel_persist_code_compile PRIVATE RLtools::Core GTest::gtest_main)
    gtest_discover_tests(test_nn_models_parallel_persist_code_compile)
endif()

if(RL_TOOLS_BACKEND_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
