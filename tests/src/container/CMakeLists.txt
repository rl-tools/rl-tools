# Container tests
add_executable(
        test_container
        container.cpp
)
target_link_libraries(
        test_container
        PRIVATE
        RLtools::Backend
        RLtools::Test
)
gtest_discover_tests(test_container)

add_executable(tensor_test
        tensor.cpp
)
target_link_libraries(
        tensor_test
        PRIVATE
        RLtools::Core
        GTest::gtest_main
)
gtest_discover_tests(tensor_test)

if(RL_TOOLS_BACKEND_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()

# Container tests
add_executable(
        test_container_persist_code_store
        persist_code_store.cpp
)
target_link_libraries(
        test_container_persist_code_store
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_container_persist_code_store)
