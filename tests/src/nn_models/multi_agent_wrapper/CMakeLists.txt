add_executable(test_nn_models_multi_agent_wrapper_persist_code multi_agent_wrapper_persist_code.cpp)
target_link_libraries(test_nn_models_multi_agent_wrapper_persist_code
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_nn_models_multi_agent_wrapper_persist_code)

if(EXISTS ${PROJECT_SOURCE_DIR}/tests/data/nn_models_multi_agent_wrapper_persist_code.h)
    add_executable(test_nn_models_multi_agent_wrapper_persist_code_compile multi_agent_wrapper_persist_code_compile.cpp)
    target_link_libraries(test_nn_models_multi_agent_wrapper_persist_code_compile
            PRIVATE
            rl_tools
            rl_tools_test
    )
    gtest_discover_tests(test_nn_models_multi_agent_wrapper_persist_code_compile)
else()
    message(WARNING "test_nn_models_multi_agent_wrapper_persist_code_compile is disabled because the file ${PROJECT_SOURCE_DIR}/tests/data/nn_models_multi_agent_wrapper_persist_code.h")
endif()
