if(RL_TOOLS_ENABLE_JSON)
    add_executable(
            test_rl_environments_l2f_validation
            validation.cpp
    )
    target_link_libraries(
            test_rl_environments_l2f_validation
            rl_tools
            rl_tools_test
    )
    gtest_discover_tests(test_rl_environments_l2f_validation)
endif()

