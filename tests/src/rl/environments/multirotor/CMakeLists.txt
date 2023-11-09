# Environment Multirotor test
add_executable(
        test_rl_environments_multirotor_multirotor
        multirotor.cpp
)
target_link_libraries(
        test_rl_environments_multirotor_multirotor
        backprop_tools
        backprop_tools_tests
)
gtest_discover_tests(test_rl_environments_multirotor_multirotor)



# Multirotor UI test
if(${BACKPROP_TOOLS_TESTS_RL_ENVIRONMENTS_MULTIROTOR_UI_ENABLE})
    add_executable(
            test_rl_environments_multirotor_ui_test
            src/rl/environments/multirotor/ui.cpp
    )
    target_link_libraries(
            test_rl_environments_multirotor_ui_test
            backprop_tools
            backprop_tools_tests
    )
    target_include_directories(
            test_rl_environments_multirotor_ui_test
            PRIVATE
            lib/nlohmann/include
    )
    BACKPROP_TOOLS_ADD_DEFINITIONS(test_rl_environments_multirotor_ui_test)
    gtest_discover_tests(test_rl_environments_multirotor_ui_test)
endif()


if(NOT APPLE)
#    add_subdirectory(regression_test_example_old)
endif()