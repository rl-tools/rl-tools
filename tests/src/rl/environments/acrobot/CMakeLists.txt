add_executable(test_rl_environments_acrobot_comparison comparison.cpp)

target_link_libraries(test_rl_environments_acrobot_comparison
        PRIVATE
        backprop_tools
        GTest::gtest_main
        HighFive
)
BACKPROP_TOOLS_TESTS_ADD_DEFINITIONS(test_rl_environments_acrobot_comparison)
