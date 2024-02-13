project(rl_tools_random_arm)

add_executable(
        test_random_arm
        arm.cpp
)
target_link_libraries(
        test_random_arm
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_random_arm)
