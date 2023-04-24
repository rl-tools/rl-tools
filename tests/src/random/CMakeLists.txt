project(backprop_tools_random_arm)

add_executable(
        test_random_arm
        arm.cpp
)
target_link_libraries(
        test_random_arm
        PRIVATE
        backprop_tools
        GTest::gtest_main
)
gtest_discover_tests(test_random_arm)
BACKPROP_TOOLS_ADD_DEFINITIONS(test_random_arm)
