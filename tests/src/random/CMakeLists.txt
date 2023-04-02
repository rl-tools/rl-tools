project(layer_in_c_random_arm)

add_executable(
        test_random_arm
        arm.cpp
)
target_link_libraries(
        test_random_arm
        PRIVATE
        layer_in_c
        GTest::gtest_main
)
gtest_discover_tests(test_random_arm)
LAYER_IN_C_TESTS_ADD_DEBUG_DEFINITIONS(test_random_arm)
