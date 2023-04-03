project(layer_in_c_rl_algorithms_td3_arm)
set(CMAKE_CXX_STANDARD 17)
add_executable(
        test_rl_algorithms_td3_arm_test
        test.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_arm_test
        PRIVATE
        layer_in_c
)

