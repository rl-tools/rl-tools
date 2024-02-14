
add_executable(
    test_rl_environments_mujoco_ant
    ant.cpp
)
target_link_libraries(
        test_rl_environments_mujoco_ant
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_rl_environments_mujoco_ant)

add_executable(
        test_rl_environments_mujoco_ant_td3_test
        training_td3_test.cpp
)
target_link_libraries(
        test_rl_environments_mujoco_ant_td3_test
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_rl_environments_mujoco_ant_td3_test)
target_compile_definitions(
        test_rl_environments_mujoco_ant_td3_test
        PRIVATE
        -DRL_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_TRAINING_TD3_TEST
)


if(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI)
add_executable(
        test_rl_environments_mujoco_ant_ui
        ui.cpp
)
target_link_libraries(
        test_rl_environments_mujoco_ant_ui
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_rl_environments_mujoco_ant_ui)
endif()


add_executable(
        test_rl_environments_mujoco_ant_ppo_test
        training_ppo_test.cpp
)
target_link_libraries(
        test_rl_environments_mujoco_ant_ppo_test
        PRIVATE
        rl_tools
        GTest::gtest_main
        HighFive
        tensorboard_logger
        mujoco
)
gtest_discover_tests(test_rl_environments_mujoco_ant_ppo_test)


add_executable(
        test_rl_environments_mujoco_ant_throughput
        throughput.cpp
)
target_link_libraries(
        test_rl_environments_mujoco_ant_throughput
        PRIVATE
        rl_tools
        GTest::gtest_main
        HighFive
        tensorboard_logger
        mujoco
)
gtest_discover_tests(test_rl_environments_mujoco_ant_throughput)

