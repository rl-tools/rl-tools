
add_executable(
    test_rl_environments_mujoco_ant
    ant.cpp
)
target_link_libraries(
        test_rl_environments_mujoco_ant
        PRIVATE
        backprop_tools
        GTest::gtest_main
        HighFive
        tensorboard_logger
        mujoco
)
gtest_discover_tests(test_rl_environments_mujoco_ant)
BACKPROP_TOOLS_ADD_DEFINITIONS(test_rl_environments_mujoco_ant)
BACKPROP_TOOLS_TESTS_ADD_DEFINITIONS(test_rl_environments_mujoco_ant)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_rl_environments_mujoco_ant)

add_executable(
        test_rl_environments_mujoco_ant_training_td3_test
        training_td3_test.cpp
)
target_link_libraries(
        test_rl_environments_mujoco_ant_training_td3_test
        PRIVATE
        backprop_tools
        GTest::gtest_main
        HighFive
        tensorboard_logger
        mujoco
)
gtest_discover_tests(test_rl_environments_mujoco_ant_training_td3_test)
target_compile_definitions(
        test_rl_environments_mujoco_ant_training_td3_test
        PRIVATE
        -DBACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_TRAINING_TD3_TEST
)
BACKPROP_TOOLS_ADD_DEFINITIONS(test_rl_environments_mujoco_ant_training_td3_test)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_rl_environments_mujoco_ant_training_td3_test)


if(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI)
add_executable(
        test_rl_environments_mujoco_ant_ui
        ui.cpp
)
target_link_libraries(
        test_rl_environments_mujoco_ant_ui
        PRIVATE
        backprop_tools
        GTest::gtest_main
        HighFive
        tensorboard_logger
        mujoco
        glfw
)
gtest_discover_tests(test_rl_environments_mujoco_ant_ui)
BACKPROP_TOOLS_ADD_DEFINITIONS(test_rl_environments_mujoco_ant_ui)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_rl_environments_mujoco_ant_ui)
endif()


add_executable(
        test_rl_environments_mujoco_ant_training_ppo_test
        training_ppo_test.cpp
)
target_link_libraries(
        test_rl_environments_mujoco_ant_training_ppo_test
        PRIVATE
        backprop_tools
        GTest::gtest_main
        HighFive
        tensorboard_logger
        mujoco
)
gtest_discover_tests(test_rl_environments_mujoco_ant_training_ppo_test)
BACKPROP_TOOLS_ADD_DEFINITIONS(test_rl_environments_mujoco_ant_training_ppo_test)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_rl_environments_mujoco_ant_training_ppo_test)


add_executable(
        test_rl_environments_mujoco_ant_throughput
        throughput.cpp
)
target_link_libraries(
        test_rl_environments_mujoco_ant_throughput
        PRIVATE
        backprop_tools
        GTest::gtest_main
        HighFive
        tensorboard_logger
        mujoco
)
gtest_discover_tests(test_rl_environments_mujoco_ant_throughput)
BACKPROP_TOOLS_ADD_DEFINITIONS(test_rl_environments_mujoco_ant_throughput)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_rl_environments_mujoco_ant_throughput)

