add_executable(
        rl_environments_mujoco_ant_training_ppo_standalone
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_training_ppo_standalone
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
        mujoco
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_ppo_standalone)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_training_ppo_standalone)

add_executable(
        rl_environments_mujoco_ant_training_ppo_benchmark
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_training_ppo_benchmark
        PRIVATE
        backprop_tools
        mujoco
)
target_compile_definitions(
        rl_environments_mujoco_ant_training_ppo_benchmark
        PRIVATE
        -DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_BENCHMARK
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_ppo_benchmark)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_benchmark PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_training_ppo_benchmark)


