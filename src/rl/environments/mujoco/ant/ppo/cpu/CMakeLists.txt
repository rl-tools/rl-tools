add_executable(
        rl_environments_mujoco_ant_training_ppo_mkl
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_training_ppo_mkl
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
)
# todo: link mujoco like this everywhere
BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_training_ppo_mkl)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_ppo_mkl)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_training_ppo_mkl)

add_executable(
        rl_environments_mujoco_ant_training_ppo_mkl_benchmark
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_training_ppo_mkl_benchmark
        PRIVATE
        backprop_tools
        mujoco
)
target_compile_definitions(
        rl_environments_mujoco_ant_training_ppo_mkl_benchmark
        PRIVATE
        -DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_BENCHMARK
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_ppo_mkl_benchmark)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_mkl_benchmark PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_mkl_benchmark PRIVATE BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_BENCHMARK)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_training_ppo_mkl_benchmark)


