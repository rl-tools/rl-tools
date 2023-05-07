add_executable(
        rl_environments_mujoco_ant_training_ppo_standalone
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_training_ppo_standalone
        PRIVATE
        backprop_tools
)
# todo: link mujoco like this everywhere
BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_training_ppo_standalone)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_ppo_standalone)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_standalone PRIVATE BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_standalone PRIVATE BACKPROP_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_standalone PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)

if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS AND BACKPROP_TOOLS_ENABLE_HDF5 AND BACKPROP_TOOLS_ENABLE_TENSORBOARD)
add_executable(
        rl_environments_mujoco_ant_training_ppo_blas
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_training_ppo_blas
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
)
# todo: link mujoco like this everywhere
BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_training_ppo_blas)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_ppo_blas)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_training_ppo_blas)
endif()

