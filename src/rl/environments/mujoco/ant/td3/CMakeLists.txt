add_executable(
        rl_environments_mujoco_ant_training_td3_standalone
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_training_td3_standalone
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_td3_standalone)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_training_td3_standalone)
BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_training_td3_standalone)
target_compile_definitions(rl_environments_mujoco_ant_training_td3_standalone PRIVATE BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
target_compile_definitions(rl_environments_mujoco_ant_training_td3_standalone PRIVATE BACKPROP_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_mujoco_ant_training_td3_standalone PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
BACKPROP_TOOLS_INSTALL(rl_environments_mujoco_ant_training_td3_standalone)

if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS AND BACKPROP_TOOLS_ENABLE_HDF5 AND BACKPROP_TOOLS_ENABLE_TENSORBOARD)
add_executable(
        rl_environments_mujoco_ant_training_td3_blas
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_training_td3_blas
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_td3_blas)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_training_td3_blas)
BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_training_td3_blas)
BACKPROP_TOOLS_INSTALL(rl_environments_mujoco_ant_training_td3_blas)
endif()