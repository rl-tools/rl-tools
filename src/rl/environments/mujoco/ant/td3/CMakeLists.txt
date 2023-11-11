find_package(Threads REQUIRED)

add_executable(
        rl_environments_mujoco_ant_training_td3_standalone
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_training_td3_standalone
        PRIVATE
        rl_tools
        Threads::Threads
)
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
        rl_tools
)
BACKPROP_TOOLS_INSTALL(rl_environments_mujoco_ant_training_td3_blas)
endif()