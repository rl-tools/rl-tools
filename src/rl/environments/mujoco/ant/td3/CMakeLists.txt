find_package(Threads REQUIRED)

add_executable(
        rl_environments_mujoco_ant_td3_standalone
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_td3_standalone
        PRIVATE
        RLtools::Core
        mujoco::mujoco
        Threads::Threads
)
target_compile_definitions(rl_environments_mujoco_ant_td3_standalone PRIVATE RL_TOOLS_BACKEND_DISABLE_BLAS)
target_compile_definitions(rl_environments_mujoco_ant_td3_standalone PRIVATE RL_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_mujoco_ant_td3_standalone PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
RL_TOOLS_INSTALL(rl_environments_mujoco_ant_td3_standalone)

if(RL_TOOLS_BACKEND_ENABLE_BLAS AND RL_TOOLS_ENABLE_HDF5 AND RL_TOOLS_ENABLE_TENSORBOARD)
add_executable(
        rl_environments_mujoco_ant_td3_blas
        standalone.cpp
)
target_link_libraries(
        rl_environments_mujoco_ant_td3_blas
        PRIVATE
        RLtools::RLtools
        mujoco::mujoco
)
RL_TOOLS_INSTALL(rl_environments_mujoco_ant_td3_blas)
endif()