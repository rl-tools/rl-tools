add_executable(
        rl_environments_pendulum_sac_training_standalone
        training.cpp
)
target_link_libraries(
        rl_environments_pendulum_sac_training_standalone
        PRIVATE
        rl_tools
)
target_compile_definitions(rl_environments_pendulum_sac_training_standalone PRIVATE RL_TOOLS_BACKEND_DISABLE_BLAS)
target_compile_definitions(rl_environments_pendulum_sac_training_standalone PRIVATE RL_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_sac_training_standalone PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
RL_TOOLS_INSTALL(rl_environments_pendulum_sac_training_standalone)

# RL Algorithm sac test full training blas
if(RL_TOOLS_BACKEND_ENABLE_BLAS)
add_executable(
        rl_environments_pendulum_sac_training_blas
        training.cpp
)
target_link_libraries(
        rl_environments_pendulum_sac_training_blas
        PRIVATE
        rl_tools
)
target_compile_definitions(rl_environments_pendulum_sac_training_blas PRIVATE RL_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_sac_training_blas PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
RL_TOOLS_INSTALL(rl_environments_pendulum_sac_training_blas)
endif()
