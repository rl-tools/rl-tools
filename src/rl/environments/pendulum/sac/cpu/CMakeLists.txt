add_executable(
        rl_environments_pendulum_sac_training_standalone
        training.cpp
)
target_link_libraries(
        rl_environments_pendulum_sac_training_standalone
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_sac_training_standalone)
target_compile_definitions(rl_environments_pendulum_sac_training_standalone PRIVATE BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
target_compile_definitions(rl_environments_pendulum_sac_training_standalone PRIVATE BACKPROP_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_sac_training_standalone PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
BACKPROP_TOOLS_INSTALL(rl_environments_pendulum_sac_training_standalone)
if(BACKPROP_TOOLS_ENABLE_HDF5)
    target_link_libraries(
            rl_environments_pendulum_sac_training_standalone
            PRIVATE
            HighFive
    )
    target_compile_definitions(rl_environments_pendulum_sac_training_standalone PRIVATE BACKPROP_TOOLS_ENABLE_HDF5)
endif()

# RL Algorithm sac test full training blas
if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS)
add_executable(
        rl_environments_pendulum_sac_training_blas
        training.cpp
)
target_link_libraries(
        rl_environments_pendulum_sac_training_blas
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_sac_training_blas)
BACKPROP_TOOLS_TARGET_LINK_DEPENDENCIES(rl_environments_pendulum_sac_training_blas)
target_compile_definitions(rl_environments_pendulum_sac_training_blas PRIVATE BACKPROP_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_sac_training_blas PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
BACKPROP_TOOLS_INSTALL(rl_environments_pendulum_sac_training_blas)
if(BACKPROP_TOOLS_ENABLE_HDF5)
    target_link_libraries(
            rl_environments_pendulum_sac_training_blas
            PRIVATE
            HighFive
    )
    target_compile_definitions(rl_environments_pendulum_sac_training_blas PRIVATE BACKPROP_TOOLS_ENABLE_HDF5)
endif()
endif()
