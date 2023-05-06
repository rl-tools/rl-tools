# RL Algorithm td3 test full training standalone
add_executable(
        rl_environments_pendulum_td3_training_standalone
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_standalone
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_td3_training_standalone)
target_compile_definitions(rl_environments_pendulum_td3_training_standalone PRIVATE BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
target_compile_definitions(rl_environments_pendulum_td3_training_standalone PRIVATE BACKPROP_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_td3_training_standalone PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)

# RL Algorithm td3 test full training blas
if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS)
add_executable(
        rl_environments_pendulum_td3_training_blas
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_blas
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_td3_training_blas)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_pendulum_td3_training_blas)
target_compile_definitions(rl_environments_pendulum_td3_training_blas PRIVATE BACKPROP_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_td3_training_blas PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
endif()

# RL Algorithm td3 test full training benchmark (disabling evaluation)
if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS)
add_executable(
        rl_environments_pendulum_td3_training_blas_benchmark
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_blas_benchmark
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_td3_training_blas_benchmark)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_pendulum_td3_training_blas_benchmark)
target_compile_definitions(rl_environments_pendulum_td3_training_blas_benchmark PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
target_compile_definitions(rl_environments_pendulum_td3_training_blas_benchmark PRIVATE BACKPROP_TOOLS_RL_ENVIRONMENTS_PENDULUM_DISABLE_EVALUATION)
endif()

# RL Algorithm td3 test full training tensorboard
if(BACKPROP_TOOLS_ENABLE_TENSORBOARD AND BACKPROP_TOOLS_BACKEND_ENABLE_BLAS)
add_executable(
        rl_environments_pendulum_td3_training_blas_tensorboard
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_blas_tensorboard
        PRIVATE
        backprop_tools
        tensorboard_logger
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_td3_training_blas_tensorboard)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_pendulum_td3_training_blas_tensorboard)
endif()
