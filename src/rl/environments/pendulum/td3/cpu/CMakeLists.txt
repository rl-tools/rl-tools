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
target_compile_definitions(rl_environments_pendulum_td3_training_standalone PRIVATE BACKPROP_TOOLS_BACKEND_DISABLE_MKL)
target_compile_definitions(rl_environments_pendulum_td3_training_standalone PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)

# RL Algorithm td3 test full training mkl
if(BACKPROP_TOOLS_BACKEND_ENABLE_MKL)
add_executable(
        rl_environments_pendulum_td3_training_mkl
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_mkl
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_td3_training_mkl)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_pendulum_td3_training_mkl)
target_compile_definitions(rl_environments_pendulum_td3_training_mkl PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
endif()

# RL Algorithm td3 test full training benchmark (disabling evaluation)
if(BACKPROP_TOOLS_BACKEND_ENABLE_MKL)
add_executable(
        rl_environments_pendulum_td3_training_mkl_benchmark
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_mkl_benchmark
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_td3_training_mkl_benchmark)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_pendulum_td3_training_mkl_benchmark)
target_compile_definitions(rl_environments_pendulum_td3_training_mkl_benchmark PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
target_compile_definitions(rl_environments_pendulum_td3_training_mkl_benchmark PRIVATE BACKPROP_TOOLS_DISABLE_EVALUATION)
endif()

# RL Algorithm td3 test full training tensorboard
if(BACKPROP_TOOLS_ENABLE_TENSORBOARD AND BACKPROP_TOOLS_BACKEND_ENABLE_MKL)
add_executable(
        rl_environments_pendulum_td3_training_mkl_tensorboard
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_mkl_tensorboard
        PRIVATE
        backprop_tools
        tensorboard_logger
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_td3_training_mkl_tensorboard)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_pendulum_td3_training_mkl_tensorboard)
endif()
