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
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_pendulum_td3_training_standalone)
target_compile_definitions(rl_environments_pendulum_td3_training_standalone PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_td3_training_standalone)

# RL Algorithm td3 test full training tensorboard
if(BACKPROP_TOOLS_ENABLE_TENSORBOARD)
add_executable(
        rl_environments_pendulum_td3_training_tensorboard
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_tensorboard
        PRIVATE
        backprop_tools
        tensorboard_logger
)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_pendulum_td3_training_tensorboard)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_td3_training_tensorboard)
endif()
