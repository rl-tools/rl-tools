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
if(BACKPROP_TOOLS_ENABLE_TENSORBOARD)
target_link_libraries(
        rl_environments_pendulum_td3_training_standalone
        PRIVATE
        tensorboard_logger
)
endif()
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_pendulum_td3_training_standalone)
target_compile_definitions(rl_environments_pendulum_td3_training_standalone PRIVATE BACKPROP_TOOLS_RL_ENVIRONMENTS_PENDULUM_TD3_TRAINING_STANDALONE_DISABLE_TENSORBOARD)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_pendulum_td3_training_standalone)
