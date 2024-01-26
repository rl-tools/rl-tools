add_executable(
        rl_environments_pendulum_ppo_training
        training.cpp
)
target_link_libraries(
        rl_environments_pendulum_ppo_training
        PRIVATE
        RLtools::RLtools
)
target_compile_definitions(rl_environments_pendulum_ppo_training PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
RL_TOOLS_INSTALL(rl_environments_pendulum_ppo_training)
