add_executable(
        rl_environments_pendulum_ppo_training
        training.cpp
)
target_link_libraries(
        rl_environments_pendulum_ppo_training
        PRIVATE
        RLtools::RLtools
)
RL_TOOLS_INSTALL(rl_environments_pendulum_ppo_training)

add_executable(
        rl_environments_pendulum_ppo_benchmark
        training.cpp
)
target_link_libraries(
        rl_environments_pendulum_ppo_benchmark
        PRIVATE
        RLtools::RLtools
)
target_compile_definitions(rl_environments_pendulum_ppo_benchmark PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
target_compile_definitions(rl_environments_pendulum_ppo_benchmark PRIVATE BENCHMARK)
target_compile_options(rl_environments_pendulum_ppo_benchmark PRIVATE -Ofast -march=native)
RL_TOOLS_INSTALL(rl_environments_pendulum_ppo_benchmark)
