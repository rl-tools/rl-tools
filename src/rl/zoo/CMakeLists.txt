if(RL_TOOLS_ENABLE_ZLIB)
    # SAC Pendulum-v1
    add_executable(rl_zoo_sac_pendulum_v1 zoo.cpp)
    target_link_libraries(rl_zoo_sac_pendulum_v1 PRIVATE RLtools::RLtools ZLIB::ZLIB)
    target_compile_definitions(rl_zoo_sac_pendulum_v1 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
    target_compile_definitions(rl_zoo_sac_pendulum_v1 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
    # SAC AcrobotSwingup-v0
    add_executable(rl_zoo_sac_acrobot_swingup_v0 zoo.cpp)
    target_link_libraries(rl_zoo_sac_acrobot_swingup_v0 PRIVATE RLtools::RLtools ZLIB::ZLIB)
    target_compile_definitions(rl_zoo_sac_acrobot_swingup_v0 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
    target_compile_definitions(rl_zoo_sac_acrobot_swingup_v0 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_ACROBOT_SWINGUP_V0)
    # TD3 Pendulum-v1
    add_executable(rl_zoo_td3_pendulum_v1 zoo.cpp)
    target_link_libraries(rl_zoo_td3_pendulum_v1 PRIVATE RLtools::RLtools ZLIB::ZLIB)
    target_compile_definitions(rl_zoo_td3_pendulum_v1 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
    target_compile_definitions(rl_zoo_td3_pendulum_v1 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
    # PPO Pendulum-v1
    add_executable(rl_zoo_ppo_pendulum_v1 zoo.cpp)
    target_link_libraries(rl_zoo_ppo_pendulum_v1 PRIVATE RLtools::RLtools ZLIB::ZLIB)
    target_compile_definitions(rl_zoo_ppo_pendulum_v1 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
    target_compile_definitions(rl_zoo_ppo_pendulum_v1 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)

    if(RL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO)
        # PPO Ant-v4
        add_executable(rl_zoo_ppo_ant_v4 zoo.cpp)
        target_link_libraries(rl_zoo_ppo_ant_v4 PRIVATE RLtools::RLtools ZLIB::ZLIB)
        target_compile_definitions(rl_zoo_ppo_ant_v4 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
        target_compile_definitions(rl_zoo_ppo_ant_v4 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4)
    endif()
endif()




if(RL_TOOLS_ENABLE_CLI11)
add_executable(
        rl_zoo_enjoy
        enjoy.cpp
)
target_link_libraries(
        rl_zoo_enjoy
        PRIVATE
        RLtools::RLtools
        CLI11::CLI11
)
target_compile_definitions(rl_zoo_enjoy PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
target_compile_definitions(rl_zoo_enjoy PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
endif()