
function(RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT TARGET)
    if(RL_TOOLS_ENABLE_CLI11)
        target_link_libraries(${TARGET} PRIVATE CLI11::CLI11)
    endif()
endfunction()

# SAC Pendulum-v1
add_executable(rl_zoo_pendulum_v1_sac zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_pendulum_v1_sac PRIVATE RLtools::RLtools)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_pendulum_v1_sac)
target_compile_definitions(rl_zoo_pendulum_v1_sac PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
target_compile_definitions(rl_zoo_pendulum_v1_sac PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
# SAC AcrobotSwingup-v0
add_executable(rl_zoo_acrobot_swingup_v0_sac zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_acrobot_swingup_v0_sac PRIVATE RLtools::RLtools)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_acrobot_swingup_v0_sac)
target_compile_definitions(rl_zoo_acrobot_swingup_v0_sac PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
target_compile_definitions(rl_zoo_acrobot_swingup_v0_sac PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_ACROBOT_SWINGUP_V0)
# SAC Learning to Fly
add_executable(rl_zoo_l2f_sac zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_l2f_sac PRIVATE RLtools::RLtools)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_l2f_sac)
target_compile_definitions(rl_zoo_l2f_sac PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
target_compile_definitions(rl_zoo_l2f_sac PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)
# TD3 Pendulum-v1
add_executable(rl_zoo_pendulum_v1_td3 zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_pendulum_v1_td3 PRIVATE RLtools::RLtools)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_pendulum_v1_td3)
target_compile_definitions(rl_zoo_pendulum_v1_td3 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
target_compile_definitions(rl_zoo_pendulum_v1_td3 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
# TD3 Learning to Fly
add_executable(rl_zoo_l2f_td3 zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_l2f_td3 PRIVATE RLtools::RLtools)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_l2f_td3)
target_compile_definitions(rl_zoo_l2f_td3 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
target_compile_definitions(rl_zoo_l2f_td3 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)

if(RL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO)
    # TD3 Ant-v4
    add_executable(rl_zoo_ant_v4_td3 zoo_cli.cpp zoo.cpp)
    target_link_libraries(rl_zoo_ant_v4_td3 PRIVATE RLtools::RLtools)
    RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_ant_v4_td3)
    target_compile_definitions(rl_zoo_ant_v4_td3 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
    target_compile_definitions(rl_zoo_ant_v4_td3 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4)
endif()

# PPO Pendulum-v1
add_executable(rl_zoo_pendulum_v1_ppo zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_pendulum_v1_ppo PRIVATE RLtools::RLtools)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_pendulum_v1_ppo)
target_compile_definitions(rl_zoo_pendulum_v1_ppo PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
target_compile_definitions(rl_zoo_pendulum_v1_ppo PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)

## PPO Bottleneck-v0
add_executable(rl_zoo_bottleneck_v0_ppo zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_bottleneck_v0_ppo PRIVATE RLtools::RLtools)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_bottleneck_v0_ppo)
target_compile_definitions(rl_zoo_bottleneck_v0_ppo PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
target_compile_definitions(rl_zoo_bottleneck_v0_ppo PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_BOTTLENECK_V0)
target_compile_options(rl_zoo_bottleneck_v0_ppo PRIVATE -Ofast -march=native)

if(RL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO)
    # PPO Ant-v4
    add_executable(rl_zoo_ant_v4_ppo zoo_cli.cpp zoo.cpp)
    target_link_libraries(rl_zoo_ant_v4_ppo PRIVATE RLtools::RLtools)
    RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_ant_v4_ppo)
    target_compile_definitions(rl_zoo_ant_v4_ppo PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
    target_compile_definitions(rl_zoo_ant_v4_ppo PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4)
endif()
