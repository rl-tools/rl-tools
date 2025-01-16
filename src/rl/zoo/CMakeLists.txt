function(RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT TARGET)
    if(RL_TOOLS_ENABLE_CLI11)
        target_link_libraries(${TARGET} PRIVATE CLI11::CLI11)
    endif()
endfunction()


if(UNIX)
    file(GLOB_RECURSE GIT_DIFF_PROJECT_SOURCES
            CONFIGURE_DEPENDS
            "${CMAKE_PROJECT_SOURCE_DIR}/src/*"
            "${CMAKE_PROJECT_SOURCE_DIR}/include/*"
    )
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            DEPENDS ${GIT_DIFF_PROJECT_SOURCES}
            COMMAND ${CMAKE_COMMAND} -E echo "namespace rl_tools::extrack::git{ extern const char* const diff = R\"git_diff(" > ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND git diff >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo ")git_diff\"; }" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating git diff source"
            VERBATIM
    )

    add_library(git_diff STATIC ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp)
    target_compile_definitions(git_diff PUBLIC RL_TOOLS_EXTRACK_GIT_DIFF)
endif()

# Pendulum-v1
# PPO
add_executable(rl_zoo_pendulum_v1_ppo zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_pendulum_v1_ppo PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_pendulum_v1_ppo)
target_compile_definitions(rl_zoo_pendulum_v1_ppo PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
target_compile_definitions(rl_zoo_pendulum_v1_ppo PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
# SAC
add_executable(rl_zoo_pendulum_v1_sac zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_pendulum_v1_sac PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_pendulum_v1_sac)
target_compile_definitions(rl_zoo_pendulum_v1_sac PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
target_compile_definitions(rl_zoo_pendulum_v1_sac PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
# TD3
add_executable(rl_zoo_pendulum_v1_td3 zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_pendulum_v1_td3 PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_pendulum_v1_td3)
target_compile_definitions(rl_zoo_pendulum_v1_td3 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
target_compile_definitions(rl_zoo_pendulum_v1_td3 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)

# Flag
# SAC
add_executable(rl_zoo_flag_sac zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_flag_sac PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_flag_sac)
target_compile_definitions(rl_zoo_flag_sac PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
target_compile_definitions(rl_zoo_flag_sac PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_FLAG)
# TD3
add_executable(rl_zoo_flag_td3 zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_flag_td3 PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_flag_td3)
target_compile_definitions(rl_zoo_flag_td3 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
target_compile_definitions(rl_zoo_flag_td3 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_FLAG)
# PPO
add_executable(rl_zoo_flag_ppo zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_flag_ppo PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_flag_ppo)
target_compile_definitions(rl_zoo_flag_ppo PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
target_compile_definitions(rl_zoo_flag_ppo PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_FLAG)

# AcrobotSwingup-v0
# SAC
add_executable(rl_zoo_acrobot_swingup_v0_sac zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_acrobot_swingup_v0_sac PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_acrobot_swingup_v0_sac)
target_compile_definitions(rl_zoo_acrobot_swingup_v0_sac PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
target_compile_definitions(rl_zoo_acrobot_swingup_v0_sac PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_ACROBOT_SWINGUP_V0)

# Learning to Fly
# SAC
add_executable(rl_zoo_l2f_sac zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_l2f_sac PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_l2f_sac)
target_compile_definitions(rl_zoo_l2f_sac PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
target_compile_definitions(rl_zoo_l2f_sac PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)
# SAC BENCHMARK
add_executable(rl_zoo_l2f_sac_benchmark zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_l2f_sac_benchmark PRIVATE RLtools::Optimizations)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_l2f_sac_benchmark)
target_compile_definitions(rl_zoo_l2f_sac_benchmark PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
target_compile_definitions(rl_zoo_l2f_sac_benchmark PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)
target_compile_definitions(rl_zoo_l2f_sac_benchmark PRIVATE RL_TOOLS_RL_ZOO_BENCHMARK)
# TD3
add_executable(rl_zoo_l2f_td3 zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_l2f_td3 PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_l2f_td3)
target_compile_definitions(rl_zoo_l2f_td3 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
target_compile_definitions(rl_zoo_l2f_td3 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)
# PPO
add_executable(rl_zoo_l2f_ppo zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_l2f_ppo PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_l2f_ppo)
target_compile_definitions(rl_zoo_l2f_ppo PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
target_compile_definitions(rl_zoo_l2f_ppo PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)

# Bottleneck-v0
# PPO
add_executable(rl_zoo_bottleneck_v0_ppo zoo_cli.cpp zoo.cpp)
target_link_libraries(rl_zoo_bottleneck_v0_ppo PRIVATE RLtools::RLtools git_diff)
RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_bottleneck_v0_ppo)
target_compile_definitions(rl_zoo_bottleneck_v0_ppo PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
target_compile_definitions(rl_zoo_bottleneck_v0_ppo PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_BOTTLENECK_V0)
target_compile_options(rl_zoo_bottleneck_v0_ppo PRIVATE -Ofast -march=native)

if(RL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO)
    # Ant-v4
    # PPO
    add_executable(rl_zoo_ant_v4_ppo zoo_cli.cpp zoo.cpp)
    target_link_libraries(rl_zoo_ant_v4_ppo PRIVATE RLtools::RLtools git_diff)
    RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_ant_v4_ppo)
    target_compile_definitions(rl_zoo_ant_v4_ppo PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
    target_compile_definitions(rl_zoo_ant_v4_ppo PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4)
    # TD3
    add_executable(rl_zoo_ant_v4_td3 zoo_cli.cpp zoo.cpp)
    target_link_libraries(rl_zoo_ant_v4_td3 PRIVATE RLtools::RLtools git_diff)
    RL_TOOLS_RL_ZOO_LINK_CLI11_OR_NOT(rl_zoo_ant_v4_td3)
    target_compile_definitions(rl_zoo_ant_v4_td3 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
    target_compile_definitions(rl_zoo_ant_v4_td3 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4)
endif()
