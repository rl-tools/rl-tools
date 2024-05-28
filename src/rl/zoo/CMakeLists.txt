if(RL_TOOLS_ENABLE_ZLIB)
    # SAC Pendulum-v1
    add_executable(rl_zoo_sac_pendulum_v1 zoo.cpp)
    target_link_libraries(rl_zoo_sac_pendulum_v1 PRIVATE RLtools::RLtools ZLIB::ZLIB)
    target_compile_definitions(rl_zoo_sac_pendulum_v1 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
    target_compile_definitions(rl_zoo_sac_pendulum_v1 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
    # TD3 Pendulum-v1
    add_executable(rl_zoo_td3_pendulum_v1 zoo.cpp)
    target_link_libraries(rl_zoo_td3_pendulum_v1 PRIVATE RLtools::RLtools ZLIB::ZLIB)
    target_compile_definitions(rl_zoo_td3_pendulum_v1 PRIVATE RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
    target_compile_definitions(rl_zoo_td3_pendulum_v1 PRIVATE RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
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
endif()