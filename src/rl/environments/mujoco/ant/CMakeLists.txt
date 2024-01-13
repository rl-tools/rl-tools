add_subdirectory(ppo)
add_subdirectory(td3)

if(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI AND RL_TOOLS_ENABLE_CLI11)
    add_executable(
            rl_environments_mujoco_ant_evaluation_td3
            evaluate_actor.cpp
    )
    target_link_libraries(
            rl_environments_mujoco_ant_evaluation_td3
            PRIVATE
            RLtools::RLtools
            CLI11::CLI11
    )
    RL_TOOLS_INSTALL(rl_environments_mujoco_ant_evaluation_td3)
endif()

if(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI AND RL_TOOLS_ENABLE_CLI11)
    add_executable(
            rl_environments_mujoco_ant_evaluation_ppo
            evaluate_actor.cpp
    )
    target_link_libraries(
            rl_environments_mujoco_ant_evaluation_ppo
            PRIVATE
            RLtools::RLtools
            CLI11::CLI11
    )
    target_compile_definitions(
            rl_environments_mujoco_ant_evaluation_ppo
            PRIVATE
            -DRL_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
    )
    RL_TOOLS_INSTALL(rl_environments_mujoco_ant_evaluation_ppo)
endif()
