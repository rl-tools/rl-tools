if(RL_TOOLS_ENABLE_HYPERLUX)
    add_executable(rl_environments_hyperdrone_tasks_moving_gate moving_gate.cpp)
    target_link_libraries(rl_environments_hyperdrone_tasks_moving_gate
        PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
    )
endif()
