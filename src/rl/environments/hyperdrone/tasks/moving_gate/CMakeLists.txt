if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    add_executable(rl_environments_hyperdrone_tasks_moving_gate moving_gate.cpp)
    target_link_libraries(rl_environments_hyperdrone_tasks_moving_gate
        PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
    )
endif()
