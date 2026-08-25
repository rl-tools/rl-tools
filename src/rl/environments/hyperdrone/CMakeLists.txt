if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    add_executable(rl_environments_hyperdrone_moving_gate_demo tasks/moving_gate.cpp)
    target_link_libraries(rl_environments_hyperdrone_moving_gate_demo
        PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
    )
    add_executable(rl_environments_hyperdrone_target_frame_demo tasks/target_frame.cpp)
    target_link_libraries(rl_environments_hyperdrone_target_frame_demo
        PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
    )
endif()
