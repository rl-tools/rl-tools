if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    add_executable(rl_environments_hyperdrone_tasks_target_frame target_frame.cpp)
    target_link_libraries(rl_environments_hyperdrone_tasks_target_frame
        PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
    )
    add_subdirectory(imitation)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "OPTIX")
        add_subdirectory(on_policy)
        add_subdirectory(mixed)
        add_subdirectory(yaw)
    endif()
endif()
