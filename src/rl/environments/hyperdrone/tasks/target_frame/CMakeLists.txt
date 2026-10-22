if(RL_TOOLS_ENABLE_HYPERLUX)
    add_executable(rl_environments_hyperdrone_tasks_target_frame target_frame.cpp)
    target_link_libraries(rl_environments_hyperdrone_tasks_target_frame
        PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
    )
    add_subdirectory(imitation)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_HYPERLUX_BACKEND_RESOLVED STREQUAL "OPTIX")
        add_subdirectory(on_policy)
        add_subdirectory(mixed)
        add_subdirectory(yaw)
    endif()
endif()
