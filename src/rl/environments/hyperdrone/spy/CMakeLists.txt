if(TARGET RLtools::RaytracingSpy AND NOT RL_TOOLS_DISABLE_TARGETS)
    add_executable(example_hyperdrone_spy example.cpp)
    target_link_libraries(example_hyperdrone_spy PRIVATE RLtools::RLtools RLtools::RaytracingSpy rendering_raytracing_backend rendering_raytracing_stb)
endif()
