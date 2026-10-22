if(TARGET RLtools::HyperLuxSpy AND NOT RL_TOOLS_DISABLE_TARGETS)
    add_executable(example_hyperdrone_spy example.cpp)
    target_link_libraries(example_hyperdrone_spy PRIVATE RLtools::RLtools RLtools::HyperLuxSpy hyperlux_backend hyperlux_stb)
endif()
