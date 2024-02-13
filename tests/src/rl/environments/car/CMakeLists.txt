add_executable(rl_environments_car_comparison car_comparison.cpp)
target_link_libraries(rl_environments_car_comparison
        rl_tools
        rl_tools_test
)