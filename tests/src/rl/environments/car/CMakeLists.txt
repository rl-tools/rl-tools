add_executable(rl_environments_car_comparison car_comparison.cpp)
target_link_libraries(rl_environments_car_comparison
        backprop_tools
        backprop_tools_tests
)