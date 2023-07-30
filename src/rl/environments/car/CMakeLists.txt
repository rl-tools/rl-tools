add_executable(rl_environments_car_training car.cpp)
target_link_libraries(rl_environments_car_training
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_car_training)