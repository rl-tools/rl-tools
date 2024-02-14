add_subdirectory(nn)
add_subdirectory(rl)
add_subdirectory(container)
add_subdirectory(random)

add_subdirectory(nn/layers)


add_subdirectory(nn_models/sequential)
add_subdirectory(nn_models/mlp)
add_subdirectory(rl/environments/car)
add_subdirectory(rl/environments/acrobot)
if(RL_TOOLS_BACKEND_ENABLE_CUDA)
    add_subdirectory(rl/cuda)
endif()
add_subdirectory(logging)
add_subdirectory(math)
add_subdirectory(ui_server)
