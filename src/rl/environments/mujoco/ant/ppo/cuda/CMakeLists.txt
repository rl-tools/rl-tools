
project(rl_environments_mujoco_ant_cuda LANGUAGES CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
message("CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message("CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")

add_executable(
        rl_environments_mujoco_ant_throughput_cuda
        throughput.cu
)
target_link_libraries(
        rl_environments_mujoco_ant_throughput_cuda
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
        CUDA::cublas
)
set_target_properties(rl_environments_mujoco_ant_throughput_cuda PROPERTIES CUDA_ARCHITECTURES ${BACKPROP_TOOLS_BACKEND_CUDA_ARCHITECTURES})
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_throughput_cuda)
BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_throughput_cuda)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_throughput_cuda)


add_executable(
        rl_environments_mujoco_ant_training_ppo_cuda
        training_ppo.cu
)
target_link_libraries(
        rl_environments_mujoco_ant_training_ppo_cuda
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
        CUDA::cublas
        CLI11::CLI11
)
set_target_properties(rl_environments_mujoco_ant_training_ppo_cuda PROPERTIES CUDA_ARCHITECTURES ${BACKPROP_TOOLS_BACKEND_CUDA_ARCHITECTURES})
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_ppo_cuda)
BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_training_ppo_cuda)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_training_ppo_cuda)

add_executable(
        rl_environments_mujoco_ant_collection_cuda
        collection.cu
)
target_link_libraries(
        rl_environments_mujoco_ant_collection_cuda
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
        mujoco
        CUDA::cublas
)
set_target_properties(rl_environments_mujoco_ant_collection_cuda PROPERTIES CUDA_ARCHITECTURES ${BACKPROP_TOOLS_BACKEND_CUDA_ARCHITECTURES})
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_collection_cuda)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_collection_cuda)
