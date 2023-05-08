
project(rl_environments_mujoco_ant_cuda LANGUAGES CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
message("CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message("CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")

# add_executable(
#         rl_environments_mujoco_ant_throughput_cuda
#         throughput.cu
# )
# target_link_libraries(
#         rl_environments_mujoco_ant_throughput_cuda
#         PRIVATE
#         backprop_tools
#         HighFive
#         tensorboard_logger
#         CUDA::cublas
# )
# set_target_properties(rl_environments_mujoco_ant_throughput_cuda PROPERTIES CUDA_ARCHITECTURES ${BACKPROP_TOOLS_BACKEND_CUDA_ARCHITECTURES})
# BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_throughput_cuda)
# BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_throughput_cuda)
# BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_throughput_cuda)


add_executable(
        rl_environments_mujoco_ant_training_ppo_cuda_standalone
        training_ppo.cu
)
target_link_libraries(
        rl_environments_mujoco_ant_training_ppo_cuda_standalone
        PRIVATE
        backprop_tools
        CUDA::cublas
)
set_target_properties(rl_environments_mujoco_ant_training_ppo_cuda_standalone PROPERTIES CUDA_ARCHITECTURES ${BACKPROP_TOOLS_BACKEND_CUDA_ARCHITECTURES})
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_ppo_cuda_standalone)
BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_training_ppo_cuda_standalone)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_cuda_standalone PRIVATE BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_cuda_standalone PRIVATE BACKPROP_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_cuda_standalone PRIVATE BACKPROP_TOOLS_DISABLE_CLI11)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_cuda_standalone PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
BACKPROP_TOOLS_INSTALL(rl_environments_mujoco_ant_training_ppo_cuda_standalone)

if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS AND BACKPROP_TOOLS_ENABLE_CLI11)
add_executable(
        rl_environments_mujoco_ant_training_ppo_cuda_benchmark
        training_ppo.cu
)
target_link_libraries(
        rl_environments_mujoco_ant_training_ppo_cuda_benchmark
        PRIVATE
        backprop_tools
        CUDA::cublas
        CLI11::CLI11
)
set_target_properties(rl_environments_mujoco_ant_training_ppo_cuda_benchmark PROPERTIES CUDA_ARCHITECTURES ${BACKPROP_TOOLS_BACKEND_CUDA_ARCHITECTURES})
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_ppo_cuda_benchmark)
BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_training_ppo_cuda_benchmark)
if(NOT WIN32)
# for some reason this does not work on windows and crashes with an access violation in MKL (strangely enough this works when linking tensorboard_logger)
        BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_training_ppo_cuda_benchmark)
else()
        target_compile_definitions(rl_environments_mujoco_ant_training_ppo_cuda_benchmark PRIVATE BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
endif()
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_cuda_benchmark PRIVATE BACKPROP_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_cuda_benchmark PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
target_compile_definitions(rl_environments_mujoco_ant_training_ppo_cuda_benchmark PRIVATE BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_DISABLE_EVALUATION)
BACKPROP_TOOLS_INSTALL(rl_environments_mujoco_ant_training_ppo_cuda_benchmark)
endif()


if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS AND BACKPROP_TOOLS_ENABLE_HDF5 AND BACKPROP_TOOLS_ENABLE_TENSORBOARD AND BACKPROP_TOOLS_ENABLE_CLI11)
add_executable(
        rl_environments_mujoco_ant_training_ppo_cuda_full
        training_ppo.cu
)
target_link_libraries(
        rl_environments_mujoco_ant_training_ppo_cuda_full
        PRIVATE
        backprop_tools
        CUDA::cublas
        HighFive
        tensorboard_logger
        CLI11::CLI11
)
set_target_properties(rl_environments_mujoco_ant_training_ppo_cuda_full PROPERTIES CUDA_ARCHITECTURES ${BACKPROP_TOOLS_BACKEND_CUDA_ARCHITECTURES})
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_training_ppo_cuda_full)
BACKPROP_TOOLS_TARGET_LINK_MUJOCO(rl_environments_mujoco_ant_training_ppo_cuda_full)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_training_ppo_cuda_full)
BACKPROP_TOOLS_INSTALL(rl_environments_mujoco_ant_training_ppo_cuda_full)
endif()

# add_executable(
#         rl_environments_mujoco_ant_collection_cuda
#         collection.cu
# )
# target_link_libraries(
#         rl_environments_mujoco_ant_collection_cuda
#         PRIVATE
#         backprop_tools
#         HighFive
#         tensorboard_logger
#         mujoco
#         CUDA::cublas
# )
# set_target_properties(rl_environments_mujoco_ant_collection_cuda PROPERTIES CUDA_ARCHITECTURES ${BACKPROP_TOOLS_BACKEND_CUDA_ARCHITECTURES})
# BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_mujoco_ant_collection_cuda)
# BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_mujoco_ant_collection_cuda)
