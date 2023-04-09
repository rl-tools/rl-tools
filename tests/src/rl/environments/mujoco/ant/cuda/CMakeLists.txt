
project(test_rl_environments_mujoco_ant_cuda LANGUAGES CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
message("CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message("CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")

add_executable(
        test_rl_environments_mujoco_ant_throughput_cuda
        throughput.cu
)
target_link_libraries(
        test_rl_environments_mujoco_ant_throughput_cuda
        PRIVATE
        layer_in_c
        GTest::gtest_main
        HighFive
        tensorboard_logger
        mujoco
        CUDA::cublas
)
set_target_properties(test_rl_environments_mujoco_ant_throughput_cuda PROPERTIES CUDA_ARCHITECTURES ${LAYER_IN_C_BACKEND_CUDA_ARCHITECTURES})
gtest_discover_tests(test_rl_environments_mujoco_ant_throughput_cuda)
LAYER_IN_C_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_environments_mujoco_ant_throughput_cuda)
LAYER_IN_C_TARGET_LINK_BLAS(test_rl_environments_mujoco_ant_throughput_cuda)



add_executable(
        test_rl_environments_mujoco_ant_training_ppo_cuda
        training_ppo.cu
)
target_link_libraries(
        test_rl_environments_mujoco_ant_training_ppo_cuda
        PRIVATE
        layer_in_c
        HighFive
        tensorboard_logger
        mujoco
        CUDA::cublas
        CLI11::CLI11
)
set_target_properties(test_rl_environments_mujoco_ant_training_ppo_cuda PROPERTIES CUDA_ARCHITECTURES ${LAYER_IN_C_BACKEND_CUDA_ARCHITECTURES})
LAYER_IN_C_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_environments_mujoco_ant_training_ppo_cuda)
LAYER_IN_C_TARGET_LINK_BLAS(test_rl_environments_mujoco_ant_training_ppo_cuda)

add_executable(
        test_rl_environments_mujoco_ant_collection_cuda
        collection.cu
)
target_link_libraries(
        test_rl_environments_mujoco_ant_collection_cuda
        PRIVATE
        layer_in_c
        GTest::gtest_main
        HighFive
        tensorboard_logger
        mujoco
        CUDA::cublas
)
set_target_properties(test_rl_environments_mujoco_ant_collection_cuda PROPERTIES CUDA_ARCHITECTURES ${LAYER_IN_C_BACKEND_CUDA_ARCHITECTURES})
gtest_discover_tests(test_rl_environments_mujoco_ant_collection_cuda)
LAYER_IN_C_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_environments_mujoco_ant_collection_cuda)
LAYER_IN_C_TARGET_LINK_BLAS(test_rl_environments_mujoco_ant_collection_cuda)
