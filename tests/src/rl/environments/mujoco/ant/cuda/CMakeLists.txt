
set(CMAKE_CUDA_ARCHITECTURES "native")
project(test_rl_environments_mujoco_ant_cuda LANGUAGES CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

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
        GTest::gtest_main
        HighFive
        tensorboard_logger
        mujoco
        CUDA::cublas
)
gtest_discover_tests(test_rl_environments_mujoco_ant_training_ppo_cuda)
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
gtest_discover_tests(test_rl_environments_mujoco_ant_collection_cuda)
LAYER_IN_C_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_environments_mujoco_ant_collection_cuda)
LAYER_IN_C_TARGET_LINK_BLAS(test_rl_environments_mujoco_ant_collection_cuda)
