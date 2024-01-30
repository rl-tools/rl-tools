option(RL_TOOLS_DOWNLOAD_GTEST OFF)

option(RL_TOOLS_TEST_LOCAL "Local dev" OFF)

#option(RL_TOOLS_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY "Visual Evaluation" ${RL_TOOLS_TEST_LOCAL})
option(RL_TOOLS_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY "Visual Evaluation" OFF)

option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS "Visual Evaluation" OFF)
#option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY "Visual Evaluation" ${RL_TOOLS_TEST_LOCAL})
option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY "Visual Evaluation" OFF)

#option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS "Visual Evaluation" ON)
option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS "Visual Evaluation" OFF)
#option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY "Visual Evaluation" ${RL_TOOLS_TEST_LOCAL})
option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY "Visual Evaluation" OFF)

option(RL_TOOLS_TESTS_RL_ENVIRONMENTS_MULTIROTOR_UI_ENABLE "Multirotor UI" OFF)

set(INSTALL_GTEST OFF CACHE BOOL "")
set(BUILD_GMOCK OFF CACHE BOOL "")
add_subdirectory(lib/googletest EXCLUDE_FROM_ALL)
include(GoogleTest)

set(RL_TOOLS_TESTS_DATA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/data)
add_library(rl_tools_tests INTERFACE)
target_link_libraries(rl_tools_tests INTERFACE GTest::gtest_main)
target_compile_definitions(rl_tools_tests INTERFACE RL_TOOLS_TESTS_DATA_PATH=${RL_TOOLS_TESTS_DATA_PATH})


if(NOT APPLE)
#    add_subdirectory(lib/matplotlib EXCLUDE_FROM_ALL)
endif()


# function to do target_compile_definitions(test_neural_network_mlp PRIVATE RL_TOOLS_TESTS_DATA_PATH=${CMAKE_CURRENT_SOURCE_DIR}/data)




if(RL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO)
add_subdirectory(src/rl/environments/mujoco/ant)
endif()


if(RL_TOOLS_TESTS_ENABLE_EIGEN)
find_package(Eigen3 REQUIRED)
endif()
#find_package(OpenMP REQUIRED)
if(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI)
find_package(Boost REQUIRED)
endif()


if(RL_TOOLS_BACKEND_ENABLE_CUDA)
#    set(CUTLASS_ENABLE_CUBLAS ON)
#    add_subdirectory(lib/cutlass EXCLUDE_FROM_ALL)
#    add_subdirectory(src/nn/cuda)
#    add_subdirectory(src/rl/cuda)
endif()

add_executable(
        test_nn_models_sequential
        src/nn/sequential.cpp
)
target_link_libraries(
        test_nn_models_sequential
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_nn_models_sequential)

if(RL_TOOLS_ENABLE_HDF5)
# Neural Network Test MLP
add_executable(
        test_neural_network_mlp
        src/nn/test_mlp.cpp
)
target_link_libraries(
        test_neural_network_mlp
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_neural_network_mlp)

# Neural Network Test Full Training MLP
add_executable(
        test_neural_network_full_training_mlp
        src/nn/full_training_mlp.cpp
)
target_link_libraries(
        test_neural_network_full_training_mlp
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_neural_network_full_training_mlp)

# Neural Network Persistence Test
add_executable(
        test_neural_network_persistence
        src/nn/persisting.cpp
)
target_link_libraries(
        test_neural_network_persistence
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_neural_network_persistence)

# RL Environment Pendulum test
add_executable(
        test_rl_environments_pendulum
        src/rl/environments/pendulum/pendulum.cpp
)
target_link_libraries(
        test_rl_environments_pendulum
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_rl_environments_pendulum)


if(RL_TOOLS_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY)
    # RL Environment Pendulum Render test

    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    add_executable(
            test_rl_environments_pendulum_render
            src/rl/environments/pendulum/pendulum_render.cpp
    )
    target_link_libraries(
            test_rl_environments_pendulum_render
            rl_tools
            rl_tools_tests
            Qt5::Widgets
    )
    gtest_discover_tests(test_rl_environments_pendulum_render)
endif()

# RL Algorithm off_policy_runner test
add_executable(
        test_rl_algorithms_td3_off_policy_runner
        src/rl/algorithms/off_policy_runner.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_off_policy_runner
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_rl_algorithms_td3_off_policy_runner)

# RL Algorithm td3 test first stage mlp
add_executable(
        test_rl_algorithms_td3_first_stage_mlp
        src/rl/algorithms/td3/first_stage_mlp.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_first_stage_mlp
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_rl_algorithms_td3_first_stage_mlp)


# RL Algorithm td3 test second stage MLP
add_executable(
        test_rl_algorithms_td3_second_stage_mlp
        src/rl/algorithms/td3/second_stage_mlp.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_second_stage_mlp
        rl_tools
        rl_tools_tests
)
if(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS)
    target_link_libraries(
            test_rl_algorithms_td3_second_stage_mlp
            matplotlib_cpp
    )
    target_include_directories(test_rl_algorithms_td3_second_stage_mlp PRIVATE lib)
    target_compile_definitions(test_rl_algorithms_td3_second_stage_mlp PRIVATE RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS)
endif()
if(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY)
    target_link_libraries(
            test_rl_algorithms_td3_second_stage_mlp
            Qt5::Widgets
    )
    target_compile_definitions(test_rl_algorithms_td3_second_stage_mlp PRIVATE RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY)
endif()
gtest_discover_tests(test_rl_algorithms_td3_second_stage_mlp)

# RL Algorithm td3 test full training
add_executable(
        test_rl_algorithms_td3_full_training_debug
        src/rl/algorithms/td3/full_training_test.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training_debug
        PRIVATE
        rl_tools
        rl_tools_tests
)

if(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS)
    target_link_libraries(
            test_rl_algorithms_td3_full_training_debug
            matplotlib_cpp
    )
    target_include_directories(test_rl_algorithms_td3_full_training_debug PRIVATE lib)
    target_compile_definitions(test_rl_algorithms_td3_full_training_debug PRIVATE RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS)
endif()
if(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY)
    target_link_libraries(
            test_rl_algorithms_td3_full_training_debug
            Qt5::Widgets
    )
    target_compile_definitions(test_rl_algorithms_td3_full_training_debug PRIVATE RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY)
endif()
target_compile_definitions(test_rl_algorithms_td3_full_training_debug PRIVATE RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_DEBUG)
gtest_discover_tests(test_rl_algorithms_td3_full_training_debug)


# RL Algorithm td3 test full training
add_executable(
        test_rl_algorithms_td3_full_training
        src/rl/algorithms/td3/full_training_test.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training
        PRIVATE
        rl_tools
        rl_tools_tests
)
target_compile_options(test_rl_algorithms_td3_full_training PRIVATE -Ofast)
gtest_discover_tests(test_rl_algorithms_td3_full_training)

# RL Algorithm td3 test full training dummy dependencies compilation without any deps
add_executable(
        test_rl_algorithms_td3_full_training_dummy_dep_compilation
        src/rl/algorithms/td3/full_training_dummy_dep.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training_dummy_dep_compilation
        rl_tools
        rl_tools_tests
)
add_test(
        NAME test_rl_algorithms_td3_full_training_dummy_dep_compilation
        COMMAND test_rl_algorithms_td3_full_training_dummy_dep_compilation
)

# RL Algorithm td3 test full training dummy dependencies training by just replacing the context
add_executable(
        test_rl_algorithms_td3_full_training_dummy_dep_training
        src/rl/algorithms/td3/full_training_dummy_dep.cpp
)
target_compile_definitions(test_rl_algorithms_td3_full_training_dummy_dep_training PRIVATE RL_TOOLS_OPERATIONS_CPU)
target_link_libraries(
        test_rl_algorithms_td3_full_training_dummy_dep_training
        rl_tools
        rl_tools_tests
)
add_test(
        NAME test_rl_algorithms_td3_full_training_dummy_dep_training
        COMMAND test_rl_algorithms_td3_full_training_dummy_dep_training
)

## RL Algorithm td3 test training stats
#find_package(TBB REQUIRED)
#add_executable(
#        test_rl_algorithms_td3_pendulum_training_stats
#        src/rl/algorithms/td3/pendulum_training_stats.cpp
#)
#target_link_libraries(
#        test_rl_algorithms_td3_pendulum_training_stats
#        rl_tools
#        GTest::gtest_main
#        HighFive
#        matplotlib_cpp
#        TBB::tbb
#)
#target_include_directories(test_rl_algorithms_td3_pendulum_training_stats PRIVATE lib)
#gtest_discover_tests(test_rl_algorithms_td3_pendulum_training_stats)



# Neural Network Test Conversions
add_executable(
        test_neural_network_conversions
        src/nn/conversions.cpp
)
target_link_libraries(
        test_neural_network_conversions
        PRIVATE
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_neural_network_conversions)




if(RL_TOOLS_TESTS_ENABLE_EIGEN)
add_executable(
        test_neural_network_benchmark
        src/nn/benchmark.cpp
)
target_link_libraries(
        test_neural_network_benchmark
        PRIVATE
        rl_tools
        rl_tools_tests
        Eigen3::Eigen
#        CUDA::cudart
)
gtest_discover_tests(test_neural_network_benchmark)
endif()


# Container tests
add_executable(
        test_container
        src/container/container.cpp
)
target_link_libraries(
        test_container
        PRIVATE
        rl_tools_core
        rl_tools_tests
)
gtest_discover_tests(test_container)

add_executable(
        test_rl_components_replay_buffer
        src/rl/components/replay_buffer.cpp
)
target_link_libraries(
        test_rl_components_replay_buffer
        PRIVATE
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_rl_components_replay_buffer)

# Container tests
add_executable(
        test_container_persist_code_store
        src/container/persist_code_store.cpp
)
target_link_libraries(
        test_container_persist_code_store
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_container_persist_code_store)

## Container tests
#add_executable(
#        test_container_persist_code_load
#        src/container/persist_code_load.cpp
#)
#target_link_libraries(
#        test_container_persist_code_load
#        rl_tools
#        GTest::gtest_main
#)
#gtest_discover_tests(test_container_persist_code_load)

# ARM tests
add_executable(
        test_nn_arm
        src/nn/arm.cpp
)
target_link_libraries(
        test_nn_arm
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_nn_arm)

# On Policy Runner tests
add_executable(
        test_rl_components_on_policy_runner
        src/rl/components/on_policy_runner.cpp
)
target_link_libraries(
        test_rl_components_on_policy_runner
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_rl_components_on_policy_runner)

# PPO tests
if(RL_TOOLS_ENABLE_TENSORBOARD)
add_executable(
        test_rl_algorithms_ppo
        src/rl/algorithms/ppo/ppo.cpp
)
target_link_libraries(
        test_rl_algorithms_ppo
        PRIVATE
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_rl_algorithms_ppo)
endif()


add_executable(
        test_rl_components_running_normalizer
        src/rl/components/running_normalizer.cpp
)
target_link_libraries(
        test_rl_components_running_normalizer
        PRIVATE
        rl_tools
        rl_tools_tests
)
gtest_discover_tests(test_rl_components_running_normalizer)

add_subdirectory(src/random)
endif()

add_subdirectory(src/nn/layers)


add_subdirectory(src/nn_models/sequential)
add_subdirectory(src/nn_models/mlp)
add_subdirectory(src/rl/environments/car)
add_subdirectory(src/rl/environments/acrobot)
if(RL_TOOLS_BACKEND_ENABLE_CUDA)
add_subdirectory(src/rl/cuda)
endif()
add_subdirectory(src/logging)
add_subdirectory(src/math)
add_subdirectory(src/container)
add_subdirectory(src/rl/algorithms/sac)

