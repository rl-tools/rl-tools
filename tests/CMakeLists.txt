option(BACKPROP_TOOLS_DOWNLOAD_GTEST OFF)

option(BACKPROP_TOOLS_TEST_LOCAL "Local dev" OFF)

option(BACKPROP_TOOLS_TEST_MACHINE_LENOVO_P1 "Flag to enable performance/timing tests" OFF)
if(BACKPROP_TOOLS_TEST_MACHINE_LENOVO_P1)
    add_compile_definitions(BACKPROP_TOOLS_TEST_MACHINE_LENOVO_P1)
endif()

option(BACKPROP_TOOLS_TEST_MACHINE_MACBOOK_M1 "Flag to enable performance/timing tests" OFF)
if(BACKPROP_TOOLS_TEST_MACHINE_MACBOOK_M1)
    add_compile_definitions(BACKPROP_TOOLS_TEST_MACHINE_MACBOOK_M1)
endif()

function(BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS TARGET)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message (STATUS "Adding debug definitions to ${TARGET}")
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_DEVICE_CUDA_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_GATHER_BATCH_CHECK_REPLAY_BUFFER_POSITION)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_ALGORITHMS_PPO_GAE_CHECK_TERMINATED_TRUNCATED)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_ENVIRONMENTS_MUJOCO_CHECK_INIT)
    endif()
endfunction()

#option(BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY "Visual Evaluation" ${BACKPROP_TOOLS_TEST_LOCAL})
option(BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY "Visual Evaluation" OFF)

option(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS "Visual Evaluation" OFF)
#option(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY "Visual Evaluation" ${BACKPROP_TOOLS_TEST_LOCAL})
option(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY "Visual Evaluation" OFF)

#option(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS "Visual Evaluation" ON)
option(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS "Visual Evaluation" OFF)
#option(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY "Visual Evaluation" ${BACKPROP_TOOLS_TEST_LOCAL})
option(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY "Visual Evaluation" OFF)

option(BACKPROP_TOOLS_TESTS_RL_ENVIRONMENTS_MULTIROTOR_UI_ENABLE "Multirotor UI" OFF)

set(INSTALL_GTEST OFF CACHE BOOL "")
set(BUILD_GMOCK OFF CACHE BOOL "")
add_subdirectory(lib/googletest EXCLUDE_FROM_ALL)
add_subdirectory(lib/cli11 EXCLUDE_FROM_ALL)
include(GoogleTest)

if(NOT APPLE)
#    add_subdirectory(lib/matplotlib EXCLUDE_FROM_ALL)
endif()

if(BACKPROP_TOOLS_ENABLE_HDF5)
add_subdirectory(lib/highfive EXCLUDE_FROM_ALL)
endif()
if(BACKPROP_TOOLS_ENABLE_TENSORBOARD)
add_subdirectory(lib/tensorboard EXCLUDE_FROM_ALL)
endif()

add_subdirectory(src/rl/environments/mujoco/ant)


if(BACKPROP_TOOLS_TESTS_ENABLE_EIGEN)
find_package(Eigen3 REQUIRED)
endif()
#find_package(OpenMP REQUIRED)
if(BACKPROP_TOOLS_TESTS_ENABLE_UI)
find_package(Boost REQUIRED)
endif()


if(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA)
#    set(CUTLASS_ENABLE_CUBLAS ON)
#    add_subdirectory(lib/cutlass EXCLUDE_FROM_ALL)
    add_subdirectory(src/nn/cuda)
    add_subdirectory(src/rl/cuda)
endif()

# RL Algorithm td3 test full training standalone
add_executable(
        test_rl_algorithms_td3_full_training_standalone
        src/rl/algorithms/td3/full_training_standalone.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training_standalone
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_rl_algorithms_td3_full_training_standalone)
target_compile_definitions(test_rl_algorithms_td3_full_training_standalone PRIVATE BACKPROP_TOOLS_TESTS_RL_ALGORITHMS_TD3_FULL_TRAINING_STANDALONE_BASIC_DISABLE_TENSORBOARD)

if(BACKPROP_TOOLS_ENABLE_HDF5 AND BACKPROP_TOOLS_ENABLE_TENSORBOARD)
# Neural Network Test MLP
add_executable(
        test_neural_network_mlp
        src/nn/test_mlp.cpp
)
target_link_libraries(
        test_neural_network_mlp
        backprop_tools
        GTest::gtest_main
        HighFive
)
#BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_neural_network_mlp)
gtest_discover_tests(test_neural_network_mlp)

# Neural Network Test Full Training MLP
add_executable(
        test_neural_network_full_training_mlp
        src/nn/full_training_mlp.cpp
)
target_link_libraries(
        test_neural_network_full_training_mlp
        backprop_tools
        GTest::gtest_main
        HighFive
)
#BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_neural_network_full_training_mlp)
gtest_discover_tests(test_neural_network_full_training_mlp)

# Neural Network Persistence Test
add_executable(
        test_neural_network_persistence
        src/nn/persisting.cpp
)
target_link_libraries(
        test_neural_network_persistence
        backprop_tools
        GTest::gtest_main
        HighFive
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_neural_network_persistence)
gtest_discover_tests(test_neural_network_persistence)

# RL Environment Pendulum test
add_executable(
        test_rl_environments_pendulum
        src/rl/environments/pendulum/pendulum.cpp
)
target_link_libraries(
        test_rl_environments_pendulum
        backprop_tools
        GTest::gtest_main
        HighFive
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_environments_pendulum)
gtest_discover_tests(test_rl_environments_pendulum)


if(BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY)
    # RL Environment Pendulum Render test

    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    add_executable(
            test_rl_environments_pendulum_render
            src/rl/environments/pendulum/pendulum_render.cpp
    )
    target_link_libraries(
            test_rl_environments_pendulum_render
            backprop_tools
            GTest::gtest_main
            Qt5::Widgets
    )
    BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_environments_pendulum_render)
    gtest_discover_tests(test_rl_environments_pendulum_render)
endif()

# RL Algorithm off_policy_runner test
add_executable(
        test_rl_algorithms_td3_off_policy_runner
        src/rl/algorithms/off_policy_runner.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_off_policy_runner
        backprop_tools
        GTest::gtest_main
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_algorithms_td3_off_policy_runner)
gtest_discover_tests(test_rl_algorithms_td3_off_policy_runner)

# RL Algorithm td3 test first stage mlp
add_executable(
        test_rl_algorithms_td3_first_stage_mlp
        src/rl/algorithms/td3/first_stage_mlp.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_first_stage_mlp
        backprop_tools
        GTest::gtest_main
        HighFive
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_algorithms_td3_first_stage_mlp)
gtest_discover_tests(test_rl_algorithms_td3_first_stage_mlp)


# RL Algorithm td3 test second stage MLP
add_executable(
        test_rl_algorithms_td3_second_stage_mlp
        src/rl/algorithms/td3/second_stage_mlp.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_second_stage_mlp
        backprop_tools
        GTest::gtest_main
        HighFive
)
if(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS)
    target_link_libraries(
            test_rl_algorithms_td3_second_stage_mlp
            matplotlib_cpp
    )
    target_include_directories(test_rl_algorithms_td3_second_stage_mlp PRIVATE lib)
    target_compile_definitions(test_rl_algorithms_td3_second_stage_mlp PRIVATE BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS)
endif()
if(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY)
    target_link_libraries(
            test_rl_algorithms_td3_second_stage_mlp
            Qt5::Widgets
    )
    target_compile_definitions(test_rl_algorithms_td3_second_stage_mlp PRIVATE BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY)
endif()
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_algorithms_td3_second_stage_mlp)
gtest_discover_tests(test_rl_algorithms_td3_second_stage_mlp)

# RL Algorithm td3 test full training
add_executable(
        test_rl_algorithms_td3_full_training_debug
        src/rl/algorithms/td3/full_training_test.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training_debug
        PRIVATE
        backprop_tools
        GTest::gtest_main
        tensorboard_logger
)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_rl_algorithms_td3_full_training_debug)

if(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS)
    target_link_libraries(
            test_rl_algorithms_td3_full_training_debug
            matplotlib_cpp
    )
    target_include_directories(test_rl_algorithms_td3_full_training_debug PRIVATE lib)
    target_compile_definitions(test_rl_algorithms_td3_full_training_debug PRIVATE BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS)
endif()
if(BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY)
    target_link_libraries(
            test_rl_algorithms_td3_full_training_debug
            Qt5::Widgets
    )
    target_compile_definitions(test_rl_algorithms_td3_full_training_debug PRIVATE BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY)
endif()
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_algorithms_td3_full_training_debug)
target_compile_definitions(test_rl_algorithms_td3_full_training_debug PRIVATE BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_DEBUG)
gtest_discover_tests(test_rl_algorithms_td3_full_training_debug)


# RL Algorithm td3 test full training
add_executable(
        test_rl_algorithms_td3_full_training
        src/rl/algorithms/td3/full_training_test.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training
        PRIVATE
        backprop_tools
        GTest::gtest_main
        tensorboard_logger
)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_rl_algorithms_td3_full_training)
gtest_discover_tests(test_rl_algorithms_td3_full_training)

# RL Algorithm td3 test full training dummy dependencies compilation without any deps
add_executable(
        test_rl_algorithms_td3_full_training_dummy_dep_compilation
        src/rl/algorithms/td3/full_training_dummy_dep.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training_dummy_dep_compilation
        backprop_tools
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
target_compile_definitions(test_rl_algorithms_td3_full_training_dummy_dep_training PRIVATE BACKPROP_TOOLS_OPERATIONS_CPU)
target_link_libraries(
        test_rl_algorithms_td3_full_training_dummy_dep_training
        backprop_tools
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
#        backprop_tools
#        GTest::gtest_main
#        HighFive
#        matplotlib_cpp
#        TBB::tbb
#)
#target_include_directories(test_rl_algorithms_td3_pendulum_training_stats PRIVATE lib)
#gtest_discover_tests(test_rl_algorithms_td3_pendulum_training_stats)


# Environment Multirotor test
add_executable(
        test_rl_environments_multirotor_multirotor
        src/rl/environments/multirotor/multirotor.cpp
)
target_link_libraries(
        test_rl_environments_multirotor_multirotor
        backprop_tools
        GTest::gtest_main
        HighFive
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_environments_multirotor_multirotor)
gtest_discover_tests(test_rl_environments_multirotor_multirotor)



function(factory_test_rl_environments_multirotor TARGET DEBUG)
# Multirotor Training test
add_executable(
        ${TARGET}
        src/rl/environments/multirotor_training/multirotor_training.cpp
)
target_link_libraries(
        ${TARGET}
        PRIVATE
        backprop_tools
        GTest::gtest_main
        HighFive
        tensorboard_logger
#        Boost::headers
)
BACKPROP_TOOLS_TARGET_LINK_BLAS(${TARGET})
target_include_directories(
        ${TARGET}
        PRIVATE
        lib/tensorboard/include
        lib/nlohmann/include
)
if(${DEBUG})
    target_compile_definitions(${TARGET} PRIVATE BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_DEBUG)
    BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(${TARGET})
endif()
gtest_discover_tests(${TARGET})
endfunction()

factory_test_rl_environments_multirotor(test_rl_environments_multirotor_training_debug true)
factory_test_rl_environments_multirotor(test_rl_environments_multirotor_training false)

# Neural Network Test Conversions
add_executable(
        test_neural_network_conversions
        src/nn/conversions.cpp
)
target_link_libraries(
        test_neural_network_conversions
        PRIVATE
        backprop_tools
        GTest::gtest_main
)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_neural_network_conversions)
gtest_discover_tests(test_neural_network_conversions)




if(BACKPROP_TOOLS_TESTS_ENABLE_EIGEN)
add_executable(
        test_neural_network_benchmark
        src/nn/benchmark.cpp
)
target_link_libraries(
        test_neural_network_benchmark
        PRIVATE
        backprop_tools
        GTest::gtest_main
        Eigen3::Eigen
#        CUDA::cudart
)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_neural_network_benchmark)
gtest_discover_tests(test_neural_network_benchmark)
endif()

# Multirotor UI test
if(${BACKPROP_TOOLS_TESTS_RL_ENVIRONMENTS_MULTIROTOR_UI_ENABLE})
add_executable(
        test_rl_environments_multirotor_ui_test
        src/rl/environments/multirotor/ui.cpp
)
target_link_libraries(
        test_rl_environments_multirotor_ui_test
        backprop_tools
        GTest::gtest_main
        Boost::headers
        HighFive
)
target_include_directories(
        test_rl_environments_multirotor_ui_test
        PRIVATE
        lib/nlohmann/include
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_environments_multirotor_ui_test)
gtest_discover_tests(test_rl_environments_multirotor_ui_test)
endif()

# Container tests
add_executable(
        test_container
        src/container/container.cpp
)
target_link_libraries(
        test_container
        backprop_tools
        GTest::gtest_main
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_container)
gtest_discover_tests(test_container)

add_executable(
        test_rl_components_replay_buffer
        src/rl/components/replay_buffer.cpp
)
target_link_libraries(
        test_rl_components_replay_buffer
        PRIVATE
        backprop_tools
        GTest::gtest_main
        HighFive
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_components_replay_buffer)
gtest_discover_tests(test_rl_components_replay_buffer)

# Container tests
add_executable(
        test_container_persist_code_store
        src/container/persist_code_store.cpp
)
target_link_libraries(
        test_container_persist_code_store
        backprop_tools
        GTest::gtest_main
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_container_persist_code_store)
gtest_discover_tests(test_container_persist_code_store)

## Container tests
#add_executable(
#        test_container_persist_code_load
#        src/container/persist_code_load.cpp
#)
#target_link_libraries(
#        test_container_persist_code_load
#        backprop_tools
#        GTest::gtest_main
#)
#BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_container_persist_code_load)
#gtest_discover_tests(test_container_persist_code_load)

# ARM tests
add_executable(
        test_nn_arm
        src/nn/arm.cpp
)
target_link_libraries(
        test_nn_arm
        backprop_tools
        GTest::gtest_main
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_nn_arm)
gtest_discover_tests(test_nn_arm)

# On Policy Runner tests
add_executable(
        test_rl_components_on_policy_runner
        src/rl/components/on_policy_runner.cpp
)
target_link_libraries(
        test_rl_components_on_policy_runner
        backprop_tools
        GTest::gtest_main
        HighFive
)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_components_on_policy_runner)
gtest_discover_tests(test_rl_components_on_policy_runner)

# PPO tests
add_executable(
        test_rl_algorithms_ppo
        src/rl/algorithms/ppo/ppo.cpp
)
target_link_libraries(
        test_rl_algorithms_ppo
        PRIVATE
        backprop_tools
        GTest::gtest_main
        HighFive
        tensorboard_logger
)
gtest_discover_tests(test_rl_algorithms_ppo)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_algorithms_ppo)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_rl_algorithms_ppo)


add_executable(
        test_rl_components_running_normalizer
        src/rl/components/running_normalizer.cpp
)
target_link_libraries(
        test_rl_components_running_normalizer
        PRIVATE
        backprop_tools
        GTest::gtest_main
        HighFive
        tensorboard_logger
)
gtest_discover_tests(test_rl_components_running_normalizer)
BACKPROP_TOOLS_TESTS_ADD_DEBUG_DEFINITIONS(test_rl_components_running_normalizer)
BACKPROP_TOOLS_TARGET_LINK_BLAS(test_rl_components_running_normalizer)

add_subdirectory(src/rl/algorithms/td3/wasm)
add_subdirectory(src/rl/algorithms/td3/arm)
add_subdirectory(src/random)
endif()
