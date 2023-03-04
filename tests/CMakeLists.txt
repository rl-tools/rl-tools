option(LAYER_IN_C_DOWNLOAD_GTEST OFF)

option(LAYER_IN_C_TEST_LOCAL "Local dev" OFF)

option(LAYER_IN_C_TEST_MACHINE_LENOVO_P1 "Flag to enable performance/timing tests" OFF)
if(LAYER_IN_C_TEST_MACHINE_LENOVO_P1)
    add_compile_definitions(LAYER_IN_C_TEST_MACHINE_LENOVO_P1)
endif()

option(LAYER_IN_C_TEST_MACHINE_MACBOOK_M1 "Flag to enable performance/timing tests" OFF)
if(LAYER_IN_C_TEST_MACHINE_MACBOOK_M1)
    add_compile_definitions(LAYER_IN_C_TEST_MACHINE_MACBOOK_M1)
endif()

#option(LAYER_IN_C_DEBUG "Debug" ON)

function(LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS TARGET)
#    if(LAYER_IN_C_DEBUG)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_DEVICE_CUDA_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_CONTAINER_CHECK_BOUNDS)
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC)
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_CONTAINER_MALLOC_INIT_NAN)
        target_compile_definitions(${TARGET} PRIVATE -DLAYER_IN_C_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT)
    endif()
endfunction()

#option(LAYER_IN_C_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY "Visual Evaluation" ${LAYER_IN_C_TEST_LOCAL})
option(LAYER_IN_C_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY "Visual Evaluation" OFF)

option(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS "Visual Evaluation" OFF)
#option(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY "Visual Evaluation" ${LAYER_IN_C_TEST_LOCAL})
option(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY "Visual Evaluation" OFF)

#option(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS "Visual Evaluation" ON)
option(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS "Visual Evaluation" OFF)
#option(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY "Visual Evaluation" ${LAYER_IN_C_TEST_LOCAL})
option(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY "Visual Evaluation" OFF)

set(INSTALL_GTEST OFF CACHE BOOL "")
set(BUILD_GMOCK OFF CACHE BOOL "")
add_subdirectory(lib/googletest EXCLUDE_FROM_ALL)
include(GoogleTest)

if(NOT APPLE)
    add_subdirectory(lib/matplotlib EXCLUDE_FROM_ALL)
endif()
add_subdirectory(lib/highfive EXCLUDE_FROM_ALL)
add_subdirectory(lib/tensorboard EXCLUDE_FROM_ALL)

find_package(Eigen3 REQUIRED)
#find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)

#find_package(CUDAToolkit REQUIRED)

if(LAYER_IN_C_BACKEND_ENABLE_CUDA)
    set(CUTLASS_ENABLE_CUBLAS ON)
    add_subdirectory(lib/cutlass EXCLUDE_FROM_ALL)
    add_subdirectory(src/nn/cuda)
    add_subdirectory(src/rl/cuda)
endif()

# Neural Network Test MLP
add_executable(
        test_neural_network_mlp
        src/nn/test_mlp.cpp
)
target_link_libraries(
        test_neural_network_mlp
        layer_in_c
        GTest::gtest_main
        HighFive
)
#LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_neural_network_mlp)
gtest_discover_tests(test_neural_network_mlp)

# Neural Network Test Full Training MLP
add_executable(
        test_neural_network_full_training_mlp
        src/nn/full_training_mlp.cpp
)
target_link_libraries(
        test_neural_network_full_training_mlp
        layer_in_c
        GTest::gtest_main
        HighFive
)
#LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_neural_network_full_training_mlp)
gtest_discover_tests(test_neural_network_full_training_mlp)

# Neural Network Persistence Test
add_executable(
        test_neural_network_persistence
        src/nn/persisting.cpp
)
target_link_libraries(
        test_neural_network_persistence
        layer_in_c
        GTest::gtest_main
        HighFive
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_neural_network_persistence)
gtest_discover_tests(test_neural_network_persistence)

# RL Environment Pendulum test
add_executable(
        test_rl_environments_pendulum
        src/rl/environments/pendulum/pendulum.cpp
)
target_link_libraries(
        test_rl_environments_pendulum
        layer_in_c
        GTest::gtest_main
        HighFive
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_environments_pendulum)
gtest_discover_tests(test_rl_environments_pendulum)


if(LAYER_IN_C_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY)
    # RL Environment Pendulum Render test

    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    add_executable(
            test_rl_environments_pendulum_render
            src/rl/environments/pendulum/pendulum_render.cpp
    )
    target_link_libraries(
            test_rl_environments_pendulum_render
            layer_in_c
            GTest::gtest_main
            Qt5::Widgets
    )
    LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_environments_pendulum_render)
    gtest_discover_tests(test_rl_environments_pendulum_render)
endif()

# RL Algorithm off_policy_runner test
add_executable(
        test_rl_algorithms_td3_off_policy_runner
        src/rl/algorithms/off_policy_runner.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_off_policy_runner
        layer_in_c
        GTest::gtest_main
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_algorithms_td3_off_policy_runner)
gtest_discover_tests(test_rl_algorithms_td3_off_policy_runner)

# RL Algorithm td3 test first stage mlp
add_executable(
        test_rl_algorithms_td3_first_stage_mlp
        src/rl/algorithms/td3/first_stage_mlp.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_first_stage_mlp
        layer_in_c
        GTest::gtest_main
        HighFive
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_algorithms_td3_first_stage_mlp)
gtest_discover_tests(test_rl_algorithms_td3_first_stage_mlp)


# RL Algorithm td3 test second stage MLP
add_executable(
        test_rl_algorithms_td3_second_stage_mlp
        src/rl/algorithms/td3/second_stage_mlp.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_second_stage_mlp
        layer_in_c
        GTest::gtest_main
        HighFive
)
if(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS)
    target_link_libraries(
            test_rl_algorithms_td3_second_stage_mlp
            matplotlib_cpp
    )
    target_include_directories(test_rl_algorithms_td3_second_stage_mlp PRIVATE lib)
    target_compile_definitions(test_rl_algorithms_td3_second_stage_mlp PRIVATE LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS)
endif()
if(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY)
    target_link_libraries(
            test_rl_algorithms_td3_second_stage_mlp
            Qt5::Widgets
    )
    target_compile_definitions(test_rl_algorithms_td3_second_stage_mlp PRIVATE LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY)
endif()
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_algorithms_td3_second_stage_mlp)
gtest_discover_tests(test_rl_algorithms_td3_second_stage_mlp)

# RL Algorithm td3 test full training
add_executable(
        test_rl_algorithms_td3_full_training_debug
        src/rl/algorithms/td3/full_training.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training_debug
        PRIVATE
        layer_in_c
        GTest::gtest_main
        tensorboard_logger
)
LAYER_IN_C_TARGET_LINK_BLAS(test_rl_algorithms_td3_full_training_debug)

if(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS)
    target_link_libraries(
            test_rl_algorithms_td3_full_training_debug
            matplotlib_cpp
    )
    target_include_directories(test_rl_algorithms_td3_full_training_debug PRIVATE lib)
    target_compile_definitions(test_rl_algorithms_td3_full_training_debug PRIVATE LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS)
endif()
if(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY)
    target_link_libraries(
            test_rl_algorithms_td3_full_training_debug
            Qt5::Widgets
    )
    target_compile_definitions(test_rl_algorithms_td3_full_training_debug PRIVATE LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY)
endif()
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_algorithms_td3_full_training_debug)
target_compile_definitions(test_rl_algorithms_td3_full_training_debug PRIVATE LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_DEBUG)
gtest_discover_tests(test_rl_algorithms_td3_full_training_debug)


# RL Algorithm td3 test full training
add_executable(
        test_rl_algorithms_td3_full_training
        src/rl/algorithms/td3/full_training.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training
        PRIVATE
        layer_in_c
        GTest::gtest_main
        tensorboard_logger
)
LAYER_IN_C_TARGET_LINK_BLAS(test_rl_algorithms_td3_full_training)
gtest_discover_tests(test_rl_algorithms_td3_full_training)

# RL Algorithm td3 test full training dummy dependencies compilation without any deps
add_executable(
        test_rl_algorithms_td3_full_training_dummy_dep_compilation
        src/rl/algorithms/td3/full_training_dummy_dep.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training_dummy_dep_compilation
        layer_in_c
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
target_compile_definitions(test_rl_algorithms_td3_full_training_dummy_dep_training PRIVATE LAYER_IN_C_OPERATIONS_CPU)
target_link_libraries(
        test_rl_algorithms_td3_full_training_dummy_dep_training
        layer_in_c
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
#        layer_in_c
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
        layer_in_c
        GTest::gtest_main
        HighFive
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_environments_multirotor_multirotor)
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
        layer_in_c
        GTest::gtest_main
        HighFive
        tensorboard_logger
        Boost::headers
)
LAYER_IN_C_TARGET_LINK_BLAS(${TARGET})
target_include_directories(
        ${TARGET}
        PRIVATE
        lib/tensorboard/include
        lib/nlohmann/include
)
if(${DEBUG})
    target_compile_definitions(${TARGET} PRIVATE LAYER_IN_C_TEST_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_DEBUG)
    LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(${TARGET})
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
        layer_in_c
        GTest::gtest_main
)
LAYER_IN_C_TARGET_LINK_BLAS(test_neural_network_conversions)
gtest_discover_tests(test_neural_network_conversions)




add_executable(
        test_neural_network_benchmark
        src/nn/benchmark.cpp
)
target_link_libraries(
        test_neural_network_benchmark
        PRIVATE
        layer_in_c
        GTest::gtest_main
        Eigen3::Eigen
#        CUDA::cudart
)
LAYER_IN_C_TARGET_LINK_BLAS(test_neural_network_benchmark)
gtest_discover_tests(test_neural_network_benchmark)

# Multirotor UI test
add_executable(
        test_rl_environments_multirotor_ui_test
        src/rl/environments/multirotor/ui.cpp
)
target_link_libraries(
        test_rl_environments_multirotor_ui_test
        layer_in_c
        GTest::gtest_main
        Boost::headers
        HighFive
)
target_include_directories(
        test_rl_environments_multirotor_ui_test
        PRIVATE
        lib/nlohmann/include
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_environments_multirotor_ui_test)
gtest_discover_tests(test_rl_environments_multirotor_ui_test)

# Container tests
add_executable(
        test_container
        src/container/container.cpp
)
target_link_libraries(
        test_container
        layer_in_c
        GTest::gtest_main
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_container)
gtest_discover_tests(test_container)

add_executable(
        test_rl_components_replay_buffer
        src/rl/components/replay_buffer.cpp
)
target_link_libraries(
        test_rl_components_replay_buffer
        PRIVATE
        layer_in_c
        GTest::gtest_main
        HighFive
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_components_replay_buffer)
gtest_discover_tests(test_rl_components_replay_buffer)

# Container tests
add_executable(
        test_container_persist_code_store
        src/container/persist_code_store.cpp
)
target_link_libraries(
        test_container_persist_code_store
        layer_in_c
        GTest::gtest_main
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_container_persist_code_store)
gtest_discover_tests(test_container_persist_code_store)

# Container tests
add_executable(
        test_container_persist_code_load
        src/container/persist_code_load.cpp
)
target_link_libraries(
        test_container_persist_code_load
        layer_in_c
        GTest::gtest_main
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_container_persist_code_load)
gtest_discover_tests(test_container_persist_code_load)

# ARM tests
add_executable(
        test_nn_arm
        src/nn/arm.cpp
)
target_link_libraries(
        test_nn_arm
        layer_in_c
        GTest::gtest_main
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_nn_arm)
gtest_discover_tests(test_nn_arm)

# On Policy Runner tests
add_executable(
        test_rl_components_on_policy_runner
        src/rl/components/on_policy_runner.cpp
)
target_link_libraries(
        test_rl_components_on_policy_runner
        layer_in_c
        GTest::gtest_main
        HighFive
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_components_on_policy_runner)
gtest_discover_tests(test_rl_components_on_policy_runner)

# PPO tests
add_executable(
        test_rl_algorithms_ppo
        src/rl/algorithms/ppo/ppo.cpp
)
target_link_libraries(
        test_rl_algorithms_ppo
        layer_in_c
        GTest::gtest_main
        HighFive
)
LAYER_IN_C_TEST_ADD_DEBUG_DEFINITIONS(test_rl_algorithms_ppo)
gtest_discover_tests(test_rl_algorithms_ppo)
