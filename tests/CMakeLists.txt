set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

option(LAYER_IN_C_DOWNLOAD_GTEST OFF)

option(LAYER_IN_C_TEST_LOCAL "Local dev" OFF)

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
add_subdirectory(lib/googletest)
include(GoogleTest)


if(NOT APPLE)
add_subdirectory(lib/matplotlib)
endif()
add_subdirectory(lib/highfive)

#set(CMAKE_CXX_STANDARD 17)

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
gtest_discover_tests(test_rl_algorithms_td3_second_stage_mlp)

# RL Algorithm td3 test full training
add_executable(
        test_rl_algorithms_td3_full_training
        src/rl/algorithms/td3/full_training.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_full_training
        layer_in_c
        GTest::gtest_main
)
if(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS)
    target_link_libraries(
            test_rl_algorithms_td3_full_training
            matplotlib_cpp
    )
    target_include_directories(test_rl_algorithms_td3_full_training PRIVATE lib)
    target_compile_definitions(test_rl_algorithms_td3_full_training PRIVATE LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS)
endif()
if(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY)
    target_link_libraries(
            test_rl_algorithms_td3_full_training
            Qt5::Widgets
    )
    target_compile_definitions(test_rl_algorithms_td3_full_training PRIVATE LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY)
endif()
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
gtest_discover_tests(test_rl_environments_multirotor_multirotor)



add_subdirectory(lib/tensorboard)
# Multirotor Training test
add_executable(
        test_rl_environments_multirotor_training_test
        src/rl/environments/multirotor_training/multirotor_training.cpp
)
target_link_libraries(
        test_rl_environments_multirotor_training_test
        layer_in_c
        GTest::gtest_main
        HighFive
        tensorboard_logger
)
target_include_directories(
        test_rl_environments_multirotor_training_test
        PRIVATE
        lib/tensorboard/include
)
gtest_discover_tests(test_rl_environments_multirotor_training_test)

