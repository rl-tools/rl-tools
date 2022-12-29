set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

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


include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_subdirectory(lib/matplotlib)
add_subdirectory(lib/highfive)

#set(CMAKE_CXX_STANDARD 17)

# Neural Network Test
add_executable(
        test_neural_network
        src/nn/test.cpp
)
target_link_libraries(
        test_neural_network
        layer_in_c
        GTest::gtest_main
        HighFive
)
gtest_discover_tests(test_neural_network)

# Neural Network Test Full Training
add_executable(
        test_neural_network_full_training
        src/nn/full_training.cpp
)
target_link_libraries(
        test_neural_network_full_training
        layer_in_c
        GTest::gtest_main
        HighFive
)
gtest_discover_tests(test_neural_network_full_training)

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
        src/rl/environments/pendulum.cpp
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
            src/rl/environments/pendulum_render.cpp
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

# RL Algorithm td3 test first stage
add_executable(
        test_rl_algorithms_td3_first_stage
        src/rl/algorithms/td3/first_stage.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_first_stage
        layer_in_c
        GTest::gtest_main
        HighFive
)
gtest_discover_tests(test_rl_algorithms_td3_first_stage)

# RL Algorithm td3 test second stage
add_executable(
        test_rl_algorithms_td3_second_stage
        src/rl/algorithms/td3/second_stage.cpp
)
target_link_libraries(
        test_rl_algorithms_td3_second_stage
        layer_in_c
        GTest::gtest_main
        HighFive
)
if(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS)
    target_link_libraries(
            test_rl_algorithms_td3_second_stage
            matplotlib_cpp
    )
    target_include_directories(test_rl_algorithms_td3_second_stage PRIVATE lib)
    target_compile_definitions(test_rl_algorithms_td3_second_stage PRIVATE LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS)
endif()
if(LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY)
    target_link_libraries(
            test_rl_algorithms_td3_second_stage
            Qt5::Widgets
    )
    target_compile_definitions(test_rl_algorithms_td3_second_stage PRIVATE LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY)
endif()
gtest_discover_tests(test_rl_algorithms_td3_second_stage)

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





