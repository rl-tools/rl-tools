add_executable(
        test_nn_models_sequential
        sequential.cpp
)
target_link_libraries(
        test_nn_models_sequential
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_nn_models_sequential)

if(RL_TOOLS_ENABLE_HDF5)
    # Neural Network Test MLP
    add_executable(
            test_neural_network_mlp
            test_mlp.cpp
    )
    target_link_libraries(
            test_neural_network_mlp
            rl_tools
            rl_tools_test
    )
    gtest_discover_tests(test_neural_network_mlp)
    # Neural Network Test Full Training MLP
    add_executable(
            test_neural_network_full_training_mlp
            full_training_mlp.cpp
    )
    target_link_libraries(
            test_neural_network_full_training_mlp
            rl_tools
            rl_tools_test
    )
    gtest_discover_tests(test_neural_network_full_training_mlp)

    # Neural Network Persistence Test
    add_executable(
            test_neural_network_persistence
            persisting.cpp
    )
    target_link_libraries(
            test_neural_network_persistence
            rl_tools
            rl_tools_test
    )
    gtest_discover_tests(test_neural_network_persistence)
    # Neural Network Test Conversions
    add_executable(
            test_neural_network_conversions
            conversions.cpp
    )
    target_link_libraries(
            test_neural_network_conversions
            PRIVATE
            rl_tools
            rl_tools_test
    )
    gtest_discover_tests(test_neural_network_conversions)




    if(RL_TOOLS_TESTS_ENABLE_EIGEN)
        add_executable(
                test_neural_network_benchmark
                benchmark.cpp
        )
        target_link_libraries(
                test_neural_network_benchmark
                PRIVATE
                rl_tools
                rl_tools_test
                Eigen3::Eigen
                #        CUDA::cudart
        )
        gtest_discover_tests(test_neural_network_benchmark)
    endif()
endif()
# ARM tests
add_executable(
        test_nn_arm
        arm.cpp
)
target_link_libraries(
        test_nn_arm
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_nn_arm)

add_subdirectory(layers)
if(RL_TOOLS_BACKEND_ENABLE_CUDA)
#    add_subdirectory(cuda)
endif()


