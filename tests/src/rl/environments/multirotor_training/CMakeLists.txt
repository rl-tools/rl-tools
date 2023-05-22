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
        BACKPROP_TOOLS_ADD_DEFINITIONS(${TARGET})
    endif()
    gtest_discover_tests(${TARGET})
endfunction()

factory_test_rl_environments_multirotor(test_rl_environments_multirotor_training_debug true)
factory_test_rl_environments_multirotor(test_rl_environments_multirotor_training false)

