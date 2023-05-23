if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS AND BACKPROP_TOOLS_ENABLE_HDF5 AND BACKPROP_TOOLS_ENABLE_TENSORBOARD AND BACKPROP_TOOLS_ENABLE_JSON)
    function(factory_test_rl_environments_multirotor TARGET DEBUG)
        # Multirotor Training test
        add_executable(${TARGET} training.cpp)
        target_link_libraries(
                ${TARGET}
                PRIVATE
                backprop_tools
                HighFive
                tensorboard_logger
                nlohmann_json::nlohmann_json
        )
        BACKPROP_TOOLS_TARGET_LINK_BLAS(${TARGET})
        BACKPROP_TOOLS_ADD_DEFINITIONS(${TARGET})
        if(${DEBUG})
            target_compile_definitions(${TARGET} PRIVATE BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_DEBUG)
        endif()
    endfunction()

    factory_test_rl_environments_multirotor(rl_environments_multirotor_training_td3_debug true)
    factory_test_rl_environments_multirotor(rl_environments_multirotor_training_td3 false)
endif()


add_executable(rl_environments_multirotor_training_checkpoint_test evaluation.cpp)
target_link_libraries(
        rl_environments_multirotor_training_checkpoint_test
        PRIVATE
        backprop_tools
)
