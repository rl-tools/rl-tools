if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS AND BACKPROP_TOOLS_ENABLE_HDF5 AND BACKPROP_TOOLS_ENABLE_TENSORBOARD)
    function(factory_test_rl_environments_multirotor TARGET DEBUG)
        # Multirotor Training test
        add_executable(${TARGET} training.cpp)
        target_link_libraries(
                ${TARGET}
                PRIVATE
                backprop_tools
                HighFive
                tensorboard_logger
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


#add_executable(rl_environments_multirotor_training_checkpoint_test evaluation.cpp)
#target_link_libraries(
#        rl_environments_multirotor_training_checkpoint_test
#        PRIVATE
#        backprop_tools
#)
#
#if(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_TD3_ENABLE_MULTI_CONFIG)
#if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS AND BACKPROP_TOOLS_ENABLE_HDF5 AND BACKPROP_TOOLS_ENABLE_TENSORBOARD)
#    foreach(ID RANGE 1 ${BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_TD3_NUM_CONFIGS})
#        set(TARGET_NAME rl_environments_multirotor_training_td3_${ID})
#        add_executable(${TARGET_NAME} training.cpp)
#
#        target_link_libraries(
#                ${TARGET_NAME}
#                PRIVATE
#                backprop_tools
#                HighFive
#                tensorboard_logger
#        )
#
#        BACKPROP_TOOLS_TARGET_LINK_BLAS(${TARGET_NAME})
#        BACKPROP_TOOLS_ADD_DEFINITIONS(${TARGET_NAME})
#
#        target_compile_definitions(${TARGET_NAME} PRIVATE JOB_ID=${ID})
#        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_MULTI_CONFIG)
#    endforeach()
#endif()
#endif()