if(RL_TOOLS_ENABLE_HDF5 AND RL_TOOLS_ENABLE_TENSORBOARD)
    add_executable(
            nn_mnist_training
            training.cpp
    )
    target_link_libraries(
            nn_mnist_training
            PRIVATE
            rl_tools
            HighFive
            tensorboard_logger
    )
endif()
