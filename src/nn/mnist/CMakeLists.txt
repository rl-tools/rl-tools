if(BACKPROP_TOOLS_ENABLE_HDF5 AND BACKPROP_TOOLS_ENABLE_TENSORBOARD)
    add_executable(
            nn_mnist_training
            training.cpp
    )
    target_link_libraries(
            nn_mnist_training
            PRIVATE
            backprop_tools
            HighFive
            tensorboard_logger
    )
endif()
