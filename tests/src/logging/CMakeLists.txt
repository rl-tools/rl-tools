

add_executable(logging_tensorboard
    tensorboard.cpp
)
target_link_libraries(
        logging_tensorboard
        PRIVATE
        backprop_tools
        GTest::gtest_main
)

gtest_discover_tests(logging_tensorboard)