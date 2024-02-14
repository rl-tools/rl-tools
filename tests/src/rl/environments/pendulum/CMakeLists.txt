if(RL_TOOLS_ENABLE_HDF5)
    # RL Environment Pendulum test
    add_executable(
            test_rl_environments_pendulum
            pendulum.cpp
    )
    target_link_libraries(
            test_rl_environments_pendulum
            rl_tools
            rl_tools_test
    )
    gtest_discover_tests(test_rl_environments_pendulum)


    if(RL_TOOLS_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY)
        # RL Environment Pendulum Render test

        find_package(Qt5 COMPONENTS Widgets REQUIRED)
        add_executable(
                test_rl_environments_pendulum_render
                pendulum_render.cpp
        )
        target_link_libraries(
                test_rl_environments_pendulum_render
                rl_tools
                rl_tools_test
                Qt5::Widgets
        )
        gtest_discover_tests(test_rl_environments_pendulum_render)
    endif()
endif()
