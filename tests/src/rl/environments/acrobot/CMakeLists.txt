if(RL_TOOLS_ENABLE_HDF5 AND RL_TOOLS_ENABLE_GTK)
add_executable(test_rl_environments_acrobot_comparison comparison.cpp)

target_link_libraries(test_rl_environments_acrobot_comparison
        PRIVATE
        rl_tools
        rl_tools_test
)


if(RL_TOOLS_ENABLE_GTK)
add_executable(test_rl_environments_acrobot_ui ui.cpp)

target_link_libraries(test_rl_environments_acrobot_ui
        PRIVATE
        rl_tools
        rl_tools_test
        )
endif()

endif()