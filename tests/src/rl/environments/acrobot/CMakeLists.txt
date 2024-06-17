if(RL_TOOLS_ENABLE_HDF5)
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

if(RL_TOOLS_ENABLE_LIBWEBSOCKETS)
add_executable(rl_environments_acrobot_ui_client ui_client.cpp)

target_link_libraries(rl_environments_acrobot_ui_client
        PRIVATE
        rl_tools
        rl_tools_test
)
endif()
