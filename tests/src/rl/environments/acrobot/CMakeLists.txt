add_executable(test_rl_environments_acrobot_comparison comparison.cpp)

target_link_libraries(test_rl_environments_acrobot_comparison
        PRIVATE
        backprop_tools
        backprop_tools_tests
)


if(BACKPROP_TOOLS_ENABLE_GTK)
add_executable(test_rl_environments_acrobot_ui ui.cpp)

target_link_libraries(test_rl_environments_acrobot_ui
        PRIVATE
        backprop_tools
        backprop_tools_tests
        )
endif()
