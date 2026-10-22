function(rl_tools_configure_golden_generator TARGET_NAME BACKEND_NAME)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        RL_TOOLS_HYPERLUX_GOLDEN_BACKEND_NAME="${BACKEND_NAME}"
        RL_TOOLS_TEST_DATA_PATH=${RL_TOOLS_TEST_DATA_PATH}
    )
endfunction()
