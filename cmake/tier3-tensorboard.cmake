if(NOT RL_TOOLS_DISABLE_TENSORBOARD)
    find_package(Protobuf QUIET)
    if(Protobuf_FOUND AND Protobuf_PROTOC_EXECUTABLE)
        FetchContent_Declare(tensorboard
                GIT_REPOSITORY https://github.com/rl-tools/tensorboard_logger.git
                GIT_TAG   9761224e07c34f672523808c3f688651e2822cbc
                GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(tensorboard)
        target_link_libraries(rl_tools_full INTERFACE tensorboard_logger)
        target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_TENSORBOARD)
    else()
        message(STATUS "Protobuf not found, TensorBoard disabled.")
    endif()
endif()
