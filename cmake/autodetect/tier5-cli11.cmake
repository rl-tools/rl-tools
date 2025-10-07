if(NOT RL_TOOLS_DISABLE_CLI11)
    set(CLI11_PRECOMPILED ON)
    find_package(CLI11 QUIET)
    if(CLI11_FOUND)
        message(STATUS "Found existing/system CLI11 ${CLI11_VERSION} at ${CLI11_DIR}")
    else()
        FetchContent_Declare(CLI11
                GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
                GIT_TAG   291c58789c031208f08f4f261a858b5b7083e8e2
        )
        FetchContent_MakeAvailable(CLI11)
    endif()
    set(RL_TOOLS_ENABLE_CLI11 ON)
    target_link_libraries(rl_tools_full INTERFACE CLI11::CLI11)
    target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_CLI11)
endif()
