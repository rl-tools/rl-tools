if(NOT RL_TOOLS_DISABLE_CLI11)
    set(CLI11_PRECOMPILED ON)
    find_package(CLI11 QUIET)
    if(CLI11_FOUND)
        message(VERBOSE "Found existing/system CLI11 ${CLI11_VERSION} at ${CLI11_DIR}")
        target_link_libraries(rl_tools_full INTERFACE CLI11::CLI11)
        target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_CLI11)
        set(RL_TOOLS_ENABLE_CLI11 ON)
        set(RL_TOOLS_SUMMARY_CLI11_REASON "system ${CLI11_VERSION}")
    elseif(GIT_FOUND)
        FetchContent_Declare(CLI11
                GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
                GIT_TAG        v2.3.2
                GIT_SHALLOW    TRUE
        )
        rl_tools_fetchcontent_makeavailable_quiet(CLI11)
        target_link_libraries(rl_tools_full INTERFACE CLI11::CLI11)
        target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_CLI11)
        set(RL_TOOLS_ENABLE_CLI11 ON)
        set(RL_TOOLS_SUMMARY_CLI11_REASON "FetchContent")
    else()
        set(RL_TOOLS_SUMMARY_CLI11_REASON "not found and no git")
    endif()
endif()
