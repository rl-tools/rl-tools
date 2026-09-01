find_package(GTest QUIET)
if(GTest_FOUND)
    message(VERBOSE "Found existing/system GTest ${GTest_VERSION} at ${GTest_DIR}")
    set(RL_TOOLS_ENABLE_GTEST ON)
    set(RL_TOOLS_SUMMARY_TESTS_REASON "system gtest ${GTest_VERSION}")
else()
    find_package(Git QUIET)
    if(GIT_FOUND)
        FetchContent_Declare(googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG   52eb8108c5bdec04579160ae17225d66034bd723
        )
        rl_tools_fetchcontent_makeavailable_quiet(googletest)
        set(RL_TOOLS_ENABLE_GTEST ON)
        set(RL_TOOLS_SUMMARY_TESTS_REASON "gtest via FetchContent")
    else()
        message(VERBOSE "Git not found - GTest disabled")
    endif()
endif()