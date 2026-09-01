find_package(Git)
if(GIT_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE RL_TOOLS_COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(VERBOSE "Commit hash: ${RL_TOOLS_COMMIT_HASH}")
    execute_process(
            COMMAND ${GIT_EXECUTABLE} show -s --format=%ct HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE RL_TOOLS_COMMIT_TIME
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(VERBOSE "Commit time: ${RL_TOOLS_COMMIT_TIME}")
endif()
if(RL_TOOLS_COMMIT_HASH)
    target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_COMMIT_HASH_EXTERNAL=${RL_TOOLS_COMMIT_HASH})
    target_compile_definitions(rl_tools_full INTERFACE METRA_COMMIT=${RL_TOOLS_COMMIT_HASH})
endif()
if(RL_TOOLS_COMMIT_TIME)
    target_compile_definitions(rl_tools_full INTERFACE METRA_COMMIT_TIME=${RL_TOOLS_COMMIT_TIME})
endif()
