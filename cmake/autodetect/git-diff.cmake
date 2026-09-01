# ==============================================================================
# Git snapshot tracking for ExTrack
# ==============================================================================

if(RL_TOOLS_DISABLE_GIT_DIFF)
    return()
endif()

if(NOT GIT_FOUND)
    message(STATUS "Git not found - git snapshot tracking disabled")
    return()
endif()

function(rl_tools_git_toplevel OUT_IS_REPO OUT_TOPLEVEL REPO_DIR)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --show-toplevel
        WORKING_DIRECTORY "${REPO_DIR}"
        RESULT_VARIABLE GIT_TOPLEVEL_RESULT
        OUTPUT_VARIABLE GIT_TOPLEVEL_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(GIT_TOPLEVEL_RESULT EQUAL 0)
        file(REAL_PATH "${GIT_TOPLEVEL_OUTPUT}" GIT_TOPLEVEL_NORMALIZED)
        set(${OUT_IS_REPO} TRUE PARENT_SCOPE)
        set(${OUT_TOPLEVEL} "${GIT_TOPLEVEL_NORMALIZED}" PARENT_SCOPE)
    else()
        set(${OUT_IS_REPO} FALSE PARENT_SCOPE)
        set(${OUT_TOPLEVEL} "" PARENT_SCOPE)
    endif()
endfunction()

file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" RL_TOOLS_ROOT_DIR)
file(REAL_PATH "${CMAKE_SOURCE_DIR}" CMAKE_SOURCE_DIR_NORMALIZED)

if(NOT CMAKE_SOURCE_DIR_NORMALIZED STREQUAL RL_TOOLS_ROOT_DIR)
    set(RL_TOOLS_PARENT_DIR "${CMAKE_SOURCE_DIR_NORMALIZED}")
    set(RL_TOOLS_HAS_PARENT TRUE)
else()
    set(RL_TOOLS_PARENT_DIR "")
    set(RL_TOOLS_HAS_PARENT FALSE)
endif()

rl_tools_git_toplevel(RL_TOOLS_IS_INSIDE_GIT_REPO RL_TOOLS_GIT_TOPLEVEL "${RL_TOOLS_ROOT_DIR}")
if(RL_TOOLS_IS_INSIDE_GIT_REPO AND RL_TOOLS_GIT_TOPLEVEL STREQUAL RL_TOOLS_ROOT_DIR)
    set(RL_TOOLS_IS_GIT_REPO TRUE)
else()
    set(RL_TOOLS_IS_GIT_REPO FALSE)
endif()

set(RL_TOOLS_PARENT_IS_GIT_REPO FALSE)
if(RL_TOOLS_HAS_PARENT)
    rl_tools_git_toplevel(RL_TOOLS_PARENT_IS_INSIDE_GIT_REPO RL_TOOLS_PARENT_GIT_TOPLEVEL "${RL_TOOLS_PARENT_DIR}")
    if(RL_TOOLS_PARENT_IS_INSIDE_GIT_REPO AND RL_TOOLS_PARENT_GIT_TOPLEVEL STREQUAL RL_TOOLS_PARENT_DIR)
        set(RL_TOOLS_PARENT_IS_GIT_REPO TRUE)
    endif()
endif()

if(NOT RL_TOOLS_IS_GIT_REPO AND NOT RL_TOOLS_PARENT_IS_GIT_REPO)
    message(STATUS "No git repositories detected - git snapshot tracking disabled")
    return()
endif()

message(STATUS "Git snapshot tracking enabled:")
message(STATUS "  Library: ${RL_TOOLS_ROOT_DIR} [git: ${RL_TOOLS_IS_GIT_REPO}]")
if(RL_TOOLS_HAS_PARENT)
    message(STATUS "  Project: ${RL_TOOLS_PARENT_DIR} [git: ${RL_TOOLS_PARENT_IS_GIT_REPO}]")
endif()

set(GIT_SNAPSHOT_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_snapshot.cpp")
set(GIT_SNAPSHOT_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/generate-git-snapshot.cmake")

set(GIT_SNAPSHOT_PARENT_ARGS "")
if(RL_TOOLS_HAS_PARENT)
    set(GIT_SNAPSHOT_PARENT_ARGS -DPARENT_ROOT=${RL_TOOLS_PARENT_DIR})
endif()

add_custom_target(rl_tools_git_snapshot_update
    COMMAND ${CMAKE_COMMAND}
        -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
        -DOUTPUT_FILE=${GIT_SNAPSHOT_OUTPUT}
        -DRL_TOOLS_ROOT=${RL_TOOLS_ROOT_DIR}
        -DRL_TOOLS_IS_REPO=${RL_TOOLS_IS_GIT_REPO}
        -DPARENT_IS_REPO=${RL_TOOLS_PARENT_IS_GIT_REPO}
        ${GIT_SNAPSHOT_PARENT_ARGS}
        -P "${GIT_SNAPSHOT_GENERATOR}"
    BYPRODUCTS ${GIT_SNAPSHOT_OUTPUT}
    COMMENT "Updating git snapshot for ExTrack"
    VERBATIM
)

set_source_files_properties(${GIT_SNAPSHOT_OUTPUT} PROPERTIES GENERATED TRUE)
add_library(rl_tools_git_snapshot STATIC ${GIT_SNAPSHOT_OUTPUT})
add_dependencies(rl_tools_git_snapshot rl_tools_git_snapshot_update)
target_compile_features(rl_tools_git_snapshot PRIVATE cxx_std_11)
target_compile_definitions(rl_tools_git_snapshot PUBLIC RL_TOOLS_EXTRACK_GIT_DIFF)

if(TARGET rl_tools_full)
    target_link_libraries(rl_tools_full INTERFACE rl_tools_git_snapshot)
endif()
