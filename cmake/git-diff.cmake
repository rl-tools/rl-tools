option(RL_TOOLS_ENABLE_GIT_DIFF "Enable embedding git diff into ExTrack runs" OFF)

if(RL_TOOLS_ENABLE_GIT_DIFF)
    file(GLOB_RECURSE GIT_DIFF_PROJECT_SOURCES
            CONFIGURE_DEPENDS
            "${PROJECT_SOURCE_DIR}/src/*"
            "${PROJECT_SOURCE_DIR}/include/*"
    )
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            DEPENDS ${GIT_DIFF_PROJECT_SOURCES}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack
            COMMAND ${CMAKE_COMMAND} -E echo "namespace rl_tools::utils::extrack::git{" > ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo_append "    extern const char* const commit = R\"git_diff(" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND git rev-parse HEAD >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo ")git_diff\";" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo "    extern const char* const diff = R\"git_diff(" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND git diff >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo ")git_diff\";" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo "    extern const char* const diff_color = R\"git_diff(" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND git diff --color=always >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo ")git_diff\";" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo "    extern const char* const word_diff = R\"git_diff(" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND git diff --word-diff >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo ")git_diff\";" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo "    extern const char* const word_diff_color = R\"git_diff(" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND git diff --word-diff --color=always >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo ")git_diff\";" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo "    extern const char* const diff_staged = R\"git_diff(" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND git diff --cached >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo ")git_diff\";" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo "    extern const char* const diff_staged_color = R\"git_diff(" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND git diff --cached --color=always >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo ")git_diff\";" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo "    extern const char* const word_diff_staged = R\"git_diff(" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND git diff --cached --word-diff >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo ")git_diff\";" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo "    extern const char* const word_diff_staged_color = R\"git_diff(" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND git diff --cached --word-diff --color=always >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo ")git_diff\";" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            COMMAND ${CMAKE_COMMAND} -E echo "}" >> ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Generating git diff source"
            VERBATIM
    )

    add_library(git_diff STATIC ${CMAKE_CURRENT_BINARY_DIR}/rl_tools/extrack/git_diff.cpp)
    target_compile_definitions(git_diff PUBLIC RL_TOOLS_EXTRACK_GIT_DIFF)
    target_link_libraries(rl_tools_full INTERFACE git_diff)
endif()
