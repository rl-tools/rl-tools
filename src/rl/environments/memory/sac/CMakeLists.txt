add_executable(rl_environments_memory_sac memory.cpp)
target_link_libraries(rl_environments_memory_sac PRIVATE rl_tools)
target_compile_options(rl_environments_memory_sac PRIVATE -ftemplate-backtrace-limit=0)
if(RL_TOOLS_ENABLE_TRACY)
    target_link_libraries(rl_environments_memory_sac PRIVATE Tracy::TracyClient)
    target_include_directories(rl_environments_memory_sac PRIVATE ${PROJECT_SOURCE_DIR}/external/tracy/public/tracy)
    target_compile_definitions(rl_environments_memory_sac PRIVATE RL_TOOLS_ENABLE_TRACY)
endif()

