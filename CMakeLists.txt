
project(rl_tools_core)
add_library(rl_tools_core INTERFACE)
set_property(TARGET rl_tools_core PROPERTY EXPORT_NAME Core)
target_compile_features(rl_tools_core INTERFACE cxx_std_17)
target_include_directories(
    rl_tools_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(RL_TOOLS_ENABLE_TARGETS)
    add_subdirectory(src)
endif()

if (RL_TOOLS_ENABLE_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()
