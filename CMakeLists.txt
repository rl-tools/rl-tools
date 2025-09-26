cmake_minimum_required(VERSION 3.14)
project(rl_tools_core)

add_library(rl_tools_core INTERFACE)
set_property(TARGET rl_tools_core PROPERTY EXPORT_NAME Core)
add_library(RLtools::Core ALIAS rl_tools_core)
add_library(RLtools::RLtools ALIAS rl_tools_core)
add_library(RLtools::Optimizations ALIAS rl_tools_core)
target_compile_features(rl_tools_core INTERFACE cxx_std_17)
target_include_directories(rl_tools_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# tiers of optional dependencies
# tier 0: none
# tier 1: backend (MKL, OpenBLAS, Accelerate)
# tier 2: json, hdf5
# tier 3: tensorboard

#if(RL_TOOLS_ENABLE_JSON)
#    find_package(nlohmann_json)
#    if(NOT nlohmann_json_FOUND)
#        message(FATAL_ERROR "nlohmann_json is required but not found. Using FetchContent.")
#        FetchContent_Declare(nlohmann_json
#                GIT_REPOSITORY https://github.com/nlohmann/json.git
#                GIT_TAG        v3.11.3
#        )
#        FetchContent_MakeAvailable(nlohmann_json)
#        set(nlohmann_json_DIR "${nlohmann_json_BINARY_DIR}")
#        find_package(nlohmann_json CONFIG REQUIRED)
#        target_link_libraries(rl_tools_core INTERFACE nlohmann_json::nlohmann_json)
#    endif()
#endif()
if(RL_TOOLS_ENABLE_TESTS)
    find_package(GTest REQUIRED)
    target_link_libraries(rl_tools_core INTERFACE GTest::gtest_main)
endif()

if(RL_TOOLS_ENABLE_TARGETS)
    add_subdirectory(src)
endif()

if (RL_TOOLS_ENABLE_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()
