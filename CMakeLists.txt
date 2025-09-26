cmake_minimum_required(VERSION 3.14)
project(rl_tools_core)

add_library(rl_tools_core INTERFACE)
add_library(RLtools::Core ALIAS rl_tools_core)
target_compile_features(rl_tools_core INTERFACE cxx_std_17)
target_include_directories(rl_tools_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(rl_tools_full INTERFACE)
add_library(RLtools::RLtools ALIAS rl_tools_full)
target_link_libraries(rl_tools_full INTERFACE rl_tools_core)

# tiers of optional dependencies
# tier 0: none
# tier 1: backend (MKL, OpenBLAS, Accelerate)
# tier 2: json, hdf5
# tier 3: tensorboard

if(NOT DEFINED FETCHCONTENT_BASE_DIR)
    set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.fc" CACHE PATH "Base dir for FetchContent downloads" FORCE)
else()
    set(FETCHCONTENT_BASE_DIR "${FETCHCONTENT_BASE_DIR}" CACHE PATH "Base dir for FetchContent downloads")
endif()
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
include(FetchContent)


####################################################################################################
############################################## TIER 1 ##############################################
####################################################################################################
if(RL_TOOLS_BACKEND_ENABLE_MKL)
    if(NOT MKL_ROOT)
        set(MKL_ROOT /opt/intel/oneapi/mkl/latest)
        message(WARNING "MKL_ROOT is not set. Using default: ${MKL_ROOT}")
    endif()
    find_package(MKL REQUIRED)
    target_link_libraries(rl_tools_full INTERFACE MKL::MKL)
    target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_BACKEND_ENABLE_MKL)
endif()

####################################################################################################
############################################## TIER 2 ##############################################
####################################################################################################
if(RL_TOOLS_ENABLE_JSON)
    find_package(nlohmann_json QUIET)
    if(NOT nlohmann_json_FOUND)
        message(INFO "nlohmann_json is required but not found. Using FetchContent.")
        FetchContent_Declare(nlohmann_json
                GIT_REPOSITORY https://github.com/nlohmann/json.git
                GIT_TAG        v3.11.3
        )
        FetchContent_MakeAvailable(nlohmann_json)
    endif()
    target_link_libraries(rl_tools_full INTERFACE nlohmann_json::nlohmann_json)
    target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_JSON)
endif()

if(RL_TOOLS_ENABLE_HDF5)
    find_package(HighFive QUIET)
    if(NOT HighFive_FOUND)
        message(INFO "HighFive is required but not found. Using FetchContent.")
        FetchContent_Declare(highfive
                GIT_REPOSITORY https://github.com/rl-tools/highfive.git
                GIT_TAG   be68bd0efcef338a016fba448d6444089fd196d5
        )
        FetchContent_MakeAvailable(highfive)
    endif()
    target_link_libraries(rl_tools_full INTERFACE HighFive)
    target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_HDF5)
endif()



if(RL_TOOLS_ENABLE_TESTS)
    find_package(GTest REQUIRED)
endif()

if(RL_TOOLS_ENABLE_TARGETS)
    add_subdirectory(src)
endif()

if (RL_TOOLS_ENABLE_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()
