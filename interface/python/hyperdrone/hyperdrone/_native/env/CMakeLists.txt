cmake_minimum_required(VERSION 3.24)
project(hyperdrone_env LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RLTOOLS_ROOT "" CACHE PATH "Path to the rl-tools source root")
set(HYPERDRONE_VARIANT "OPTIX" CACHE STRING "Raytracing backend (OPTIX|METAL|VULKAN|WEBGPU|GENERIC)")

if(NOT RLTOOLS_ROOT)
    message(FATAL_ERROR "RLTOOLS_ROOT must point to the rl-tools source root")
endif()

set(RL_TOOLS_DISABLE_TARGETS ON CACHE BOOL "" FORCE)
set(RL_TOOLS_DISABLE_TESTS ON CACHE BOOL "" FORCE)
set(RL_TOOLS_DISABLE_GIT_DIFF ON CACHE BOOL "" FORCE)
set(RL_TOOLS_RL_ENVIRONMENTS_DISABLE_MUJOCO ON CACHE BOOL "" FORCE)
set(RL_TOOLS_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
# an explicit backend selection hard-requires raytracing inside the rl_tools subtree
set(RL_TOOLS_RENDERING_RAYTRACING_BACKEND ${HYPERDRONE_VARIANT} CACHE STRING "" FORCE)
if(HYPERDRONE_VARIANT STREQUAL "OPTIX")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
if(HYPERDRONE_VARIANT STREQUAL "VULKAN")
    find_package(Vulkan REQUIRED COMPONENTS glslangValidator)
endif()

find_package(assimp REQUIRED)
add_subdirectory(${RLTOOLS_ROOT} rl_tools EXCLUDE_FROM_ALL)
set(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED ${HYPERDRONE_VARIANT})
add_subdirectory(${RLTOOLS_ROOT}/src/rendering/raytracing/backends rl_tools_rt_backends EXCLUDE_FROM_ALL)

include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        f1c79c02822848a9bed4315b12c8c8f3761e1296
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(stb)
add_library(hyperdrone_env_stb INTERFACE)
target_include_directories(hyperdrone_env_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

# the observation path only needs the RGB device program
function(hyperdrone_component_libraries DEFINES OUT_LIBRARIES)
    set(${OUT_LIBRARIES} rendering_raytracing_backend PARENT_SCOPE)
endfunction()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hyperdrone_jit.cmake)
hyperdrone_jit_add_configs(
    COMPONENT env
    SOURCES impl.cpp
    LINK RLtools::RLtools hyperdrone_env_stb
)
