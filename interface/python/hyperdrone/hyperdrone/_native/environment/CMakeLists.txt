cmake_minimum_required(VERSION 3.24)
project(hyperdrone_environment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RLTOOLS_ROOT "" CACHE PATH "Path to the rl-tools source root")
set(HYPERLUX_VARIANT "OPTIX" CACHE STRING "Raytracing backend (OPTIX|METAL|VULKAN|WEBGPU|GENERIC)")

if(NOT RLTOOLS_ROOT)
    message(FATAL_ERROR "RLTOOLS_ROOT must point to the rl-tools source root")
endif()

set(RL_TOOLS_DISABLE_TARGETS ON CACHE BOOL "" FORCE)
set(RL_TOOLS_DISABLE_TESTS ON CACHE BOOL "" FORCE)
set(RL_TOOLS_DISABLE_GIT_DIFF ON CACHE BOOL "" FORCE)
set(RL_TOOLS_RL_ENVIRONMENTS_DISABLE_MUJOCO ON CACHE BOOL "" FORCE)
set(RL_TOOLS_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
# an explicit backend selection hard-requires raytracing inside the rl_tools subtree
set(RL_TOOLS_HYPERLUX_BACKEND ${HYPERLUX_VARIANT} CACHE STRING "" FORCE)
if(HYPERLUX_VARIANT STREQUAL "OPTIX")
    # a normal variable before enable_language shadows the compiler default it caches, also in
    # trees configured before; CUDAARCHS overrides
    if(NOT DEFINED ENV{CUDAARCHS})
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
if(HYPERLUX_VARIANT STREQUAL "VULKAN")
    find_package(Vulkan REQUIRED COMPONENTS glslangValidator)
endif()

unset(FETCHCONTENT_SOURCE_DIR_ASSIMP CACHE)
add_subdirectory(${RLTOOLS_ROOT} rl_tools EXCLUDE_FROM_ALL)
set(RL_TOOLS_HYPERLUX_BACKEND_RESOLVED ${HYPERLUX_VARIANT})
add_subdirectory(${RLTOOLS_ROOT}/src/hyperlux/backends rl_tools_rt_backends EXCLUDE_FROM_ALL)

include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        f1c79c02822848a9bed4315b12c8c8f3761e1296
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(stb)
add_library(hyperdrone_environment_stb INTERFACE)
target_include_directories(hyperdrone_environment_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

include(${HYPERLUX_NATIVE_ROOT}/cmake/hyperlux_jit.cmake)
hyperlux_jit_find_nanobind()

function(hyperlux_component_libraries DEFINES OUT_LIBRARIES)
    hyperlux_jit_parse_defines("${DEFINES}" CONFIG)
    if(DEFINED CONFIG_HYPERDRONE_ENVIRONMENT_SPEC_HEADER)
        set(${OUT_LIBRARIES} hyperlux_backend hyperlux_backend_depth hyperlux_backend_segmentation hyperlux_backend_normals PARENT_SCOPE)
    elseif(CONFIG_HYPERLUX_CAMERA_BACKEND_REQUIREMENT STREQUAL "normals")
        set(${OUT_LIBRARIES} hyperlux_backend_normals PARENT_SCOPE)
    elseif(CONFIG_HYPERLUX_CAMERA_BACKEND_REQUIREMENT STREQUAL "segmentation")
        set(${OUT_LIBRARIES} hyperlux_backend_segmentation PARENT_SCOPE)
    elseif(CONFIG_HYPERLUX_CAMERA_BACKEND_REQUIREMENT STREQUAL "depth")
        set(${OUT_LIBRARIES} hyperlux_backend_depth PARENT_SCOPE)
    else()
        set(${OUT_LIBRARIES} hyperlux_backend PARENT_SCOPE)
    endif()
endfunction()

# under OptiX impl.cpp, which holds the environment behind the binding, is compiled by nvcc so
# the World can run on the GPU (Config.device); the nanobind translation unit stays C++
if(HYPERLUX_VARIANT STREQUAL "OPTIX")
    add_compile_definitions(HYPERDRONE_ENVIRONMENT_CUDA=1)
    set_source_files_properties(impl.cpp PROPERTIES LANGUAGE CUDA)
endif()

set(hyperdrone_environment_libraries RLtools::RLtools hyperdrone_environment_stb)
if(TARGET RLtools::HyperLuxSpy)
    list(APPEND hyperdrone_environment_libraries RLtools::HyperLuxSpy hyperlux_backend)
endif()
if(HYPERLUX_VARIANT STREQUAL "OPTIX")
    find_package(CUDAToolkit REQUIRED)
    list(APPEND hyperdrone_environment_libraries CUDA::cudart)
endif()

hyperlux_jit_add_configs(
    COMPONENT environment
    PYTHON_MODULE
    SOURCES environment.cpp impl.cpp
    LINK ${hyperdrone_environment_libraries}
)
