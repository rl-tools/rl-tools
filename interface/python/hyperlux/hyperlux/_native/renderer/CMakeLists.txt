cmake_minimum_required(VERSION 3.24)
project(hyperlux_renderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RLTOOLS_ROOT "" CACHE PATH "Path to the rl-tools source root")
set(HYPERLUX_VARIANT "OPTIX" CACHE STRING "Raytracing backend (OPTIX|METAL|VULKAN|WEBGPU|GENERIC)")

if(NOT RLTOOLS_ROOT)
    message(FATAL_ERROR "RLTOOLS_ROOT must point to the rl-tools source root")
endif()

# rl_tools_full carries the backend compile definitions and dependencies (OWL/assimp/
# nlohmann_json); targets are disabled so the dead-pinned procthor2glb subtree is never
# configured. The backend device-program libraries are pulled in separately below.
set(RL_TOOLS_DISABLE_TARGETS ON CACHE BOOL "" FORCE)
set(RL_TOOLS_DISABLE_TESTS ON CACHE BOOL "" FORCE)
# the ExTrack git snapshot re-runs on every build (~10s of pure overhead per Renderer
# construction) and is useless for the JIT cache
set(RL_TOOLS_DISABLE_GIT_DIFF ON CACHE BOOL "" FORCE)
set(RL_TOOLS_RL_ENVIRONMENTS_DISABLE_MUJOCO ON CACHE BOOL "" FORCE)
set(RL_TOOLS_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
# an explicit backend selection hard-requires raytracing inside the rl_tools subtree
set(RL_TOOLS_HYPERLUX_BACKEND ${HYPERLUX_VARIANT} CACHE STRING "" FORCE)
# language enablement is directory-scoped too: the backends subtree calls embed_ptx and
# needs CUDA enabled from here, not just inside the rl_tools subtree
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
# the resolved-backend variable is directory-scoped inside the rl_tools subtree, so the
# backends dispatch needs it set again here
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
add_library(hyperlux_renderer_stb INTERFACE)
target_include_directories(hyperlux_renderer_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hyperlux_jit.cmake)
hyperlux_jit_find_nanobind()

# camera_orbit.h and friends: demo/benchmark tooling deliberately outside the library,
# consumed by the shim to keep Renderer.generate_cameras on the Python surface
add_library(hyperlux_renderer_tooling INTERFACE)
target_include_directories(hyperlux_renderer_tooling INTERFACE ${RLTOOLS_ROOT}/src)

nanobind_add_module(hyperlux_core scene_bindings.cpp)
target_include_directories(hyperlux_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(hyperlux_core PRIVATE RLtools::RLtools hyperlux_renderer_stb)
set_target_properties(hyperlux_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(HYPERLUX_VARIANT STREQUAL "OPTIX")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(hyperlux_core PRIVATE CUDA::cudart)
endif()

function(hyperlux_component_libraries DEFINES OUT_LIBRARIES)
    hyperlux_jit_parse_defines("${DEFINES}" CONFIG)
    if(CONFIG_HYPERLUX_CAMERA_BACKEND_REQUIREMENT STREQUAL "normals")
        set(${OUT_LIBRARIES} hyperlux_backend_normals PARENT_SCOPE)
    elseif(CONFIG_HYPERLUX_CAMERA_BACKEND_REQUIREMENT STREQUAL "segmentation")
        set(${OUT_LIBRARIES} hyperlux_backend_segmentation PARENT_SCOPE)
    elseif(CONFIG_HYPERLUX_CAMERA_BACKEND_REQUIREMENT STREQUAL "depth")
        set(${OUT_LIBRARIES} hyperlux_backend_depth PARENT_SCOPE)
    else()
        set(${OUT_LIBRARIES} hyperlux_backend PARENT_SCOPE)
    endif()
endfunction()

set(render_sources renderer.cpp)
set(render_libraries RLtools::RLtools hyperlux_renderer_stb hyperlux_renderer_tooling)
if(TARGET RLtools::HyperLuxSpy)
    list(APPEND render_libraries RLtools::HyperLuxSpy hyperlux_backend)
endif()
if(HYPERLUX_VARIANT STREQUAL "OPTIX")
    list(APPEND render_sources set_pose_device.cu)
endif()
hyperlux_jit_add_configs(
    COMPONENT renderer
    PYTHON_MODULE
    SOURCES ${render_sources}
    LINK ${render_libraries}
)
