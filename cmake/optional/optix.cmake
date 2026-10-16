enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# OWL's embed_ptx resolves bin2c with find_program, which searches PATH before the CMAKE_CUDA_COMPILER directory, so an nvcc from a different toolkit on PATH mixes two toolkits into the OptiX build
find_program(RL_TOOLS_OPTIX_NVCC_ON_PATH nvcc PATHS ENV PATH NO_DEFAULT_PATH NO_CACHE)
if(RL_TOOLS_OPTIX_NVCC_ON_PATH)
    file(REAL_PATH "${RL_TOOLS_OPTIX_NVCC_ON_PATH}" RL_TOOLS_OPTIX_NVCC_ON_PATH_REAL)
    file(REAL_PATH "${CMAKE_CUDA_COMPILER}" RL_TOOLS_OPTIX_CUDA_COMPILER_REAL)
    if(NOT RL_TOOLS_OPTIX_NVCC_ON_PATH_REAL STREQUAL RL_TOOLS_OPTIX_CUDA_COMPILER_REAL)
        message(WARNING "RLtools OptiX backend: the CUDA compiler ${CMAKE_CUDA_COMPILER} (CUDACXX/CMAKE_CUDA_COMPILER) is not the nvcc found on PATH (${RL_TOOLS_OPTIX_NVCC_ON_PATH}). OWL looks up bin2c on PATH before the compiler's directory, so the OptiX build mixes two CUDA toolkits; put the bin directory of ${CMAKE_CUDA_COMPILER} first on PATH or point CUDACXX at the PATH toolkit")
    endif()
endif()

include(FetchContent)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/../patches/owl_compile_options.cmake")
FetchContent_Declare(
    OWL
    GIT_REPOSITORY https://github.com/NVIDIA/OWL.git
    GIT_TAG        e5db326767ff759854477f892ba7a2936ad8406d
    PATCH_COMMAND "${CMAKE_COMMAND}" "-DOWL_SOURCE_DIR=<SOURCE_DIR>" -P "${CMAKE_CURRENT_LIST_DIR}/../patches/owl_compile_options.cmake"
    EXCLUDE_FROM_ALL
)
rl_tools_fetchcontent_makeavailable_quiet(OWL)
execute_process(COMMAND "${CMAKE_COMMAND}" "-DOWL_SOURCE_DIR=${owl_SOURCE_DIR}" -P "${CMAKE_CURRENT_LIST_DIR}/../patches/owl_compile_options.cmake" COMMAND_ERROR_IS_FATAL ANY)

target_link_libraries(rl_tools_full INTERFACE owl::owl assimp::assimp)
target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_RENDERING_ENABLE_RAYTRACING RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
