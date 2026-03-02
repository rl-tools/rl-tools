enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    OWL
    GIT_REPOSITORY https://github.com/NVIDIA/OWL.git
    GIT_TAG        main
    EXCLUDE_FROM_ALL
)
rl_tools_fetchcontent_makeavailable_quiet(OWL)

find_package(assimp REQUIRED)

target_link_libraries(rl_tools_full INTERFACE owl::owl assimp::assimp)
target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
