FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(stb)

add_library(rendering_raytracing_stb INTERFACE)
target_include_directories(rendering_raytracing_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

add_subdirectory(backends)
add_subdirectory(benchmark)
add_subdirectory(example)
