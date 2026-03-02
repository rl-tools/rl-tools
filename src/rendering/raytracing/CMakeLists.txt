FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(stb)

add_library(rendering_raytracing_stb INTERFACE)
target_include_directories(rendering_raytracing_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

embed_ptx(
    OUTPUT_TARGET rendering_raytracing_device_ptx
    PTX_LINK_LIBRARIES owl::owl
    SOURCES ${CMAKE_SOURCE_DIR}/include/rl_tools/rendering/raytracing/backends/optix/device.cu
)

add_executable(rendering_raytracing_benchmark benchmark.cpp)
target_link_libraries(rendering_raytracing_benchmark
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_example example.cpp)
target_link_libraries(rendering_raytracing_example
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
)
