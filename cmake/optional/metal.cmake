set(RL_TOOLS_METAL_CPP_PATH "" CACHE PATH "Path to a local metal-cpp checkout (skips the FetchContent download)")

add_library(rendering_raytracing_metal_cpp INTERFACE)
if(RL_TOOLS_METAL_CPP_PATH)
    target_include_directories(rendering_raytracing_metal_cpp SYSTEM INTERFACE ${RL_TOOLS_METAL_CPP_PATH})
else()
    include(FetchContent)
    FetchContent_Declare(
        metal_cpp
        URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15.2_iOS18.2.zip
        URL_HASH SHA256=3437e4abfbd3d45217f34772ef3502f31ba3358e5fb6ac9d0ca952a047bcfe25
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        EXCLUDE_FROM_ALL
    )
    rl_tools_fetchcontent_makeavailable_quiet(metal_cpp)
    target_include_directories(rendering_raytracing_metal_cpp SYSTEM INTERFACE ${metal_cpp_SOURCE_DIR})
endif()

find_library(RL_TOOLS_METAL_FRAMEWORK Metal REQUIRED)
find_library(RL_TOOLS_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(RL_TOOLS_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
target_link_libraries(rendering_raytracing_metal_cpp INTERFACE ${RL_TOOLS_METAL_FRAMEWORK} ${RL_TOOLS_FOUNDATION_FRAMEWORK} ${RL_TOOLS_QUARTZCORE_FRAMEWORK})

include(cmake/optional/assimp_fixup.cmake)

target_link_libraries(rl_tools_full INTERFACE assimp::assimp)
target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_RENDERING_ENABLE_RAYTRACING RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)
