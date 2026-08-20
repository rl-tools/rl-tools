# WebGPU backend runtime: wgpu-native prebuilt release, pinned by hash (no Rust toolchain).
# The backend codes against the standard webgpu.h; wgpu-native is the reference runtime and
# Dawn/emscripten remain drop-in alternatives. Linux-only for now: add the macos/windows
# archives (and their link libraries) here when those platforms are wired up.
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"))
    message(FATAL_ERROR "RLtools WEBGPU backend: only linux-x86_64 is wired up so far (cmake/optional/webgpu.cmake)")
endif()

FetchContent_Declare(wgpu_native
    URL https://github.com/gfx-rs/wgpu-native/releases/download/v29.0.1.1/wgpu-linux-x86_64-release.zip
    URL_HASH SHA256=95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(wgpu_native)

add_library(rl_tools_wgpu_native STATIC IMPORTED GLOBAL)
set_target_properties(rl_tools_wgpu_native PROPERTIES
    IMPORTED_LOCATION ${wgpu_native_SOURCE_DIR}/lib/libwgpu_native.a
    INTERFACE_INCLUDE_DIRECTORIES ${wgpu_native_SOURCE_DIR}/include
    INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS};pthread;m"
)

include(cmake/optional/assimp_fixup.cmake)

target_link_libraries(rl_tools_full INTERFACE assimp::assimp rl_tools_wgpu_native)
target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_RENDERING_ENABLE_RAYTRACING RL_TOOLS_RENDERING_RAYTRACING_BACKEND_WEBGPU)
