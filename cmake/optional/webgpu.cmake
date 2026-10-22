# WebGPU backend runtime: wgpu-native prebuilt release, pinned by hash (no Rust toolchain).
# The backend codes against the standard webgpu.h; wgpu-native is the reference runtime and
# Dawn/emscripten remain drop-in alternatives. Add further platform archives (and their link
# libraries) here when those platforms are wired up.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(RL_TOOLS_WGPU_NATIVE_ARCHIVE wgpu-linux-x86_64-release.zip)
    set(RL_TOOLS_WGPU_NATIVE_SHA256 95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a)
    set(RL_TOOLS_WGPU_NATIVE_LINK_LIBRARIES ${CMAKE_DL_LIBS} pthread m)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(RL_TOOLS_WGPU_NATIVE_ARCHIVE wgpu-macos-aarch64-release.zip)
    set(RL_TOOLS_WGPU_NATIVE_SHA256 a5797a37b1adf720bcd5dcffb291edbbd5b7b14be0a3874c28e6393a655a7a3e)
    set(RL_TOOLS_WGPU_NATIVE_LINK_LIBRARIES "-framework Metal" "-framework QuartzCore" "-framework CoreFoundation" "-framework Foundation" "-framework IOKit" "-framework IOSurface" objc)
else()
    message(FATAL_ERROR "RLtools WEBGPU backend: only linux-x86_64 and macos-arm64 are wired up so far (cmake/optional/webgpu.cmake)")
endif()

FetchContent_Declare(wgpu_native
    URL https://github.com/gfx-rs/wgpu-native/releases/download/v29.0.1.1/${RL_TOOLS_WGPU_NATIVE_ARCHIVE}
    URL_HASH SHA256=${RL_TOOLS_WGPU_NATIVE_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
rl_tools_fetchcontent_makeavailable_quiet(wgpu_native)

add_library(rl_tools_wgpu_native STATIC IMPORTED GLOBAL)
set_target_properties(rl_tools_wgpu_native PROPERTIES
    IMPORTED_LOCATION ${wgpu_native_SOURCE_DIR}/lib/libwgpu_native.a
    INTERFACE_INCLUDE_DIRECTORIES ${wgpu_native_SOURCE_DIR}/include
    INTERFACE_LINK_LIBRARIES "${RL_TOOLS_WGPU_NATIVE_LINK_LIBRARIES}"
)

target_link_libraries(rl_tools_full INTERFACE assimp::assimp rl_tools_wgpu_native)
target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_HYPERLUX RL_TOOLS_HYPERLUX_BACKEND_WEBGPU)
