set(RL_TOOLS_METAL_DEVICE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/device.metal)
set(RL_TOOLS_METAL_DEVICE_EMBED ${CMAKE_CURRENT_BINARY_DIR}/device_msl.cpp)
# resolved relative to this file so out-of-tree consumers (hypert JIT) get the right path
get_filename_component(RL_TOOLS_METAL_EMBED_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/scripts/embed_text.cmake ABSOLUTE)

add_custom_command(
    OUTPUT ${RL_TOOLS_METAL_DEVICE_EMBED}
    COMMAND ${CMAKE_COMMAND}
        -DINPUT_FILE=${RL_TOOLS_METAL_DEVICE_SOURCE}
        -DOUTPUT_FILE=${RL_TOOLS_METAL_DEVICE_EMBED}
        -DSYMBOL_NAME=rl_tools_hyperlux_metal_device_source
        -P ${RL_TOOLS_METAL_EMBED_SCRIPT}
    DEPENDS ${RL_TOOLS_METAL_DEVICE_SOURCE} ${RL_TOOLS_METAL_EMBED_SCRIPT}
    COMMENT "Embedding Metal device source"
)

add_library(hyperlux_metal STATIC ${RL_TOOLS_METAL_DEVICE_EMBED} metal_cpp_impl.cpp)
target_link_libraries(hyperlux_metal PUBLIC hyperlux_metal_cpp)
set_target_properties(hyperlux_metal PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

add_library(hyperlux_backend INTERFACE)
target_link_libraries(hyperlux_backend INTERFACE hyperlux_metal assimp::assimp)

add_library(hyperlux_backend_depth INTERFACE)
target_link_libraries(hyperlux_backend_depth INTERFACE hyperlux_metal assimp::assimp)

add_library(hyperlux_backend_segmentation INTERFACE)
target_link_libraries(hyperlux_backend_segmentation INTERFACE hyperlux_metal assimp::assimp)

add_library(hyperlux_backend_normals INTERFACE)
target_link_libraries(hyperlux_backend_normals INTERFACE hyperlux_metal assimp::assimp)
