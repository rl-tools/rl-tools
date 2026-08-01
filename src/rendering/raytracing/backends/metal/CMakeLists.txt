set(RL_TOOLS_METAL_DEVICE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/device.metal)
set(RL_TOOLS_METAL_DEVICE_EMBED ${CMAKE_CURRENT_BINARY_DIR}/device_msl.cpp)

add_custom_command(
    OUTPUT ${RL_TOOLS_METAL_DEVICE_EMBED}
    COMMAND ${CMAKE_COMMAND}
        -DINPUT_FILE=${RL_TOOLS_METAL_DEVICE_SOURCE}
        -DOUTPUT_FILE=${RL_TOOLS_METAL_DEVICE_EMBED}
        -DSYMBOL_NAME=rl_tools_rendering_raytracing_metal_device_source
        -P ${CMAKE_SOURCE_DIR}/cmake/scripts/embed_text.cmake
    DEPENDS ${RL_TOOLS_METAL_DEVICE_SOURCE} ${CMAKE_SOURCE_DIR}/cmake/scripts/embed_text.cmake
    COMMENT "Embedding Metal device source"
)

add_library(rendering_raytracing_metal STATIC ${RL_TOOLS_METAL_DEVICE_EMBED} metal_cpp_impl.cpp)
target_link_libraries(rendering_raytracing_metal PUBLIC rendering_raytracing_metal_cpp)
set_target_properties(rendering_raytracing_metal PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

add_library(rendering_raytracing_backend INTERFACE)
target_link_libraries(rendering_raytracing_backend INTERFACE rendering_raytracing_metal assimp::assimp)

add_library(rendering_raytracing_backend_depth INTERFACE)
target_link_libraries(rendering_raytracing_backend_depth INTERFACE rendering_raytracing_metal assimp::assimp)
