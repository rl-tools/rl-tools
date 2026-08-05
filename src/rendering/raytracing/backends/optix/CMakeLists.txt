embed_ptx(
    OUTPUT_TARGET rendering_raytracing_device_ptx
    PTX_LINK_LIBRARIES owl::owl
    EMBEDDED_SYMBOL_NAMES device_ptx
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/device.cu
)

embed_ptx(
    OUTPUT_TARGET rendering_raytracing_device_depth_ptx
    PTX_LINK_LIBRARIES owl::owl
    EMBEDDED_SYMBOL_NAMES device_depth_ptx
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/device.cu
)

# -G (device-debug) seems to cause OptiX JIT to emit misaligned loads on 12-byte-strided
# vertex arrays (owl::vec3f), seemingly triggering cudaErrorMisalignedAddress (716).
# --dopt re-enables device optimization despite -G, avoiding the bad codegen.
target_compile_options(rendering_raytracing_device_ptx_ptx PRIVATE $<$<CONFIG:Debug>:--dopt=on>)
target_compile_definitions(rendering_raytracing_device_depth_ptx_ptx PRIVATE RL_TOOLS_RENDERING_RAYTRACING_ENABLE_DEPTH_PROGRAMS=1)
target_compile_options(rendering_raytracing_device_depth_ptx_ptx PRIVATE $<$<CONFIG:Debug>:--dopt=on>)
set_target_properties(rendering_raytracing_device_depth_ptx rendering_raytracing_device_depth_ptx_ptx generate_rendering_raytracing_device_depth_ptx PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_library(rendering_raytracing_backend INTERFACE)
target_link_libraries(rendering_raytracing_backend INTERFACE owl::owl rendering_raytracing_device_ptx assimp::assimp)

add_library(rendering_raytracing_backend_depth INTERFACE)
target_link_libraries(rendering_raytracing_backend_depth INTERFACE owl::owl rendering_raytracing_device_depth_ptx assimp::assimp)

embed_ptx(
    OUTPUT_TARGET rendering_raytracing_device_segmentation_ptx
    PTX_LINK_LIBRARIES owl::owl
    EMBEDDED_SYMBOL_NAMES device_segmentation_ptx
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/device.cu
)
target_compile_definitions(rendering_raytracing_device_segmentation_ptx_ptx PRIVATE RL_TOOLS_RENDERING_RAYTRACING_ENABLE_SEGMENTATION_PROGRAMS=1)
target_compile_options(rendering_raytracing_device_segmentation_ptx_ptx PRIVATE $<$<CONFIG:Debug>:--dopt=on>)
set_target_properties(rendering_raytracing_device_segmentation_ptx rendering_raytracing_device_segmentation_ptx_ptx generate_rendering_raytracing_device_segmentation_ptx PROPERTIES EXCLUDE_FROM_ALL TRUE)

embed_ptx(
    OUTPUT_TARGET rendering_raytracing_device_depth_segmentation_ptx
    PTX_LINK_LIBRARIES owl::owl
    EMBEDDED_SYMBOL_NAMES device_depth_segmentation_ptx
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/device.cu
)
target_compile_definitions(rendering_raytracing_device_depth_segmentation_ptx_ptx PRIVATE RL_TOOLS_RENDERING_RAYTRACING_ENABLE_DEPTH_PROGRAMS=1 RL_TOOLS_RENDERING_RAYTRACING_ENABLE_SEGMENTATION_PROGRAMS=1)
target_compile_options(rendering_raytracing_device_depth_segmentation_ptx_ptx PRIVATE $<$<CONFIG:Debug>:--dopt=on>)
set_target_properties(rendering_raytracing_device_depth_segmentation_ptx rendering_raytracing_device_depth_segmentation_ptx_ptx generate_rendering_raytracing_device_depth_segmentation_ptx PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_library(rendering_raytracing_backend_segmentation INTERFACE)
target_link_libraries(rendering_raytracing_backend_segmentation INTERFACE owl::owl rendering_raytracing_device_segmentation_ptx rendering_raytracing_device_depth_segmentation_ptx assimp::assimp)
