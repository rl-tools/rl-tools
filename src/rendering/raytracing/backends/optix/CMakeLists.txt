add_library(rendering_raytracing_optix_overlay_accel STATIC ${CMAKE_CURRENT_SOURCE_DIR}/overlay_accel.cu)
target_link_libraries(rendering_raytracing_optix_overlay_accel PRIVATE owl::owl)

# one PTX variant per output-flag combination: the host picks the symbol matching the SPEC's
# outputs (create_context_resources), so every combination a consumer may instantiate needs a
# compiled variant. Only the base variant participates in the default build.
foreach(RL_TOOLS_PTX_VARIANT base depth segmentation depth_segmentation normals depth_normals segmentation_normals depth_segmentation_normals)
    if(RL_TOOLS_PTX_VARIANT STREQUAL "base")
        set(RL_TOOLS_PTX_SYMBOL device_ptx)
    else()
        set(RL_TOOLS_PTX_SYMBOL device_${RL_TOOLS_PTX_VARIANT}_ptx)
    endif()
    set(RL_TOOLS_PTX_TARGET rendering_raytracing_${RL_TOOLS_PTX_SYMBOL})
    embed_ptx(
        OUTPUT_TARGET ${RL_TOOLS_PTX_TARGET}
        PTX_LINK_LIBRARIES owl::owl
        EMBEDDED_SYMBOL_NAMES ${RL_TOOLS_PTX_SYMBOL}
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/device.cu
    )
    # -G (device-debug) seems to cause OptiX JIT to emit misaligned loads on 12-byte-strided
    # vertex arrays (owl::vec3f), seemingly triggering cudaErrorMisalignedAddress (716).
    # --dopt re-enables device optimization despite -G, avoiding the bad codegen.
    target_compile_options(${RL_TOOLS_PTX_TARGET}_ptx PRIVATE $<$<CONFIG:Debug>:--dopt=on>)
    if(RL_TOOLS_PTX_VARIANT MATCHES "depth")
        target_compile_definitions(${RL_TOOLS_PTX_TARGET}_ptx PRIVATE RL_TOOLS_RENDERING_RAYTRACING_ENABLE_DEPTH_PROGRAMS=1)
    endif()
    if(RL_TOOLS_PTX_VARIANT MATCHES "segmentation")
        target_compile_definitions(${RL_TOOLS_PTX_TARGET}_ptx PRIVATE RL_TOOLS_RENDERING_RAYTRACING_ENABLE_SEGMENTATION_PROGRAMS=1)
    endif()
    if(RL_TOOLS_PTX_VARIANT MATCHES "normals")
        target_compile_definitions(${RL_TOOLS_PTX_TARGET}_ptx PRIVATE RL_TOOLS_RENDERING_RAYTRACING_ENABLE_NORMALS_PROGRAMS=1)
    endif()
    if(NOT RL_TOOLS_PTX_VARIANT STREQUAL "base")
        set_target_properties(${RL_TOOLS_PTX_TARGET} ${RL_TOOLS_PTX_TARGET}_ptx generate_${RL_TOOLS_PTX_TARGET} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
    # every OptiX consumer links one of the device PTX targets (some bypass the backend interface
    # targets), so the overlay accel library rides along with them
    target_link_libraries(${RL_TOOLS_PTX_TARGET} INTERFACE rendering_raytracing_optix_overlay_accel)
endforeach()

add_library(rendering_raytracing_backend INTERFACE)
target_link_libraries(rendering_raytracing_backend INTERFACE owl::owl rendering_raytracing_device_ptx assimp::assimp)

add_library(rendering_raytracing_backend_depth INTERFACE)
target_link_libraries(rendering_raytracing_backend_depth INTERFACE owl::owl rendering_raytracing_device_depth_ptx assimp::assimp)

add_library(rendering_raytracing_backend_segmentation INTERFACE)
target_link_libraries(rendering_raytracing_backend_segmentation INTERFACE owl::owl rendering_raytracing_device_segmentation_ptx rendering_raytracing_device_depth_segmentation_ptx assimp::assimp)

add_library(rendering_raytracing_backend_normals INTERFACE)
target_link_libraries(rendering_raytracing_backend_normals INTERFACE owl::owl rendering_raytracing_device_normals_ptx rendering_raytracing_device_depth_normals_ptx rendering_raytracing_device_segmentation_normals_ptx rendering_raytracing_device_depth_segmentation_normals_ptx assimp::assimp)
