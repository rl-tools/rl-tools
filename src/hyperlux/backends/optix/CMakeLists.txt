add_library(hyperlux_optix_overlay_accel STATIC ${CMAKE_CURRENT_SOURCE_DIR}/overlay_accel.cu)
target_link_libraries(hyperlux_optix_overlay_accel PRIVATE owl::owl)

foreach(RL_TOOLS_PTX_VARIANT base normals)
    if(RL_TOOLS_PTX_VARIANT STREQUAL "base")
        set(RL_TOOLS_PTX_SYMBOL device_ptx)
    else()
        set(RL_TOOLS_PTX_SYMBOL device_${RL_TOOLS_PTX_VARIANT}_ptx)
    endif()
    set(RL_TOOLS_PTX_TARGET hyperlux_${RL_TOOLS_PTX_SYMBOL})
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
    if(RL_TOOLS_PTX_VARIANT MATCHES "normals")
        target_compile_definitions(${RL_TOOLS_PTX_TARGET}_ptx PRIVATE RL_TOOLS_HYPERLUX_ENABLE_NORMALS_PROGRAMS=1)
    endif()
    if(NOT RL_TOOLS_PTX_VARIANT STREQUAL "base")
        set_target_properties(${RL_TOOLS_PTX_TARGET} ${RL_TOOLS_PTX_TARGET}_ptx generate_${RL_TOOLS_PTX_TARGET} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
    # every OptiX consumer links one of the device PTX targets (some bypass the backend interface
    # targets), so the overlay accel library rides along with them
    target_link_libraries(${RL_TOOLS_PTX_TARGET} INTERFACE hyperlux_optix_overlay_accel)
endforeach()

add_library(hyperlux_backend INTERFACE)
target_link_libraries(hyperlux_backend INTERFACE owl::owl hyperlux_device_ptx assimp::assimp)

add_library(hyperlux_backend_depth INTERFACE)
target_link_libraries(hyperlux_backend_depth INTERFACE hyperlux_backend)

add_library(hyperlux_backend_segmentation INTERFACE)
target_link_libraries(hyperlux_backend_segmentation INTERFACE hyperlux_backend)

add_library(hyperlux_backend_normals INTERFACE)
target_link_libraries(hyperlux_backend_normals INTERFACE hyperlux_backend hyperlux_device_normals_ptx)
