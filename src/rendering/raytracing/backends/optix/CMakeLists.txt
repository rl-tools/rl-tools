embed_ptx(
    OUTPUT_TARGET rendering_raytracing_device_ptx
    PTX_LINK_LIBRARIES owl::owl
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/device.cu
)
