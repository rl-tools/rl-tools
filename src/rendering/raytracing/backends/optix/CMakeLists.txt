embed_ptx(
    OUTPUT_TARGET rendering_raytracing_device_ptx
    PTX_LINK_LIBRARIES owl::owl
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/device.cu
)
# -G (device-debug) seems to cause OptiX JIT to emit misaligned loads on 12-byte-strided
# vertex arrays (owl::vec3f), seemingly triggering cudaErrorMisalignedAddress (716).
# --dopt re-enables device optimization despite -G, avoiding the bad codegen.
target_compile_options(rendering_raytracing_device_ptx_ptx PRIVATE $<$<CONFIG:Debug>:--dopt=on>)
