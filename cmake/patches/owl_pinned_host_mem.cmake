# OWL's PinnedHostMem frees cudaMallocHost memory with cudaFree (destructor and resize);
# the second renderer lifecycle in one process then dies with "invalid argument" in
# owl Buffer.cpp once launch params are non-empty. Idempotent string patch, applied as
# FetchContent PATCH_COMMAND (and re-applied by seeding code for offline-populated trees).
# Drop once fixed upstream: NVIDIA/OWL owl/DeviceMemory.h.
if(NOT DEFINED OWL_SOURCE_DIR)
    message(FATAL_ERROR "owl_pinned_host_mem.cmake requires -DOWL_SOURCE_DIR=<owl source tree>")
endif()
set(OWL_DEVICE_MEMORY_HEADER "${OWL_SOURCE_DIR}/owl/DeviceMemory.h")
if(NOT EXISTS "${OWL_DEVICE_MEMORY_HEADER}")
    message(FATAL_ERROR "owl_pinned_host_mem.cmake: ${OWL_DEVICE_MEMORY_HEADER} not found")
endif()
file(READ "${OWL_DEVICE_MEMORY_HEADER}" OWL_DEVICE_MEMORY_CONTENT)
string(REPLACE "if (ptr) cudaFree(ptr);" "if (ptr) cudaFreeHost(ptr);" OWL_DEVICE_MEMORY_PATCHED "${OWL_DEVICE_MEMORY_CONTENT}")
if(NOT OWL_DEVICE_MEMORY_PATCHED STREQUAL OWL_DEVICE_MEMORY_CONTENT)
    file(WRITE "${OWL_DEVICE_MEMORY_HEADER}" "${OWL_DEVICE_MEMORY_PATCHED}")
    message(STATUS "owl_pinned_host_mem: patched ${OWL_DEVICE_MEMORY_HEADER}")
endif()
