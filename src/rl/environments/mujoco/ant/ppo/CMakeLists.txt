add_subdirectory(cpu)
if(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
