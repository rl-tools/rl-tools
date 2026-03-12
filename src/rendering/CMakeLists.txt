add_subdirectory(procthor2glb)

if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    add_subdirectory(raytracing)
else()
    add_subdirectory(raytracing/yaw_prediction)
endif()
