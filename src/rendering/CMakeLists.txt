if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    add_subdirectory(procthor2glb)
    add_subdirectory(raytracing)
    add_subdirectory(datasets) # the AI2ThorHab smoke tool needs the raytracing tree's assimp/stb wiring
endif()
