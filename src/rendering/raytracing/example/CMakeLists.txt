add_executable(rendering_raytracing_example example.cpp)
target_link_libraries(rendering_raytracing_example
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
)

find_package(glfw3 QUIET)
find_package(OpenGL QUIET)
if(glfw3_FOUND AND OpenGL_FOUND)
    add_executable(rendering_raytracing_interactive interactive.cpp)
    target_link_libraries(rendering_raytracing_interactive
        PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
            glfw
            OpenGL::GL
    )
endif()
