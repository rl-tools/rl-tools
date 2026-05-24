add_executable(rendering_raytracing_example example.cpp)
target_link_libraries(rendering_raytracing_example
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_example_depth EXCLUDE_FROM_ALL example.cpp)
target_link_libraries(rendering_raytracing_example_depth
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_depth_ptx
        assimp::assimp
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_example_depth PRIVATE RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE=2)

add_executable(rendering_raytracing_example_rgbd EXCLUDE_FROM_ALL example.cpp)
target_link_libraries(rendering_raytracing_example_rgbd
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_depth_ptx
        assimp::assimp
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_example_rgbd PRIVATE RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE=1)

add_executable(rendering_raytracing_motion_blur motion_blur.cpp)
target_link_libraries(rendering_raytracing_motion_blur
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_antialiasing antialiasing.cpp)
target_link_libraries(rendering_raytracing_antialiasing
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_pose_trace EXCLUDE_FROM_ALL render_pose_trace.cpp)
target_link_libraries(rendering_raytracing_pose_trace
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        rendering_raytracing_device_depth_ptx
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

    add_executable(rendering_raytracing_interactive_depth EXCLUDE_FROM_ALL interactive.cpp)
    target_link_libraries(rendering_raytracing_interactive_depth
        PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_depth_ptx
            assimp::assimp
            rendering_raytracing_stb
            glfw
            OpenGL::GL
    )
    target_compile_definitions(rendering_raytracing_interactive_depth PRIVATE RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE=2)

    add_executable(rendering_raytracing_interactive_rgbd EXCLUDE_FROM_ALL interactive.cpp)
    target_link_libraries(rendering_raytracing_interactive_rgbd
        PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_depth_ptx
            assimp::assimp
            rendering_raytracing_stb
            glfw
            OpenGL::GL
    )
    target_compile_definitions(rendering_raytracing_interactive_rgbd PRIVATE RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE=1)
endif()
