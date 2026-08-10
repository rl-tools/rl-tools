add_executable(rendering_raytracing_example example.cpp)
target_link_libraries(rendering_raytracing_example
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_example_depth EXCLUDE_FROM_ALL example.cpp)
target_link_libraries(rendering_raytracing_example_depth
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend_depth
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_example_depth PRIVATE RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE=2)

add_executable(rendering_raytracing_example_rgbd EXCLUDE_FROM_ALL example.cpp)
target_link_libraries(rendering_raytracing_example_rgbd
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend_depth
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_example_rgbd PRIVATE RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE=1)

add_executable(rendering_raytracing_motion_blur motion_blur.cpp)
target_link_libraries(rendering_raytracing_motion_blur
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_antialiasing antialiasing.cpp)
target_link_libraries(rendering_raytracing_antialiasing
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_drone drone.cpp)
target_link_libraries(rendering_raytracing_drone
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_fixed_pose fixed_pose.cpp)
target_link_libraries(rendering_raytracing_fixed_pose
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
)

add_custom_target(rendering_raytracing_fixed_pose_png
    COMMAND $<TARGET_FILE:rendering_raytracing_fixed_pose>
        --output ${CMAKE_SOURCE_DIR}/rendering_raytracing_fixed_pose.png
    DEPENDS rendering_raytracing_fixed_pose
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

function(add_rendering_raytracing_fixed_pose_target target_name shading output_mode backend_target)
    add_executable(${target_name} EXCLUDE_FROM_ALL fixed_pose.cpp)
    target_link_libraries(${target_name}
        PRIVATE
            RLtools::RLtools
            ${backend_target}
            rendering_raytracing_stb
    )
    target_compile_definitions(${target_name}
        PRIVATE
            RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_SHADING=${shading}
            RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_OUTPUT_MODE=${output_mode}
    )
endfunction()

add_rendering_raytracing_fixed_pose_target(rendering_raytracing_fixed_pose_low 0 0 rendering_raytracing_backend)
add_rendering_raytracing_fixed_pose_target(rendering_raytracing_fixed_pose_medium 1 0 rendering_raytracing_backend)
add_rendering_raytracing_fixed_pose_target(rendering_raytracing_fixed_pose_high 2 0 rendering_raytracing_backend)
add_rendering_raytracing_fixed_pose_target(rendering_raytracing_fixed_pose_veryhigh 3 0 rendering_raytracing_backend)
add_rendering_raytracing_fixed_pose_target(rendering_raytracing_fixed_pose_depth 3 2 rendering_raytracing_backend_depth)
add_rendering_raytracing_fixed_pose_target(rendering_raytracing_fixed_pose_rgbd 3 1 rendering_raytracing_backend_depth)

add_executable(rendering_raytracing_pose_trace EXCLUDE_FROM_ALL render_pose_trace.cpp)
target_link_libraries(rendering_raytracing_pose_trace
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_backend_depth
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_pose_trace_motion_blur EXCLUDE_FROM_ALL render_pose_trace_motion_blur.cpp)
target_link_libraries(rendering_raytracing_pose_trace_motion_blur
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
)

find_package(glfw3 QUIET)
find_package(OpenGL QUIET)
if(glfw3_FOUND AND OpenGL_FOUND)
    if(APPLE)
        add_compile_definitions(GL_SILENCE_DEPRECATION)
    endif()
    add_executable(rendering_raytracing_interactive interactive.cpp)
    target_link_libraries(rendering_raytracing_interactive
        PRIVATE
            RLtools::RLtools
            rendering_raytracing_backend
            rendering_raytracing_stb
            glfw
            OpenGL::GL
    )

    add_executable(rendering_raytracing_interactive_depth EXCLUDE_FROM_ALL interactive.cpp)
    target_link_libraries(rendering_raytracing_interactive_depth
        PRIVATE
            RLtools::RLtools
            rendering_raytracing_backend_depth
            rendering_raytracing_stb
            glfw
            OpenGL::GL
    )
    target_compile_definitions(rendering_raytracing_interactive_depth PRIVATE RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE=2)

    add_executable(rendering_raytracing_interactive_rgbd EXCLUDE_FROM_ALL interactive.cpp)
    target_link_libraries(rendering_raytracing_interactive_rgbd
        PRIVATE
            RLtools::RLtools
            rendering_raytracing_backend_depth
            rendering_raytracing_stb
            glfw
            OpenGL::GL
    )
    target_compile_definitions(rendering_raytracing_interactive_rgbd PRIVATE RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE=1)

    add_executable(rendering_raytracing_interactive_segmentation EXCLUDE_FROM_ALL interactive.cpp)
    target_link_libraries(rendering_raytracing_interactive_segmentation
        PRIVATE
            RLtools::RLtools
            rendering_raytracing_backend_segmentation
            rendering_raytracing_stb
            glfw
            OpenGL::GL
    )
    target_compile_definitions(rendering_raytracing_interactive_segmentation PRIVATE RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE=3)
endif()
