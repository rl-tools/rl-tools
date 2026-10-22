add_executable(hyperlux_minimal minimal.cpp)
target_link_libraries(hyperlux_minimal
    PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
)

add_executable(hyperlux_minimal_overlay minimal_overlay.cpp)
target_link_libraries(hyperlux_minimal_overlay
    PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
)

add_executable(hyperlux_example example.cpp)
target_link_libraries(hyperlux_example
    PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
)

add_executable(hyperlux_example_depth EXCLUDE_FROM_ALL example.cpp)
target_link_libraries(hyperlux_example_depth
    PRIVATE
        RLtools::RLtools
        hyperlux_backend_depth
        hyperlux_stb
)
target_compile_definitions(hyperlux_example_depth PRIVATE RL_TOOLS_HYPERLUX_EXAMPLE_OUTPUT_MODE=2)

add_executable(hyperlux_example_rgbd EXCLUDE_FROM_ALL example.cpp)
target_link_libraries(hyperlux_example_rgbd
    PRIVATE
        RLtools::RLtools
        hyperlux_backend_depth
        hyperlux_stb
)
target_compile_definitions(hyperlux_example_rgbd PRIVATE RL_TOOLS_HYPERLUX_EXAMPLE_OUTPUT_MODE=1)

add_executable(hyperlux_motion_blur motion_blur.cpp)
target_link_libraries(hyperlux_motion_blur
    PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
)

add_executable(hyperlux_antialiasing antialiasing.cpp)
target_link_libraries(hyperlux_antialiasing
    PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
)

add_executable(hyperlux_flow flow.cpp)
target_link_libraries(hyperlux_flow
    PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
)

add_executable(hyperlux_drone drone.cpp)
target_link_libraries(hyperlux_drone
    PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
)

if(RL_TOOLS_HYPERLUX_BACKEND_RESOLVED STREQUAL "OPTIX")
    add_executable(hyperlux_drone_device drone_device.cu)
    target_link_libraries(hyperlux_drone_device
        PRIVATE
            RLtools::RLtools
            hyperlux_backend
            hyperlux_stb
    )
    set_target_properties(hyperlux_drone_device PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(hyperlux_fixed_pose fixed_pose.cpp)
target_link_libraries(hyperlux_fixed_pose
    PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
)

add_custom_target(hyperlux_fixed_pose_png
    COMMAND $<TARGET_FILE:hyperlux_fixed_pose>
        --output ${CMAKE_SOURCE_DIR}/hyperlux_fixed_pose.png
    DEPENDS hyperlux_fixed_pose
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

function(add_hyperlux_fixed_pose_target target_name shading output_mode backend_target)
    add_executable(${target_name} EXCLUDE_FROM_ALL fixed_pose.cpp)
    target_link_libraries(${target_name}
        PRIVATE
            RLtools::RLtools
            ${backend_target}
            hyperlux_stb
    )
    target_compile_definitions(${target_name}
        PRIVATE
            RL_TOOLS_HYPERLUX_FIXED_POSE_SHADING=${shading}
            RL_TOOLS_HYPERLUX_FIXED_POSE_OUTPUT_MODE=${output_mode}
    )
endfunction()

add_hyperlux_fixed_pose_target(hyperlux_fixed_pose_low 0 0 hyperlux_backend)
add_hyperlux_fixed_pose_target(hyperlux_fixed_pose_medium 1 0 hyperlux_backend)
add_hyperlux_fixed_pose_target(hyperlux_fixed_pose_high 2 0 hyperlux_backend)
add_hyperlux_fixed_pose_target(hyperlux_fixed_pose_veryhigh 3 0 hyperlux_backend)
add_hyperlux_fixed_pose_target(hyperlux_fixed_pose_depth 3 2 hyperlux_backend_depth)
add_hyperlux_fixed_pose_target(hyperlux_fixed_pose_rgbd 3 1 hyperlux_backend_depth)

add_executable(hyperlux_pose_trace EXCLUDE_FROM_ALL render_pose_trace.cpp)
target_link_libraries(hyperlux_pose_trace
    PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_backend_depth
        hyperlux_stb
)

add_executable(hyperlux_pose_trace_motion_blur EXCLUDE_FROM_ALL render_pose_trace_motion_blur.cpp)
target_link_libraries(hyperlux_pose_trace_motion_blur
    PRIVATE
        RLtools::RLtools
        hyperlux_backend
        hyperlux_stb
)

find_package(glfw3 QUIET)
find_package(OpenGL QUIET)
function(add_hyperlux_interactive_target target_name output_mode backend_target)
    add_executable(${target_name} ${ARGN} interactive.cpp)
    target_link_libraries(${target_name}
        PRIVATE
            RLtools::RLtools
            ${backend_target}
            hyperlux_stb
    )
    target_compile_definitions(${target_name} PRIVATE RL_TOOLS_HYPERLUX_INTERACTIVE_OUTPUT_MODE=${output_mode})
    if(glfw3_FOUND AND OpenGL_FOUND)
        target_link_libraries(${target_name} PRIVATE glfw OpenGL::GL)
        target_compile_definitions(${target_name} PRIVATE RL_TOOLS_HYPERLUX_INTERACTIVE_ENABLE_GLFW)
        if(APPLE)
            target_compile_definitions(${target_name} PRIVATE GL_SILENCE_DEPRECATION)
        endif()
    endif()
    if(RL_TOOLS_ENABLE_ZLIB AND UNIX)
        target_link_libraries(${target_name} PRIVATE ${RL_TOOLS_ZLIB_TARGET})
        target_compile_definitions(${target_name} PRIVATE RL_TOOLS_HYPERLUX_INTERACTIVE_ENABLE_KITTY)
    endif()
endfunction()

if((glfw3_FOUND AND OpenGL_FOUND) OR (RL_TOOLS_ENABLE_ZLIB AND UNIX))
    add_hyperlux_interactive_target(hyperlux_interactive 0 hyperlux_backend)
    add_hyperlux_interactive_target(hyperlux_interactive_depth 2 hyperlux_backend_depth EXCLUDE_FROM_ALL)
    add_hyperlux_interactive_target(hyperlux_interactive_rgbd 1 hyperlux_backend_depth EXCLUDE_FROM_ALL)
    add_hyperlux_interactive_target(hyperlux_interactive_segmentation 3 hyperlux_backend_segmentation EXCLUDE_FROM_ALL)
    add_hyperlux_interactive_target(hyperlux_interactive_flow 4 hyperlux_backend EXCLUDE_FROM_ALL)
endif()
