if(NOT RL_TOOLS_DISABLE_TAR)
    add_executable(test_rl_environments_hyperdrone_raptor raptor.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_raptor PRIVATE RLtools::RLtools GTest::gtest_main)
    gtest_discover_tests(test_rl_environments_hyperdrone_raptor)
endif()

add_executable(test_rl_environments_hyperdrone_on_policy_gradient on_policy_gradient.cpp)
target_link_libraries(test_rl_environments_hyperdrone_on_policy_gradient PRIVATE RLtools::RLtools GTest::gtest_main)
gtest_discover_tests(test_rl_environments_hyperdrone_on_policy_gradient)

if(TARGET RLtools::HyperDroneSpy)
    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)
    add_executable(test_rl_environments_hyperdrone_spy_ui spy_ui.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_spy_ui PRIVATE RLtools::RLtools RLtools::Test RLtools::HyperDroneSpy glfw OpenGL::GL)
    find_path(RL_TOOLS_SPY_TEST_IMGUI_INCLUDE_DIR imgui.h HINTS "${RL_TOOLS_IMGUI_SOURCE_DIR}" PATH_SUFFIXES imgui)
    target_include_directories(test_rl_environments_hyperdrone_spy_ui SYSTEM PRIVATE "${RL_TOOLS_SPY_TEST_IMGUI_INCLUDE_DIR}")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_package(X11 QUIET)
        if(TARGET X11::X11)
            target_link_libraries(test_rl_environments_hyperdrone_spy_ui PRIVATE X11::X11)
            target_compile_definitions(test_rl_environments_hyperdrone_spy_ui PRIVATE RL_TOOLS_SPY_TEST_X11)
        endif()
    endif()
    gtest_discover_tests(test_rl_environments_hyperdrone_spy_ui PROPERTIES LABELS "hyperdrone-spy-ui" TIMEOUT 20)
endif()

if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    add_executable(test_rl_environments_hyperdrone_spy_capture spy_capture.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_spy_capture PRIVATE RLtools::RLtools RLtools::Test rendering_raytracing_backend)
    gtest_discover_tests(test_rl_environments_hyperdrone_spy_capture)
    add_executable(test_rl_environments_hyperdrone_spy_spectator spy_spectator.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_spy_spectator PRIVATE RLtools::RLtools RLtools::Test rendering_raytracing_backend)
    gtest_discover_tests(test_rl_environments_hyperdrone_spy_spectator PROPERTIES TIMEOUT 90)
    add_executable(test_rl_environments_hyperdrone_spy_lifecycle spy_lifecycle.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_spy_lifecycle PRIVATE RLtools::RLtools RLtools::Test rendering_raytracing_backend)
    gtest_discover_tests(test_rl_environments_hyperdrone_spy_lifecycle PROPERTIES TIMEOUT 90)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA AND TARGET owl::owl)
        add_executable(test_rl_environments_hyperdrone_spy_capture_cuda spy_capture_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_spy_capture_cuda PRIVATE RLtools::RLtools RLtools::Test owl::owl rendering_raytracing_device_ptx assimp::assimp rendering_raytracing_stb)
        set_target_properties(test_rl_environments_hyperdrone_spy_capture_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        gtest_discover_tests(test_rl_environments_hyperdrone_spy_capture_cuda)
    endif()
endif()

if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING AND RL_TOOLS_TEST_DATA_PATH)
    add_executable(test_rl_environments_hyperdrone_rig rig.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_rig
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_rig)
    add_executable(test_rl_environments_hyperdrone_world world.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_world
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_world)
    add_executable(test_rl_environments_hyperdrone_annotations annotations.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_annotations
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_annotations)
    add_executable(test_rl_environments_hyperdrone_target_frame target_frame.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_target_frame
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_target_frame)
    add_executable(test_rl_environments_hyperdrone_moving_gate moving_gate.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_moving_gate
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_backend_depth
            rendering_raytracing_backend_segmentation
            rendering_raytracing_backend_normals
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_moving_gate)
    add_executable(test_rl_environments_hyperdrone_visual_inertial_localization visual_inertial_localization.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_visual_inertial_localization
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_visual_inertial_localization)
    add_executable(test_rl_environments_hyperdrone_segmentation_determinism segmentation_determinism.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_segmentation_determinism
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_backend_segmentation
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_segmentation_determinism)
    add_executable(test_rl_environments_hyperdrone_self_visible self_visible.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_self_visible
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_backend_segmentation
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_self_visible)
    add_executable(test_rl_environments_hyperdrone_multi_agent multi_agent.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_multi_agent
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_backend_segmentation
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_multi_agent)
    add_executable(test_rl_environments_hyperdrone_presets presets.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_presets
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_presets)
    add_executable(test_rl_environments_hyperdrone_episodes episodes.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_episodes
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_episodes)
    add_executable(test_rl_environments_hyperdrone_on_policy_runner_batched on_policy_runner_batched.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_on_policy_runner_batched
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_on_policy_runner_batched)
    # golden-rollout generator for the Python binding (run deliberately from the repo root)
    set(HYPERDRONE_PYTHON_ENV_NATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../interface/python/hyperdrone/hyperdrone/_native/env)
    add_executable(test_rl_environments_hyperdrone_python_golden python_golden.cpp ${HYPERDRONE_PYTHON_ENV_NATIVE}/impl.cpp)
    target_compile_definitions(test_rl_environments_hyperdrone_python_golden PRIVATE
        HYPERDRONE_ENV_NUM_ENVIRONMENTS=1
        HYPERDRONE_ENV_INSTANCES=2
        HYPERDRONE_ENV_CAM_WIDTH=16
        HYPERDRONE_ENV_CAM_HEIGHT=16
        HYPERDRONE_ENV_SHADING=0
        HYPERDRONE_ENV_MB_SAMPLES=1
        HYPERDRONE_ENV_PRESET=0
        HYPERDRONE_ENV_TASK=0
        HYPERDRONE_ENV_N_AGENTS=1
        HYPERDRONE_ENV_N_CAMERAS_PER_AGENT=1
        HYPERDRONE_ENV_DEVICE_CUDA=0
        HYPERDRONE_ENV_OUTPUT_RGB=1
        HYPERDRONE_ENV_OUTPUT_DEPTH=0
        HYPERDRONE_ENV_OUTPUT_SEGMENTATION=0
        HYPERDRONE_ENV_SEMANTIC_SEGMENTATION=0
        HYPERDRONE_ENV_DATASET=0
        HYPERDRONE_ENV_OUTPUT_NORMALS=0
        HYPERDRONE_ENV_OUTPUT_FLOW=0
        HYPERDRONE_MODULE_NAME=hyperdrone_env_golden
    )
    target_link_libraries(test_rl_environments_hyperdrone_python_golden
        PRIVATE
            RLtools::RLtools
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    add_executable(test_rl_environments_hyperdrone_user_task_header user_task_header.cpp ${HYPERDRONE_PYTHON_ENV_NATIVE}/impl.cpp)
    get_target_property(HYPERDRONE_USER_TASK_DEFINES test_rl_environments_hyperdrone_python_golden COMPILE_DEFINITIONS)
    list(FILTER HYPERDRONE_USER_TASK_DEFINES EXCLUDE REGEX "^HYPERDRONE_MODULE_NAME=")
    target_compile_definitions(test_rl_environments_hyperdrone_user_task_header PRIVATE ${HYPERDRONE_USER_TASK_DEFINES} HYPERDRONE_MODULE_NAME=hyperdrone_user_task HYPERDRONE_ENV_SPEC_HEADER=${PROJECT_SOURCE_DIR}/interface/python/hyperdrone/tests/env/user_task_header.h)
    target_link_libraries(test_rl_environments_hyperdrone_user_task_header PRIVATE RLtools::RLtools RLtools::Test rendering_raytracing_backend rendering_raytracing_stb)
    gtest_discover_tests(test_rl_environments_hyperdrone_user_task_header)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA AND TARGET owl::owl)
        add_executable(test_rl_environments_hyperdrone_rig_cuda rig_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_rig_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_rig_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_rig_cuda)
        add_executable(test_rl_environments_hyperdrone_world_cuda world_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_world_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_world_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_world_cuda)
        add_executable(test_rl_environments_hyperdrone_target_frame_cuda target_frame_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_target_frame_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_target_frame_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_target_frame_cuda)
        add_executable(test_rl_environments_hyperdrone_multi_agent_cuda multi_agent_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_multi_agent_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                rendering_raytracing_device_segmentation_ptx
                rendering_raytracing_device_depth_segmentation_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_multi_agent_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_multi_agent_cuda)
        add_executable(test_rl_environments_hyperdrone_self_visible_cuda self_visible_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_self_visible_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                rendering_raytracing_device_segmentation_ptx
                rendering_raytracing_device_depth_segmentation_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_self_visible_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_self_visible_cuda)
        add_executable(test_rl_environments_hyperdrone_moving_gate_cuda moving_gate_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_moving_gate_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                rendering_raytracing_device_depth_segmentation_normals_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_moving_gate_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_moving_gate_cuda)
        add_executable(test_rl_environments_hyperdrone_episodes_cuda episodes_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_episodes_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_episodes_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_episodes_cuda)
        add_executable(test_rl_environments_hyperdrone_benchmark_cuda benchmark_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_benchmark_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_benchmark_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_benchmark_cuda PROPERTIES TIMEOUT 300)
    endif()
endif()

if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    add_executable(test_rl_environments_hyperdrone_drones drones.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_drones
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_backend_segmentation
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_drones)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA AND TARGET owl::owl)
        add_executable(test_rl_environments_hyperdrone_drones_cuda drones_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_drones_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                rendering_raytracing_device_segmentation_ptx
                rendering_raytracing_device_depth_segmentation_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_drones_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_drones_cuda)
    endif()
endif()
