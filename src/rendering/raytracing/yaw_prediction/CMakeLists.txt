
if(NOT TARGET rendering_raytracing_stb)
    include(FetchContent)
    FetchContent_Declare(
        yaw_prediction_stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG        f1c79c02822848a9bed4315b12c8c8f3761e1296
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(yaw_prediction_stb)
    add_library(rendering_raytracing_stb INTERFACE)
    target_include_directories(rendering_raytracing_stb SYSTEM INTERFACE ${yaw_prediction_stb_SOURCE_DIR})
endif()

add_executable(rendering_raytracing_yaw_prediction_real_world_eval yaw_prediction_real_world_eval.cpp)
set_target_properties(rendering_raytracing_yaw_prediction_real_world_eval PROPERTIES CXX_STANDARD 17)
target_include_directories(rendering_raytracing_yaw_prediction_real_world_eval PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rendering_raytracing_yaw_prediction_real_world_eval
    PRIVATE
        nlohmann_json::nlohmann_json
        rendering_raytracing_stb
        HighFive
)


find_package(OpenCV COMPONENTS core imgcodecs imgproc videoio)
if(OpenCV_FOUND)
add_executable(rendering_raytracing_yaw_prediction_dataset_overlay yaw_prediction_dataset_overlay.cpp)
set_target_properties(rendering_raytracing_yaw_prediction_dataset_overlay PROPERTIES CXX_STANDARD 17)
target_include_directories(rendering_raytracing_yaw_prediction_dataset_overlay PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rendering_raytracing_yaw_prediction_dataset_overlay
    PRIVATE
        nlohmann_json::nlohmann_json
        HighFive
        ${OpenCV_LIBS}
)
endif()

if(RL_TOOLS_ENABLE_HDF5)
    add_subdirectory(swift_app)
endif()

if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_BACKEND_ENABLE_CUDNN)
    # Per-resolution scene library (reused across ablation targets with same resolution)
    function(add_yaw_prediction_scene_lib RESOLUTION)
        set(TARGET rendering_raytracing_yaw_prediction_scene_${RESOLUTION})
        if(NOT TARGET ${TARGET})
            add_library(${TARGET} STATIC scene.cpp)
            set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 20)
            target_compile_definitions(${TARGET} PRIVATE ABLATION_RESOLUTION=${RESOLUTION})
            target_link_libraries(${TARGET} PRIVATE
                rl_tools_full owl::owl rendering_raytracing_device_ptx
                assimp::assimp rendering_raytracing_stb)
        endif()
    endfunction()

    # Ablation target (creates scene lib if needed, then builds training binary)
    function(add_yaw_prediction_ablation TARGET_SUFFIX)
        cmake_parse_arguments(ARG "" "CHANNEL_MULTIPLIER;USE_CROSS_CONV;RESOLUTION" "" ${ARGN})
        add_yaw_prediction_scene_lib(${ARG_RESOLUTION})
        set(TARGET_NAME rendering_raytracing_yaw_prediction_${TARGET_SUFFIX})
        add_executable(${TARGET_NAME} yaw_prediction_cuda.cu)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_STANDARD 20)
        set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(${TARGET_NAME} PRIVATE -O3)
        if(NOT APPLE)
            target_link_options(${TARGET_NAME} PRIVATE -Wl,--allow-multiple-definition)
        endif()
        target_compile_definitions(${TARGET_NAME} PRIVATE
            ABLATION_CHANNEL_MULTIPLIER=${ARG_CHANNEL_MULTIPLIER}
            ABLATION_USE_CROSS_CONV=${ARG_USE_CROSS_CONV}
            ABLATION_RESOLUTION=${ARG_RESOLUTION}
        )
        target_link_libraries(${TARGET_NAME} PRIVATE
            rl_tools_full owl::owl rendering_raytracing_device_ptx
            assimp::assimp rendering_raytracing_stb
            rendering_raytracing_yaw_prediction_scene_${ARG_RESOLUTION}
        )
    endfunction()

    add_yaw_prediction_ablation(base       CHANNEL_MULTIPLIER 1 USE_CROSS_CONV 0 RESOLUTION 64)
    add_yaw_prediction_ablation(crossconv  CHANNEL_MULTIPLIER 1 USE_CROSS_CONV 1 RESOLUTION 64)
    add_yaw_prediction_ablation(2x         CHANNEL_MULTIPLIER 2 USE_CROSS_CONV 1 RESOLUTION 64)
    add_yaw_prediction_ablation(128        CHANNEL_MULTIPLIER 1 USE_CROSS_CONV 1 RESOLUTION 128)

    add_yaw_prediction_scene_lib(64)
    if(RL_TOOLS_ENABLE_HDF5)
        add_executable(rendering_raytracing_yaw_prediction_eval yaw_prediction_eval.cu)
        set_target_properties(rendering_raytracing_yaw_prediction_eval PROPERTIES CUDA_STANDARD 20)
        set_property(TARGET rendering_raytracing_yaw_prediction_eval PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(rendering_raytracing_yaw_prediction_eval PRIVATE -O3)
        if(NOT APPLE)
            target_link_options(rendering_raytracing_yaw_prediction_eval PRIVATE -Wl,--allow-multiple-definition)
        endif()
        target_link_libraries(rendering_raytracing_yaw_prediction_eval
            PRIVATE
                rl_tools_full
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
                rendering_raytracing_yaw_prediction_scene_64
        )

        add_executable(rendering_raytracing_yaw_prediction_distill yaw_prediction_distill_cuda.cu)
        set_target_properties(rendering_raytracing_yaw_prediction_distill PROPERTIES CUDA_STANDARD 20)
        set_property(TARGET rendering_raytracing_yaw_prediction_distill PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(rendering_raytracing_yaw_prediction_distill PRIVATE -O3)
        if(NOT APPLE)
            target_link_options(rendering_raytracing_yaw_prediction_distill PRIVATE -Wl,--allow-multiple-definition)
        endif()
        target_link_libraries(rendering_raytracing_yaw_prediction_distill
            PRIVATE
                rl_tools_full
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
                rendering_raytracing_yaw_prediction_scene_64
        )

        add_executable(rendering_raytracing_yaw_prediction_viewer yaw_prediction_viewer.cu)
        set_target_properties(rendering_raytracing_yaw_prediction_viewer PROPERTIES CUDA_STANDARD 20)
        set_property(TARGET rendering_raytracing_yaw_prediction_viewer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(rendering_raytracing_yaw_prediction_viewer PRIVATE -O3)
        if(NOT APPLE)
            target_link_options(rendering_raytracing_yaw_prediction_viewer PRIVATE -Wl,--allow-multiple-definition)
        endif()
        target_link_libraries(rendering_raytracing_yaw_prediction_viewer
            PRIVATE
                rl_tools_full
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
                rendering_raytracing_yaw_prediction_scene_64
        )
    endif()
endif()
