add_library(rendering_raytracing_yaw_prediction_scene STATIC scene.cpp)
set_target_properties(rendering_raytracing_yaw_prediction_scene PROPERTIES CXX_STANDARD 20)
target_link_libraries(rendering_raytracing_yaw_prediction_scene
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_yaw_prediction yaw_prediction_cuda.cu)
set_target_properties(rendering_raytracing_yaw_prediction PROPERTIES CUDA_STANDARD 20)
target_compile_options(rendering_raytracing_yaw_prediction PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_property(TARGET rendering_raytracing_yaw_prediction PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# Allow multiple definitions: cpu_mux.h non-inline functions appear in both scene.cpp and .cu TUs
if(NOT APPLE)
    target_link_options(rendering_raytracing_yaw_prediction PRIVATE -Wl,--allow-multiple-definition)
endif()
target_link_libraries(rendering_raytracing_yaw_prediction
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
        rendering_raytracing_yaw_prediction_scene
)
