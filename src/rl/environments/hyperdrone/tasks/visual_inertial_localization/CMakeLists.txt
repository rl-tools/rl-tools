# the RAPTOR autopilot policy is gitignored (fetched by src/nn_models/port_checkpoint/raptor/setup.sh)
set(RL_TOOLS_RAPTOR_POLICY "${CMAKE_SOURCE_DIR}/src/nn_models/port_checkpoint/raptor/policy.h")
if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING AND EXISTS "${RL_TOOLS_RAPTOR_POLICY}")
    add_executable(rl_environments_hyperdrone_tasks_visual_inertial_localization visual_inertial_localization.cpp)
    target_link_libraries(rl_environments_hyperdrone_tasks_visual_inertial_localization
        PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
    )
    add_executable(rl_environments_hyperdrone_tasks_visual_inertial_localization_export export.cpp)
    target_link_libraries(rl_environments_hyperdrone_tasks_visual_inertial_localization_export
        PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
    )
endif()

# trajectory scorer: no rendering/policy dependencies — scores any causal estimator's TUM
# trajectory against an exported episode's ground truth
add_executable(rl_environments_hyperdrone_tasks_visual_inertial_localization_score score.cpp)
target_link_libraries(rl_environments_hyperdrone_tasks_visual_inertial_localization_score PRIVATE RLtools::RLtools)

# OpenVINS baseline: gated on the out-of-tree OpenVINS build (src/vio/openvins/setup.sh);
# the resulting binary is GPL-3.0 — internal benchmarking only, never distribute
set(RL_TOOLS_OPENVINS_ROOT "${CMAKE_SOURCE_DIR}/src/vio/openvins")
if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING
        AND EXISTS "${RL_TOOLS_RAPTOR_POLICY}"
        AND EXISTS "${RL_TOOLS_OPENVINS_ROOT}/prefix/lib/libov_msckf_lib.so")
    find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d features2d video)
    find_package(Eigen3 REQUIRED)
    add_executable(rl_environments_hyperdrone_tasks_visual_inertial_localization_openvins openvins.cpp)
    target_include_directories(rl_environments_hyperdrone_tasks_visual_inertial_localization_openvins SYSTEM PRIVATE
        "${RL_TOOLS_OPENVINS_ROOT}/src/ov_msckf/src"
        "${RL_TOOLS_OPENVINS_ROOT}/src/ov_core/src"
        "${RL_TOOLS_OPENVINS_ROOT}/src/ov_init/src"
    )
    target_link_libraries(rl_environments_hyperdrone_tasks_visual_inertial_localization_openvins
        PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
        "${RL_TOOLS_OPENVINS_ROOT}/prefix/lib/libov_msckf_lib.so"
        ${OpenCV_LIBS}
        Eigen3::Eigen
    )
    set_target_properties(rl_environments_hyperdrone_tasks_visual_inertial_localization_openvins PROPERTIES
        BUILD_RPATH "${RL_TOOLS_OPENVINS_ROOT}/prefix/lib"
    )
    # GCC 15 + Eigen AVX512 emits false-positive array-bounds warnings in third-party headers
    target_compile_options(rl_environments_hyperdrone_tasks_visual_inertial_localization_openvins PRIVATE -Wno-array-bounds -Wno-maybe-uninitialized)
endif()
