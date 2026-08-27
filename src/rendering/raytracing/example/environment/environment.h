#pragma once

#include <rl_tools/rendering/raytracing/renderer.h>
#include <array>

namespace rl_tools::rl::environments::raytracing_example {
    template <typename T_T, typename T_TI, T_TI T_NUM_ENVS, T_TI T_CAM_WIDTH = 64, T_TI T_CAM_HEIGHT = 64, T_TI T_NUM_PROBES = 64, typename T_SHADING = rendering::raytracing::Medium, bool T_ENABLE_MOTION_BLUR = false, T_TI T_MOTION_BLUR_SAMPLES = 1, bool T_ENABLE_ANTI_ALIASING = false, T_TI T_ANTI_ALIASING_GRID_SIZE = 1, bool T_OUTPUT_RGB = true, bool T_OUTPUT_DEPTH = false, bool T_OUTPUT_SEGMENTATION = false>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using SHADING = T_SHADING;
        static constexpr TI NUM_ENVS = T_NUM_ENVS;
        static constexpr TI CAM_WIDTH = T_CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = T_CAM_HEIGHT;
        static constexpr T FOV = 1.3962634015954636;
        static constexpr TI NUM_PROBES = T_NUM_PROBES;
        static constexpr bool ENABLE_MOTION_BLUR = T_ENABLE_MOTION_BLUR;
        static constexpr TI MOTION_BLUR_SAMPLES = T_MOTION_BLUR_SAMPLES;
        static constexpr bool ENABLE_ANTI_ALIASING = T_ENABLE_ANTI_ALIASING;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = T_ANTI_ALIASING_GRID_SIZE;
        static constexpr bool HAS_RGB = T_OUTPUT_RGB;
        static constexpr bool HAS_DEPTH = T_OUTPUT_DEPTH;
        static constexpr bool ENABLE_DEPTH = HAS_DEPTH;
        struct RENDERER_CONFIG: rendering::raytracing::config::Default<T, TI>{
            static constexpr TI CAM_WIDTH = T_CAM_WIDTH, CAM_HEIGHT = T_CAM_HEIGHT, NUM_CAMERAS = T_NUM_ENVS, NUM_PROBES = T_NUM_PROBES;
            using SHADING = T_SHADING;
            static constexpr bool OUTPUT_RGB = T_OUTPUT_RGB;
            static constexpr bool OUTPUT_DEPTH = T_OUTPUT_DEPTH;
            static constexpr bool OUTPUT_SEGMENTATION = T_OUTPUT_SEGMENTATION;
            static constexpr bool ENABLE_MOTION_BLUR = T_ENABLE_MOTION_BLUR;
            static constexpr TI MOTION_BLUR_SAMPLES = T_MOTION_BLUR_SAMPLES;
            static constexpr bool ENABLE_ANTI_ALIASING = T_ENABLE_ANTI_ALIASING;
            static constexpr TI ANTI_ALIASING_GRID_SIZE = T_ANTI_ALIASING_GRID_SIZE;
        };
        using RAYTRACING_SPEC = rendering::raytracing::Specification<RENDERER_CONFIG>;
    };

    enum class ObjectID : int {
        CAMERA_RIG = 0,
        COUNT
    };

    template <typename SPEC>
    struct Parameters {
        using T = typename SPEC::T;
        T scene_translation[3];
        T base_height;
    };

    template <typename SPEC>
    struct State {
        using T = typename SPEC::T;
        T position[3];
        T velocity[3];
        T yaw;
    };

    template <typename SPEC>
    struct ObservationRGB {
        using TI = typename SPEC::TI;
        static constexpr TI DIM = SPEC::CAM_HEIGHT * SPEC::CAM_WIDTH;
    };

    template <typename SPEC>
    struct Environment {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using RAYTRACING_SPEC = typename SPEC::RAYTRACING_SPEC;
        using Renderer = rendering::raytracing::Renderer<RAYTRACING_SPEC>;

        Renderer* renderer = nullptr;
        rendering::raytracing::Scene* scene = nullptr;
        const char* scene_path = nullptr;
        T dt = 1.0f / 60.0f;
        T max_velocity = 1.5f;
        T acceleration = 3.0f;
        T look_ahead = 1.0f;
        T eye_height = 1.6f;
        bool owns_renderer = false;
        bool owns_scene = false;

        static constexpr TI NUM_INITIAL_STATES = SPEC::NUM_ENVS;
        std::array<State<SPEC>, NUM_INITIAL_STATES> indoor_initial_states{};
        TI num_indoor_initial_states = 0;
    };
}
