#pragma once

#include <rl_tools/rendering/raytracing/renderer.h>

namespace rl_tools::rl::environments::raytracing_example {
    template <typename T_T, typename T_TI, T_TI T_NUM_ENVS, T_TI T_CAM_WIDTH = 64, T_TI T_CAM_HEIGHT = 64, T_TI T_NUM_PROBES = 64>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI NUM_ENVS = T_NUM_ENVS;
        static constexpr TI CAM_WIDTH = T_CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = T_CAM_HEIGHT;
        static constexpr TI NUM_PROBES = T_NUM_PROBES;
        using RAYTRACING_SPEC = rendering::raytracing::Specification<T, TI, CAM_WIDTH, CAM_HEIGHT, NUM_ENVS, NUM_PROBES>;
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
        const char* scene_path = nullptr;
        T dt = 1.0f / 60.0f;
        T max_velocity = 1.5f;
        T acceleration = 3.0f;
        T look_ahead = 1.0f;
        T eye_height = 1.6f;
        bool owns_renderer = false;
    };
}
