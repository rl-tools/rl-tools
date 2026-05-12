#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_MULTIROTOR_VISUAL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_MULTIROTOR_VISUAL_H

#include "../environments.h"
#include "../observation.h"
#include "../l2f/multirotor.h"
#include "../../../rendering/raytracing/renderer.h"
#include "../../../rendering/raytracing/scene/procthor/scene.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f_visual {

    template <typename T_T, typename T_TI,
        typename T_DYNAMICS_STATIC_PARAMETERS,
        T_TI T_NUM_ENVS,
        T_TI T_CAM_WIDTH = 64,
        T_TI T_CAM_HEIGHT = 64,
        T_TI T_NUM_PROBES = 64,
        bool T_HIGH_FIDELITY_SHADING = false,
        bool T_ENABLE_MOTION_BLUR = false,
        T_TI T_MOTION_BLUR_SAMPLES = 1,
        bool T_ENABLE_ANTI_ALIASING = false,
        T_TI T_ANTI_ALIASING_GRID_SIZE = 1>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using DYNAMICS_STATIC_PARAMETERS = T_DYNAMICS_STATIC_PARAMETERS;
        static constexpr TI NUM_ENVS = T_NUM_ENVS;
        static constexpr TI CAM_WIDTH = T_CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = T_CAM_HEIGHT;
        static constexpr TI NUM_PROBES = T_NUM_PROBES;
        static constexpr bool HIGH_FIDELITY_SHADING = T_HIGH_FIDELITY_SHADING;
        static constexpr bool ENABLE_MOTION_BLUR = T_ENABLE_MOTION_BLUR;
        static constexpr TI MOTION_BLUR_SAMPLES = T_MOTION_BLUR_SAMPLES;
        static constexpr bool ENABLE_ANTI_ALIASING = T_ENABLE_ANTI_ALIASING;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = T_ANTI_ALIASING_GRID_SIZE;

        using DYNAMICS_SPEC = l2f::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS>;
        using DYNAMICS_ENV = Multirotor<DYNAMICS_SPEC>;
        using RENDERER_SPEC = rendering::raytracing::Specification<T, TI, CAM_WIDTH, CAM_HEIGHT, NUM_ENVS, NUM_PROBES, HIGH_FIDELITY_SHADING, ENABLE_MOTION_BLUR, MOTION_BLUR_SAMPLES, ENABLE_ANTI_ALIASING, ANTI_ALIASING_GRID_SIZE>;
        using SCENE_SPEC = rendering::raytracing::scene::SceneSpecification<T, TI>;
    };

    struct SceneHash {
        static constexpr unsigned HASH_SIZE = 20; // SHA-1
        unsigned char hash[HASH_SIZE] = {0};
    };

    template <typename T>
    struct CameraMount {
        T offset_body[3] = {0, 0, 0};
        T forward_body[3] = {1, 0, 0};
        T up_body[3] = {0, 0, 1};
    };

    template <typename T>
    struct CameraRandomization {
        T fov_range = 0;
        T offset_body_range[3] = {0, 0, 0};
        T rotation_body_range[3] = {0, 0, 0};
    };

    template <typename T_SPEC>
    struct Parameters {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        typename SPEC::DYNAMICS_ENV::Parameters dynamics = SPEC::DYNAMICS_SPEC::PARAMETER_VALUES;
        T scene_translation[3] = {0, 0, 0};
        T scene_yaw = 0;
        SceneHash scene_hash;
        CameraMount<T> camera_mount;
        T fov = 1.1132;
        CameraRandomization<T> camera_randomization;
        T collision_distance_threshold = 0.15;
    };

    template <typename T_SPEC>
    struct MultirrotorVisual : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        using DYNAMICS_ENV = typename SPEC::DYNAMICS_ENV;
        using State = typename DYNAMICS_ENV::State;
        using Parameters = l2f_visual::Parameters<SPEC>;

        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = DYNAMICS_ENV::ACTION_DIM;
        static constexpr TI EPISODE_STEP_LIMIT = DYNAMICS_ENV::EPISODE_STEP_LIMIT;

        using Observation = rl::environments::observation::Image<TI, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, 3>;
        using ObservationPrivileged = typename DYNAMICS_ENV::Observation;
        static constexpr TI OBSERVATION_DIM = Observation::DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ObservationPrivileged::DIM;
        static constexpr bool PRIVILEGED_OBSERVATION_AVAILABLE = true;

        using RENDERER_SPEC = typename SPEC::RENDERER_SPEC;
        rendering::raytracing::Renderer<RENDERER_SPEC>* renderer = nullptr;
        rendering::raytracing::scene::procthor::Scene<typename SPEC::SCENE_SPEC>* scene = nullptr;
        bool owns_renderer = false;

        const char* scene_path = nullptr;

        bool use_target_mode = false;
        bool renderer_initialized = false;

        DYNAMICS_ENV dynamics;
        Parameters parameters;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
