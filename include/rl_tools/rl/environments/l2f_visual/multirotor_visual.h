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
        typename T_SHADING = rendering::raytracing::Medium,
        bool T_ENABLE_MOTION_BLUR = false,
        T_TI T_MOTION_BLUR_SAMPLES = 1,
        bool T_ENABLE_ANTI_ALIASING = false,
        T_TI T_ANTI_ALIASING_GRID_SIZE = 1,
        bool T_OUTPUT_RGB = true,
        bool T_OUTPUT_DEPTH = false,
        bool T_OUTPUT_SEGMENTATION = false>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using SHADING = T_SHADING;
        using DYNAMICS_STATIC_PARAMETERS = T_DYNAMICS_STATIC_PARAMETERS;
        static constexpr TI NUM_ENVS = T_NUM_ENVS;
        static constexpr TI CAM_WIDTH = T_CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = T_CAM_HEIGHT;
        static constexpr TI NUM_PROBES = T_NUM_PROBES;
        static constexpr bool ENABLE_MOTION_BLUR = T_ENABLE_MOTION_BLUR;
        static constexpr TI MOTION_BLUR_SAMPLES = T_MOTION_BLUR_SAMPLES;
        static constexpr bool ENABLE_ANTI_ALIASING = T_ENABLE_ANTI_ALIASING;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = T_ANTI_ALIASING_GRID_SIZE;
        static constexpr bool HAS_RGB = T_OUTPUT_RGB;
        static constexpr bool HAS_DEPTH = T_OUTPUT_DEPTH;
        static constexpr bool ENABLE_DEPTH = HAS_DEPTH;
        static constexpr TI IMAGE_CHANNELS = HAS_DEPTH ? (HAS_RGB ? 4 : 1) : 3;

        using DYNAMICS_SPEC = l2f::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS>;
        using DYNAMICS_ENV = Multirotor<DYNAMICS_SPEC>;
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
            // the RGB observation is consumed as the renderer's float observation output —
            // written by the ray gen at full precision, no format-conversion pass
            static constexpr bool OUTPUT_OBSERVATION = T_OUTPUT_RGB;
        };
        using RENDERER_SPEC = rendering::raytracing::Specification<RENDERER_CONFIG>;
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

        using Observation = rl::environments::observation::Image<TI, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, SPEC::IMAGE_CHANNELS>;
        using ObservationPrivileged = typename DYNAMICS_ENV::Observation;
        static constexpr TI OBSERVATION_DIM = Observation::DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ObservationPrivileged::DIM;
        static constexpr bool PRIVILEGED_OBSERVATION_AVAILABLE = true;

        // non-owning references — the target owns the renderer (typically malloc'd against a
        // shared rendering::raytracing::AssetLibrary) and the scene metadata
        using RENDERER_SPEC = typename SPEC::RENDERER_SPEC;
        rendering::raytracing::Renderer<RENDERER_SPEC>* renderer = nullptr;
        rendering::raytracing::scene::procthor::Scene<typename SPEC::SCENE_SPEC>* scene = nullptr;

        bool use_target_mode = false;

        DYNAMICS_ENV dynamics;
        Parameters parameters;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
