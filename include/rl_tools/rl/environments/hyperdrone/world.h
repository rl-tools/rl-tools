#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD_H

#include "../environments.h"
#include "../observation.h"
#include "../l2f/multirotor.h"
#include "pose.h"
#include "rig/rig.h"
#include "../../../rendering/raytracing/renderer.h"
#include "../../../rendering/raytracing/scene/procthor/scene.h"

#include <string>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone {

    namespace datasets {
        // bare GLBs, no transforms: enumerated as a lexicographically sorted directory walk so
        // the SceneSet is filesystem-order independent
        struct Plain {
            std::string directory;
        };
    }

    // a scene dataset's output: the enumerated corpus the MultiEnvironment schedules over
    struct SceneSet {
        std::vector<std::string> paths;
    };

    template <typename T_T, typename T_TI, typename T_DYNAMICS_STATIC_PARAMETERS>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using DYNAMICS_STATIC_PARAMETERS = T_DYNAMICS_STATIC_PARAMETERS;
        // derive-and-shadow: override by deriving from this Specification and shadowing members
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 64;
        // agents per instance: one FPV camera per agent (N_VIEWS == N_AGENTS); N_AGENTS > 1
        // requires SELF_VISIBLE (agents seeing each other needs geometry to see)
        static constexpr TI N_AGENTS = 1;
        static constexpr T AGENT_COLLISION_DISTANCE = 0.15;
        static constexpr TI CAM_WIDTH = 64;
        static constexpr TI CAM_HEIGHT = 64;
        static constexpr TI NUM_PROBES = 1;
        using SHADING = rendering::raytracing::High;
        static constexpr bool ENABLE_MOTION_BLUR = false;
        static constexpr TI MOTION_BLUR_SAMPLES = 1;
        static constexpr bool ENABLE_ANTI_ALIASING = false;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = 1;
        static constexpr bool OUTPUT_RGB = true;
        static constexpr bool OUTPUT_DEPTH = false;
        static constexpr bool OUTPUT_SEGMENTATION = false;
        static constexpr TI HISTORY_LENGTH = 1;
        static constexpr TI EPISODES_PER_SCENE = 1;
        static constexpr T CAMERA_FOV = 1.1132;
        static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = 0;
        static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = 0;
        static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = 0;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0;
        static constexpr T SHUTTER_FRACTION_MIN = 0.25;
        static constexpr T SHUTTER_FRACTION_MAX = 1.0;
        // everything that moves is an overlay entity: one overlay group per instance, sized by
        // this per-instance slot budget; 0 compiles the overlay machinery away (flying camera)
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 0;
        // attach the environment's own drone entity to its own cameras (FPV self-occlusion /
        // external mounts); requires drone_asset_path (body/prop_* convention) set before init
        static constexpr bool SELF_VISIBLE = false;
    };

    namespace world {
        template <typename STATE, typename = void>
        struct HasRotorPhase {
            static constexpr bool VALUE = false;
        };
        template <typename STATE>
        struct HasRotorPhase<STATE, utils::typing::void_t<decltype(STATE{}.rotor_phase)>> {
            static constexpr bool VALUE = true;
        };
        template <typename T_AGENT_STATE, typename T_TI, T_TI T_N_AGENTS>
        struct MultiAgentState {
            using AGENT_STATE = T_AGENT_STATE;
            static constexpr T_TI N_AGENTS = T_N_AGENTS;
            static constexpr T_TI DIM = N_AGENTS * AGENT_STATE::DIM;
            AGENT_STATE agent_states[N_AGENTS];
        };
        template <typename T_SPEC>
        struct RendererConfig: rendering::raytracing::config::Default<typename T_SPEC::T, typename T_SPEC::TI> {
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            static constexpr TI CAM_WIDTH = SPEC::CAM_WIDTH;
            static constexpr TI CAM_HEIGHT = SPEC::CAM_HEIGHT;
            static constexpr TI NUM_CAMERAS = SPEC::INSTANCES_PER_ENVIRONMENT * SPEC::N_AGENTS;
            static constexpr TI NUM_PROBES = SPEC::NUM_PROBES;
            using SHADING = typename SPEC::SHADING;
            static constexpr bool OUTPUT_RGB = SPEC::OUTPUT_RGB;
            static constexpr bool OUTPUT_DEPTH = SPEC::OUTPUT_DEPTH;
            static constexpr bool OUTPUT_SEGMENTATION = SPEC::OUTPUT_SEGMENTATION;
            static constexpr bool ENABLE_MOTION_BLUR = SPEC::ENABLE_MOTION_BLUR;
            static constexpr TI MOTION_BLUR_SAMPLES = SPEC::MOTION_BLUR_SAMPLES;
            static constexpr bool ENABLE_ANTI_ALIASING = SPEC::ENABLE_ANTI_ALIASING;
            static constexpr TI ANTI_ALIASING_GRID_SIZE = SPEC::ANTI_ALIASING_GRID_SIZE;
            // the RGB observation is consumed as the renderer's float observation output —
            // written by the ray gen at full precision, no format-conversion pass
            static constexpr bool OUTPUT_OBSERVATION = SPEC::OUTPUT_RGB;
            static constexpr TI NUM_OVERLAYS = SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE > 0 ? SPEC::INSTANCES_PER_ENVIRONMENT : 0;
            static constexpr TI MAX_OVERLAY_INSTANCES = SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE;
            static constexpr TI MAX_OVERLAYS_PER_CAMERA = SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE > 0 ? 1 : 0;
            static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = SPEC::ENABLE_MOTION_BLUR && SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE > 0;
        };
    }

    template <typename T_SPEC>
    struct Parameters {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using DYNAMICS_ENV = Multirotor<l2f::Specification<typename SPEC::T, typename SPEC::TI, typename SPEC::DYNAMICS_STATIC_PARAMETERS>>;
        typename DYNAMICS_ENV::Parameters dynamics = SPEC::DYNAMICS_STATIC_PARAMETERS::PARAMETER_VALUES;
        // scene-under-drone: the dynamics state stays near the origin; these place the drone's
        // frame inside the scene at camera-production time
        T scene_translation[3] = {0, 0, 0};
        T scene_yaw_cos = 1;
        T scene_yaw_sin = 0;
        CameraMount<T> camera_mount;
        T fov = SPEC::CAMERA_FOV;
        T brightness_scale = 1;
        T shutter_fraction = 1;
    };

    template <typename T_SPEC>
    struct World: Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr bool HYPERDRONE_WORLD = true;  // marker for member-generic composite overloads
        static constexpr bool BATCH_NATIVE = true;      // opts out of the generic mapped batch defaults
        static_assert(!SPEC::SELF_VISIBLE || SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE > 0, "SELF_VISIBLE needs entity slots: set MAX_ENTITY_SLOTS_PER_INSTANCE");
        static_assert(SPEC::N_AGENTS == 1 || SPEC::SELF_VISIBLE, "multi-agent Worlds require SELF_VISIBLE: agents seeing each other needs geometry to see");

        using DYNAMICS_ENV = Multirotor<l2f::Specification<T, TI, typename SPEC::DYNAMICS_STATIC_PARAMETERS>>;
        using State = utils::typing::conditional_t<SPEC::N_AGENTS == 1, typename DYNAMICS_ENV::State, world::MultiAgentState<typename DYNAMICS_ENV::State, TI, SPEC::N_AGENTS>>;
        using Parameters = hyperdrone::Parameters<SPEC>;

        static constexpr TI INSTANCES = SPEC::INSTANCES_PER_ENVIRONMENT;
        static constexpr TI N_AGENTS = SPEC::N_AGENTS;
        static constexpr TI N_VIEWS = N_AGENTS;
        static constexpr TI ACTION_DIM = N_AGENTS * DYNAMICS_ENV::ACTION_DIM;
        static constexpr TI EPISODE_STEP_LIMIT = DYNAMICS_ENV::EPISODE_STEP_LIMIT;
        static constexpr TI IMAGE_CHANNELS = SPEC::OUTPUT_DEPTH ? (SPEC::OUTPUT_RGB ? 4 : 1) : 3;

        // agent frames stack along the image height so the flat row layout matches the
        // camera-major frame history exactly
        using Observation = rl::environments::observation::Image<TI, N_AGENTS * SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, IMAGE_CHANNELS>;
        struct ObservationPrivilegedMultiAgent {
            static constexpr TI PER_AGENT_DIM = DYNAMICS_ENV::Observation::DIM;
            static constexpr TI DIM = N_AGENTS * PER_AGENT_DIM;
        };
        using ObservationPrivileged = utils::typing::conditional_t<SPEC::N_AGENTS == 1, typename DYNAMICS_ENV::Observation, ObservationPrivilegedMultiAgent>;
        static constexpr TI OBSERVATION_DIM = Observation::DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ObservationPrivileged::DIM;
        static constexpr bool PRIVILEGED_OBSERVATION_AVAILABLE = true;

        using RENDERER_CONFIG = world::RendererConfig<SPEC>;
        using RENDERER_SPEC = rendering::raytracing::Specification<RENDERER_CONFIG>;
        using RENDERER = rendering::raytracing::Renderer<RENDERER_SPEC>;
        using SCENE_SPEC = rendering::raytracing::scene::SceneSpecification<T, TI>;
        using SCENE = rendering::raytracing::scene::procthor::Scene<SCENE_SPEC>;
        using LIBRARY = rendering::raytracing::AssetLibrary<RENDERER_SPEC>;

        struct SharedContext {
            LIBRARY library;
            SceneSet scene_set;
            // pool-asset registry (entities): deduplicated by path, registered into the shared
            // library's pool before any hot slot builds
            std::vector<std::string> pool_asset_paths;
            std::vector<rendering::raytracing::AssetHandle> pool_asset_handles;
        };

        // a registered entity kind: (asset, parts, per-instance slot offset); registrations are
        // accumulated by wrapper init overloads before the innermost base init consumes them once
        struct EntityKind {
            rendering::raytracing::AssetHandle asset;
            TI parts;
            TI slot_offset;
        };

        // hot slot: the scene build in the shared library plus this World's thin per-camera
        // renderer state; every slot is initialized exactly once, rotation repoints the view.
        // With entities enabled a slot also carries its overlay placements (spawned per slot at
        // slot init, in pinned instance-major/kind order)
        struct HotSlot {
            RENDERER renderer;
            SCENE scene;
            TI scene_set_index;
            std::vector<rendering::raytracing::OverlayPlacement> entity_placements;  // [instance * kinds + kind]
        };

        RENDERER renderer;                 // view of the active slot's renderer
        std::vector<HotSlot> slots;        // this World's partition of the SceneSet
        std::vector<EntityKind> entity_kinds;  // frozen at init; empty when no wrapper registers any
        TI active_slot = 0;
        TI member_index = 0;
        TI episode_counter = 0;
        DYNAMICS_ENV dynamics;

        // per-frame staging and history, resident on the renderer's device
        static constexpr TI FRAME_DIM = SPEC::CAM_HEIGHT * SPEC::CAM_WIDTH * IMAGE_CHANNELS;
        using HISTORY_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::HISTORY_LENGTH, INSTANCES * N_VIEWS, FRAME_DIM>>;
        using PREV_CAMERAS_SPEC = tensor::Specification<rendering::raytracing::Camera<T>, TI, tensor::Shape<TI, INSTANCES * N_VIEWS>>;
        using EPISODE_START_SPEC = tensor::Specification<TI, TI, tensor::Shape<TI, INSTANCES>>;
        using ACTIVE_SCENE_SPEC = tensor::Specification<SCENE, TI, tensor::Shape<TI, 1>>;
        Tensor<HISTORY_SPEC> history;
        Tensor<PREV_CAMERAS_SPEC> prev_cameras;  // shutter-open interpolation source (motion blur)
        Tensor<EPISODE_START_SPEC> episode_start;  // history slot at which each instance's episode began
        Tensor<ACTIVE_SCENE_SPEC> active_scene;  // device-visible copy of the active slot's scene tables
        TI history_step = 0;
        void* cuda_sync_event = nullptr;  // lazily created by the CUDA verbs (caller/render stream joins)

        // the World's own drone entity (SELF_VISIBLE): asset path set before init, rig derived
        // from the assembly, poses staged per render ([12 body | 4 phase_open | 4 phase_close])
        std::string drone_asset_path;
        rig::Rotorcraft<T, TI, 4> drone_rig;
        TI drone_entity_kind_index = 0;
        using DRONE_POSE_STAGING_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, INSTANCES * N_AGENTS, 20>>;
        Tensor<DRONE_POSE_STAGING_SPEC> drone_pose_staging;
        void* cuda_drone_pose_staging = nullptr;  // pinned host mirror

        // defaults for sampled parameters (fov/mount/brightness ranges are spec constants)
        Parameters parameters;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
