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
#include "../../../rendering/datasets/procthor/procthor.h"
#include "../../../rendering/datasets/annotations/free_space.h"
#include "../../../rendering/datasets/annotations/cache.h"

#include <string>
#include <vector>
#include <memory>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone {

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
        static constexpr bool SEMANTIC_SEGMENTATION = false;
        static constexpr bool OUTPUT_NORMALS = false;
        static constexpr bool OUTPUT_FLOW = false;
        static constexpr TI HISTORY_LENGTH = 1;
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
        // external mounts); every SharedContext::drones entry then needs a body/prop_* assembly
        static constexpr bool SELF_VISIBLE = false;
        // capacity of the drone table (SharedContext::drones): every agent flies one entry,
        // sampled uniformly at reset, body included; the table is device-resident, hence the
        // compile-time bound
        static constexpr TI MAX_DRONES = 1;
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
            static constexpr bool SEMANTIC_SEGMENTATION = SPEC::SEMANTIC_SEGMENTATION;
            static constexpr bool OUTPUT_NORMALS = SPEC::OUTPUT_NORMALS;
            static constexpr bool OUTPUT_FLOW = SPEC::OUTPUT_FLOW;
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
    struct GroupParameters {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        // scene-under-drone: the dynamics state stays near the origin; these place the drone's
        // frame inside the scene at camera-production time
        T scene_translation[3] = {0, 0, 0};
        T scene_yaw_cos = 1;
        T scene_yaw_sin = 0;
        T brightness_scale = 1;
        T shutter_fraction = 1;
    };

    // a drone table entry: the assembly to render and the nominal the agent's parameters are
    // sampled from. The dynamics are the full l2f parameter set because the l2f sampler copies
    // its environment's nominal before randomizing on top of it
    template <typename T_SPEC>
    struct DroneParameters {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using DYNAMICS_ENV = Multirotor<l2f::Specification<typename SPEC::T, typename SPEC::TI, typename SPEC::DYNAMICS_STATIC_PARAMETERS>>;
        typename DYNAMICS_ENV::Parameters dynamics = SPEC::DYNAMICS_STATIC_PARAMETERS::PARAMETER_VALUES;
        Camera<T> camera;
    };
    template <typename T_SPEC>
    struct AgentParameters: DroneParameters<T_SPEC> {
        typename T_SPEC::TI drone = 0;
    };
    template <typename T_SPEC>
    struct Parameters: GroupParameters<T_SPEC> {
        using DYNAMICS_ENV = typename DroneParameters<T_SPEC>::DYNAMICS_ENV;
        AgentParameters<T_SPEC> agents[T_SPEC::N_AGENTS];
    };
    template <typename T_SPEC>
    struct Drone {
        std::string asset;  // body/prop_* assembly, a path or conta:HASH; required iff SELF_VISIBLE
        DroneParameters<T_SPEC> parameters;
    };
    // how a table entry is rendered (SELF_VISIBLE): its mesh (an index into
    // SharedContext::drone_meshes) and the body/prop_* rig matched to the entry's nominal rotors,
    // so coordinate i is the phase of rotor i whatever the assembly's prop order
    template <typename T_SPEC>
    struct DroneBody {
        using TI = typename T_SPEC::TI;
        rig::Rotorcraft<TI, 4> rig;
        TI mesh = 0;
    };

    // owned by the caller (typically the MultiEnvironment) and shared by all its Worlds: the
    // AssetLibrary, the pool-asset registry (entities, deduplicated by path, registered before
    // any hot slot builds) and the drone table, built once by init(device, shared) from the
    // entries registered before the first World init (none: that World registers its nominal
    // as the single drone). Worlds copy what they need into device tables at init
    template <typename T_SPEC>
    struct SharedContext {
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        using LIBRARY = rendering::raytracing::AssetLibrary<rendering::raytracing::Specification<world::RendererConfig<SPEC>>>;
        using DRONE = hyperdrone::Drone<SPEC>;
        LIBRARY library;
        std::shared_ptr<const rendering::SegmentationTaxonomy> segmentation_taxonomy;
        std::vector<std::string> pool_asset_paths;
        std::vector<rendering::raytracing::AssetHandle> pool_asset_handles;
        // opt-in: set the directory before init to persist and reuse free-space scans
        rendering::datasets::annotations::Cache annotation_cache;
        std::vector<Drone<SPEC>> drones;  // registered before the first World init
        TI num_drones = 0;  // 0 until built
        // SELF_VISIBLE: one pooled mesh per distinct assembly content, all with drone_parts parts
        // (a swap then keeps every slot, see replace), and one body per entry
        std::vector<rendering::raytracing::AssetHandle> drone_meshes;
        TI drone_parts = 0;
        std::vector<DroneBody<SPEC>> drone_bodies;
    };

    template <typename T_SPEC>
    struct World: Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static_assert(SPEC::MAX_DRONES > 0, "MAX_DRONES must be positive");
        static_assert(SPEC::HISTORY_LENGTH > 0, "HISTORY_LENGTH must be positive");
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
        static constexpr TI IMAGE_CHANNELS = 3;

        // agent frames stack along the image height so the flat row layout matches the
        // camera-major frame history exactly
        using Observation = rl::environments::observation::Image<TI, N_AGENTS * SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, IMAGE_CHANNELS>;
        struct ObservationPrivilegedMultiAgent {
            static constexpr TI PER_AGENT_DIM = DYNAMICS_ENV::ObservationPrivileged::DIM;
            static constexpr TI DIM = N_AGENTS * PER_AGENT_DIM;
        };
        using ObservationPrivileged = utils::typing::conditional_t<SPEC::N_AGENTS == 1, typename DYNAMICS_ENV::ObservationPrivileged, ObservationPrivilegedMultiAgent>;
        static constexpr TI OBSERVATION_DIM = Observation::DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ObservationPrivileged::DIM;
        static constexpr bool PRIVILEGED_OBSERVATION_AVAILABLE = true;

        using RENDERER_CONFIG = world::RendererConfig<SPEC>;
        using RENDERER_SPEC = rendering::raytracing::Specification<RENDERER_CONFIG>;
        static_assert(IMAGE_CHANNELS == RENDERER_SPEC::OBSERVATION_CHANNELS, "the image observation must match the renderer's RGB channels");
        using RENDERER = rendering::raytracing::Renderer<RENDERER_SPEC>;
        using ANNOTATIONS_SPEC = rendering::datasets::annotations::FreeSpaceSpecification<T, TI>;
        using ANNOTATIONS = rendering::datasets::annotations::FreeSpace<ANNOTATIONS_SPEC>;
        using LIBRARY = rendering::raytracing::AssetLibrary<RENDERER_SPEC>;
        using SharedContext = hyperdrone::SharedContext<SPEC>;

        // entity rigs produce the part poses of an entity from (base pose, coordinates); an entity
        // occupies one slot per part, so the per-instance slot budget bounds links and parts
        static constexpr TI ENTITY_RIG_CAPACITY = SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE > 0 ? SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE : 1;
        using ENTITY_RIG_SPEC = rendering::rig::Specification<float, TI, ENTITY_RIG_CAPACITY, ENTITY_RIG_CAPACITY>;
        using ENTITY_RIG = rendering::rig::Model<ENTITY_RIG_SPEC>;

        // a registered entity kind (register_entity_kind): its pooled asset, its first slot and its
        // rig, which for a rigid entity has no coordinates. Spawn order is pinned (instance-major,
        // registration order, then the drone region, first-fit), so the kind's rows sit at
        // instance * MAX_ENTITY_SLOTS_PER_INSTANCE + slot_offset in every instance and hot slot
        struct EntityKind {
            rendering::raytracing::AssetHandle asset;
            TI slot_offset;
            ENTITY_RIG rig;
        };

        // hot slot: the scene build in the shared library plus this World's thin per-camera
        // renderer state; every slot is initialized exactly once
        struct HotSlot {
            RENDERER renderer;
            ANNOTATIONS annotations;
            rendering::SceneMetadata<T> metadata;
            TI corpus_index;
        };

        std::vector<HotSlot> slots;        // this World's partition of the dataset corpus
        std::vector<EntityKind> entity_kinds;  // frozen at init; empty when no wrapper registers any
        TI active_slot = 0;
        TI scene_version = 0;  // scene selections so far, init's included
        TI member_index = 0;
        TI rng_offset = 0;
        DYNAMICS_ENV dynamics;

        // per-frame runtime state, resident on the environment's compute device
        static constexpr TI FRAME_DIM = SPEC::CAM_HEIGHT * SPEC::CAM_WIDTH * IMAGE_CHANNELS;
        using HISTORY_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::HISTORY_LENGTH, INSTANCES * N_VIEWS, FRAME_DIM>>;
        using PREV_CAMERAS_SPEC = tensor::Specification<rendering::raytracing::Camera<T>, TI, tensor::Shape<TI, INSTANCES * N_VIEWS>>;
        using EPISODE_START_SPEC = tensor::Specification<TI, TI, tensor::Shape<TI, INSTANCES>>;
        using RENDER_RESET_SPEC = tensor::Specification<bool, TI, tensor::Shape<TI, INSTANCES>>;
        using ACTIVE_ANNOTATIONS_SPEC = tensor::Specification<ANNOTATIONS, TI, tensor::Shape<TI, 1>>;
        Tensor<HISTORY_SPEC> history;
        Tensor<PREV_CAMERAS_SPEC> prev_cameras;  // shutter-open interpolation source (motion blur)
        Tensor<EPISODE_START_SPEC> episode_start;  // history slot at which each instance's episode began
        Tensor<RENDER_RESET_SPEC> render_reset;
        Tensor<ACTIVE_ANNOTATIONS_SPEC> active_annotations;  // device-visible copy of the active slot's annotation tables
        // host-resident camera staging for the CPU render verb (pre-allocated: render runs per step)
        Tensor<PREV_CAMERAS_SPEC> camera_staging_close;
        Tensor<PREV_CAMERAS_SPEC> camera_staging_previous;
        Tensor<PREV_CAMERAS_SPEC> camera_staging_open;
        TI history_step = 0;
        TI flow_scene_version = 0;
        bool render_pending = false;

        // the drone table's device copy: the sampler draws each agent's drone at reset and the body
        // follows it at every render. SELF_VISIBLE: each instance's drone region holds
        // N_AGENTS * drone_parts slots starting at drone_slot_offset, after the entity kinds
        using DRONE_PARAMETERS_SPEC = tensor::Specification<DroneParameters<SPEC>, TI, tensor::Shape<TI, SPEC::MAX_DRONES>>;
        using DRONE_BODIES_SPEC = tensor::Specification<DroneBody<SPEC>, TI, tensor::Shape<TI, SPEC::MAX_DRONES>>;
        Tensor<DRONE_PARAMETERS_SPEC> drone_parameters;
        Tensor<DRONE_BODIES_SPEC> drone_bodies;
        TI num_drones = 0;
        std::vector<rendering::raytracing::AssetHandle> drone_meshes;
        TI drone_parts = 0;
        TI drone_slot_offset = 0;

        // defaults for sampled parameters (fov/mount/brightness ranges are spec constants); the
        // dynamics and camera nominal is drone 0 after init
        Parameters parameters;
    };

    // the composite is itself an environment: it satisfies the batch-verb contract by fanning
    // out to its members over contiguous instance blocks (flat, member-major). Members are
    // identical Worlds (or task wrappers over Worlds); the shared AssetLibrary lives in the
    // member-defined SharedContext, the dataset corpus is enumerated at init
    template <typename T_ENVIRONMENT, typename T_ENVIRONMENT::TI T_NUMBER_OF_ENVIRONMENTS>
    struct MultiEnvironment{
        using ENVIRONMENT = T_ENVIRONMENT;
        using T = typename ENVIRONMENT::T;
        using TI = typename ENVIRONMENT::TI;
        static constexpr TI NUMBER_OF_ENVIRONMENTS = T_NUMBER_OF_ENVIRONMENTS;
        static_assert(NUMBER_OF_ENVIRONMENTS > 0);

        using State = typename ENVIRONMENT::State;
        using Parameters = typename ENVIRONMENT::Parameters;
        using Observation = typename ENVIRONMENT::Observation;
        using ObservationPrivileged = typename ENVIRONMENT::ObservationPrivileged;
        static constexpr TI N_AGENTS = ENVIRONMENT::N_AGENTS;
        static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
        static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
        static constexpr TI INSTANCES_PER_ENVIRONMENT = ENVIRONMENT::INSTANCES;
        static constexpr TI INSTANCES = NUMBER_OF_ENVIRONMENTS * INSTANCES_PER_ENVIRONMENT;

        using SharedContext = typename ENVIRONMENT::SharedContext;
        SharedContext shared;
        ENVIRONMENT environments[NUMBER_OF_ENVIRONMENTS];
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
