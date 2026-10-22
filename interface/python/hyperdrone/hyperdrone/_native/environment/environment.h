#pragma once
#include "verbosity.h"
#include <rl_tools/operations/cpu_mux.h>

#define HYPERDRONE_ENVIRONMENT_STRINGIFY_INNER(x) #x
#define HYPERDRONE_ENVIRONMENT_STRINGIFY(x) HYPERDRONE_ENVIRONMENT_STRINGIFY_INNER(x)

// Custom controller operations must precede their callers' template definitions.
#ifdef HYPERDRONE_ENVIRONMENT_SPEC_HEADER
#include HYPERDRONE_ENVIRONMENT_STRINGIFY(HYPERDRONE_ENVIRONMENT_SPEC_HEADER)
#endif

#include "../common/control.h"
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#if defined(HYPERDRONE_ENVIRONMENT_TASK) && HYPERDRONE_ENVIRONMENT_TASK == 1
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#endif
#if defined(HYPERDRONE_ENVIRONMENT_TASK) && HYPERDRONE_ENVIRONMENT_TASK == 2
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cpu.h>
#endif
#if defined(HYPERDRONE_ENVIRONMENT_TASK) && HYPERDRONE_ENVIRONMENT_TASK == 3
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/operations_cpu.h>
#endif
#if defined(HYPERDRONE_ENVIRONMENT_PRESET) && HYPERDRONE_ENVIRONMENT_PRESET >= 1
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#endif

#include <rl_tools/rl/environments/l2f/operations_cpu.h>

#include <cstdint>
#include <optional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "output.h"
#if defined(RL_TOOLS_HYPERLUX_BACKEND_OPTIX)
#include "device_leases.h"
#endif

#if !defined(HYPERDRONE_ENVIRONMENT_NUM_WORLDS) || !defined(HYPERDRONE_ENVIRONMENT_INSTANCES) || !defined(HYPERDRONE_ENVIRONMENT_CAM_WIDTH) || !defined(HYPERDRONE_ENVIRONMENT_CAM_HEIGHT) || !defined(HYPERDRONE_ENVIRONMENT_SHADING) || !defined(HYPERDRONE_ENVIRONMENT_MB_SAMPLES) || !defined(HYPERDRONE_ENVIRONMENT_PRESET) || !defined(HYPERDRONE_ENVIRONMENT_TASK) || !defined(HYPERDRONE_ENVIRONMENT_N_AGENTS) || !defined(HYPERDRONE_ENVIRONMENT_N_CAMERA_AGENTS) || !defined(HYPERLUX_CAMERA_CONFIG_HEADER) || !defined(HYPERDRONE_ENVIRONMENT_DEVICE_CUDA) || !defined(HYPERDRONE_ENVIRONMENT_CONTROL) || !defined(HYPERDRONE_ENVIRONMENT_ACTUATION)
#error "every HYPERDRONE_ENVIRONMENT_* configuration define must be set (hyperdrone.Config.defines())"
#endif
#if HYPERDRONE_ENVIRONMENT_DEVICE_CUDA && !defined(HYPERDRONE_ENVIRONMENT_CUDA)
#error "a CUDA World needs the OptiX variant"
#endif

#include HYPERLUX_CAMERA_CONFIG_HEADER

namespace rlt = rl_tools;
namespace rro = rlt::hyperlux::outputs;

// Each configuration needs a distinct C++ type in nanobind's shared type registry.
namespace HYPERLUX_MODULE_NAME {
    using HOST_DEVICE = rlt::devices::DEVICE_FACTORY<hyperlux::CPUSpecification>;
    using T = float;
    using TI = HOST_DEVICE::index_t;
    constexpr bool CUDA_WORLD = HYPERDRONE_ENVIRONMENT_DEVICE_CUDA == 1;
    static_assert(HYPERDRONE_ENVIRONMENT_DATASET == 0 || HYPERDRONE_ENVIRONMENT_DATASET == 1, "unknown HYPERDRONE_ENVIRONMENT_DATASET");
    using DATASET = rlt::utils::typing::conditional_t<HYPERDRONE_ENVIRONMENT_DATASET == 0,
        rlt::hyperlux::datasets::glb::Dataset, rlt::hyperlux::datasets::procthor::GLB>;

    namespace l2f = rlt::rl::environments::l2f;
#ifdef HYPERDRONE_ENVIRONMENT_SPEC_HEADER
    using WORLD = ::hyperdrone_user::WORLD;
#else
#if HYPERDRONE_ENVIRONMENT_PRESET == 0
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;
    struct DYNAMICS_STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI EPISODE_STEP_LIMIT = HYPERLUX_MODULE_NAME::EPISODE_STEP_LIMIT;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
        using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = l2f::observation::Position<l2f::observation::PositionSpecificationPrivileged<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecificationPrivileged<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecificationPrivileged<T, TI>>>>>>>>;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr auto dynamics = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)0.01};
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, REWARD_FUNCTION{}, {}, {}, {}};
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
        static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{dynamics, integration, mdp}, disturbances};
        static constexpr T STATE_LIMIT_POSITION_X = 100000;
        static constexpr T STATE_LIMIT_POSITION_Y = 100000;
        static constexpr T STATE_LIMIT_POSITION_Z = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
    };
    using PRESET_SPEC = rlt::rl::environments::hyperdrone::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS>;
#elif HYPERDRONE_ENVIRONMENT_PRESET == 1
    using PRESET_SPEC = rlt::rl::environments::hyperdrone::presets::X500FPV<T, TI>;
#elif HYPERDRONE_ENVIRONMENT_PRESET == 2
    using PRESET_SPEC = rlt::rl::environments::hyperdrone::presets::X500FPVIMU<T, TI>;
#else
#error "unknown HYPERDRONE_ENVIRONMENT_PRESET"
#endif
    struct WORLD_SPEC: PRESET_SPEC {
        using DYNAMICS_STATIC_PARAMETERS = typename hyperdrone::ActuationConfig<HYPERDRONE_ENVIRONMENT_ACTUATION>::template Apply<typename PRESET_SPEC::DYNAMICS_STATIC_PARAMETERS>;
        using CONTROLLER = typename hyperdrone::ControlConfig<HYPERDRONE_ENVIRONMENT_CONTROL>::TYPE;
        using CAMERA_OUTPUTS = hyperlux_camera_config::CAMERA_OUTPUTS;
        using OBSERVATION = hyperlux_camera_config::OBSERVATION;
        static constexpr TI INSTANCES_PER_WORLD = HYPERDRONE_ENVIRONMENT_INSTANCES;
        static constexpr TI N_AGENTS = HYPERDRONE_ENVIRONMENT_N_AGENTS;
        static constexpr TI N_CAMERA_AGENTS = HYPERDRONE_ENVIRONMENT_N_CAMERA_AGENTS;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE =
            (HYPERDRONE_ENVIRONMENT_TASK == 2 && PRESET_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE == 0 ? 8 : PRESET_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE)
            + (N_AGENTS - 1) * rlt::rl::environments::hyperdrone::MAX_DRONE_PARTS;
        // the visual-inertial localization benchmark flies without self-occlusion, like its
        // C++ harness (src/rl/environments/hyperdrone/tasks/visual_inertial_localization_harness.h)
        static constexpr bool SELF_VISIBLE = HYPERDRONE_ENVIRONMENT_TASK != 3 && PRESET_SPEC::SELF_VISIBLE;
        static constexpr TI CAM_WIDTH = HYPERDRONE_ENVIRONMENT_CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = HYPERDRONE_ENVIRONMENT_CAM_HEIGHT;
        static constexpr bool ENABLE_MOTION_BLUR = HYPERDRONE_ENVIRONMENT_MB_SAMPLES > 1;
        static constexpr TI MOTION_BLUR_SAMPLES = HYPERDRONE_ENVIRONMENT_MB_SAMPLES;
        static constexpr TI MAX_DRONES = 8;
        using SHADING = rlt::utils::typing::conditional_t<HYPERDRONE_ENVIRONMENT_SHADING == 0, rlt::hyperlux::Low,
                        rlt::utils::typing::conditional_t<HYPERDRONE_ENVIRONMENT_SHADING == 1, rlt::hyperlux::Medium,
                                                                                      rlt::hyperlux::High>>;
    };
    using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
#if HYPERDRONE_ENVIRONMENT_TASK == 0
    using WORLD = BASE_WORLD;
#elif HYPERDRONE_ENVIRONMENT_TASK == 1
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD> {};
    using WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
#elif HYPERDRONE_ENVIRONMENT_TASK == 2
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::moving_gate::Specification<BASE_WORLD> {};
    using WORLD = rlt::rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
#elif HYPERDRONE_ENVIRONMENT_TASK == 3
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::visual_inertial_localization::Specification<BASE_WORLD> {};
    using WORLD = rlt::rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
#else
#error "unknown HYPERDRONE_ENVIRONMENT_TASK"
#endif
#endif

    using OUTPUTS = typename WORLD::RENDERER_SPEC::OUTPUTS;
    using INSTANCE_SEGMENTATION = rro::Segmentation<rro::format::UInt32, rro::labels::Instance>;
    using SEMANTIC_SEGMENTATION = rro::Segmentation<rro::format::UInt32, rro::labels::Semantic>;
    constexpr TI NUM_WORLDS = HYPERDRONE_ENVIRONMENT_NUM_WORLDS;
    using ENV = rlt::rl::environments::HyperDrone<WORLD, NUM_WORLDS>;
    constexpr TI TOTAL = NUM_WORLDS * WORLD::INSTANCES;
    using DYNAMICS = typename WORLD::DYNAMICS_ENV::Parameters::Dynamics;

    // optional World surface beyond the base verbs, detected by type: an IMU observation
    // stream and a camera frame stride (the visual_inertial_localization task defines both)
    template <typename WORLD_TYPE, typename = void>
    struct ObservationIMUDim { static constexpr TI VALUE = 0; };
    template <typename WORLD_TYPE>
    struct ObservationIMUDim<WORLD_TYPE, rlt::utils::typing::void_t<typename WORLD_TYPE::ObservationIMU>> { static constexpr TI VALUE = WORLD_TYPE::ObservationIMU::DIM; };
    template <typename WORLD_TYPE, typename = void>
    struct FrameStride { static constexpr TI VALUE = 1; };
    template <typename WORLD_TYPE>
    struct FrameStride<WORLD_TYPE, rlt::utils::typing::void_t<decltype(WORLD_TYPE::FRAME_STRIDE)>> { static constexpr TI VALUE = WORLD_TYPE::FRAME_STRIDE; };
    constexpr TI OBSERVATION_DIM_IMU = ObservationIMUDim<WORLD>::VALUE;

    template <typename WORLD_TYPE, typename = void>
    struct SynchronizedResets { static constexpr bool VALUE = false; };
    template <typename WORLD_TYPE>
    struct SynchronizedResets<WORLD_TYPE, rlt::utils::typing::void_t<decltype(WORLD_TYPE::SYNCHRONIZED_RESETS)>> { static constexpr bool VALUE = WORLD_TYPE::SYNCHRONIZED_RESETS; };

    // The environment behind the binding. The World, its RNG and every instance tensor live in the
    // World's memory (GPU under CUDA_WORLD, impl.cpp is then compiled by nvcc) and every verb is
    // enqueued on stream(). A `resident` array lives in the World's memory and is read or written
    // in place on that stream; any other array is a host array, staged through the environment
    // with a synchronization. A null reset mask selects every instance.
    struct OutputBlock {
        void* pointer = nullptr;
        hyperlux::OutputLayout layout;
        int device_type = 1;
        int device_id = 0;
        std::shared_ptr<void> owner;
        std::function<void(uintptr_t)> order_consumer;
    };

    struct Environment {
        struct Impl;
#if defined(RL_TOOLS_HYPERLUX_BACKEND_OPTIX)
        hyperlux::DeviceLeases leases;
#endif
        std::shared_ptr<Impl> impl;
        Environment();
        ~Environment();
        Environment(const Environment&) = delete;
        Environment& operator=(const Environment&) = delete;
        // drones: the shared table, one asset (SELF_VISIBLE) and one resolved parameter set per entry
        void init(const std::vector<std::string>& scenes, const std::vector<std::optional<std::string>>& drone_assets, const std::vector<std::string>& drone_parameters, const std::map<std::string, std::string>& assets, unsigned long long seed);
        std::string segmentation_taxonomy() const;
        std::map<uint32_t, std::string> segmentation_classes() const;
        std::string parameters_json();
        std::string drones_json();
        std::string instance_parameters_json();
        // per instance: the active scene's content hash, the drone entry, the scene placement,
        // the agent poses (xyz, wxyz) and the shutter fraction, which reset replays
        std::string spawn_points_json();
        void set_instance_parameters(const bool* mask, const std::string& json);
        std::optional<std::string> model(const std::string& name);
        void rotate_scene();

        // spawn_points (one per selected instance, see spawn_points_json, a host mask): the
        // entry's nominal drone at the recorded placement and poses; empty samples as usual
        void reset(const bool* mask, bool resident, const std::string& spawn_points = "");
        void step(const float* actions, bool resident);
        void observe(float* out, bool resident);
        void render();
        void probe();
        const std::vector<hyperlux::OutputMetadata>& output_metadata() const;
        void output_read(size_t index, void* destination, int device_type, const std::string& field, uintptr_t stream);
        OutputBlock output_block(size_t index, size_t block, const std::string& field);

        void observe_privileged(float* out, bool resident);
        void observe_imu(float* out, bool resident);
        void rewards(float* out, bool resident);
        void terminated(bool* out, bool resident);
        uintptr_t stream() const;
        void set_stream(uintptr_t stream);
        int device_id() const;
        void synchronize();
        void spy_open();
        bool spy_update();
        void spy_close();
    };

    struct ObservationBlock {
        std::string name;
        TI offset;
        TI size;
    };
    struct ObservationLayout {
        std::vector<TI> shape;
        std::string axis;
        std::vector<ObservationBlock> blocks;
    };

    inline ObservationLayout observation_layout(){
        if constexpr (!WORLD::HAS_IMAGE_OBSERVATION){
            return {{0}, "flat", {}};
        }
        constexpr TI HEIGHT = WORLD::Observation::HEIGHT;
        constexpr TI WIDTH = WORLD::Observation::WIDTH;
        constexpr TI CHANNELS = WORLD::Observation::CHANNELS;
        ObservationLayout layout{{HEIGHT, WIDTH, CHANNELS}, "channel", {}};
#if HYPERDRONE_ENVIRONMENT_TASK == 1
        constexpr TI STACK_CHANNELS = TASK_SPEC::IMAGE_STACK_N * BASE_WORLD::IMAGE_CHANNELS;
        layout.blocks.push_back({"image_stack", 0, STACK_CHANNELS});
        layout.blocks.push_back({"target_image", STACK_CHANNELS, BASE_WORLD::IMAGE_CHANNELS});
        if constexpr (CHANNELS > STACK_CHANNELS + BASE_WORLD::IMAGE_CHANNELS){
            constexpr TI USED = STACK_CHANNELS + BASE_WORLD::IMAGE_CHANNELS;
            layout.blocks.push_back({"pad", USED, CHANNELS - USED});
        }
#else
        layout.blocks.push_back({"image", 0, CHANNELS});
#endif
        return layout;
    }

    inline ObservationLayout observation_layout_privileged(){
        ObservationLayout layout{{WORLD::OBSERVATION_DIM_PRIVILEGED}, "flat", {}};
#ifdef HYPERDRONE_ENVIRONMENT_SPEC_HEADER
        layout.blocks.push_back({"observation", 0, WORLD::OBSERVATION_DIM_PRIVILEGED});
#else
        // the dynamics observation chain: Position(3), OrientationRotationMatrix(9),
        // LinearVelocity(3), AngularVelocity(3) — per agent, agents contiguous
        constexpr TI PER_AGENT = 18;
        for(TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++){
            std::string prefix = WORLD::N_AGENTS > 1 ? ("agent" + std::to_string(agent_i) + "/") : "";
            TI base = agent_i * PER_AGENT;
            layout.blocks.push_back({prefix + "position", base, 3});
            layout.blocks.push_back({prefix + "orientation_rotation_matrix", base + 3, 9});
            layout.blocks.push_back({prefix + "linear_velocity", base + 12, 3});
            layout.blocks.push_back({prefix + "angular_velocity", base + 15, 3});
        }
#if HYPERDRONE_ENVIRONMENT_TASK == 2
        layout.blocks.push_back({"gate_state", WORLD::N_AGENTS * PER_AGENT, 8});
#endif
#if HYPERDRONE_ENVIRONMENT_TASK == 3
        layout.blocks.push_back({"waypoint", WORLD::N_AGENTS * PER_AGENT, 3});
#endif
#endif
        return layout;
    }

    inline ObservationLayout observation_layout_imu(){
#if HYPERDRONE_ENVIRONMENT_TASK == 3
        return {{OBSERVATION_DIM_IMU}, "flat", {{"accelerometer", 0, 3}, {"gyroscope", 3, 3}, {"frame_age", 6, 1}, {"new_frame", 7, 1}}};
#else
        return {{OBSERVATION_DIM_IMU}, "flat", {{"imu", 0, OBSERVATION_DIM_IMU}}};
#endif
    }
}
