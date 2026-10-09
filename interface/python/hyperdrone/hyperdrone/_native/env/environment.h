#pragma once
#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#if defined(HYPERDRONE_ENV_TASK) && HYPERDRONE_ENV_TASK == 1
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#endif
#if defined(HYPERDRONE_ENV_TASK) && HYPERDRONE_ENV_TASK == 2
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cpu.h>
#endif
#if defined(HYPERDRONE_ENV_TASK) && HYPERDRONE_ENV_TASK == 3
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/operations_cpu.h>
#endif
#if defined(HYPERDRONE_ENV_PRESET) && HYPERDRONE_ENV_PRESET >= 1
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#endif

#include <rl_tools/rl/environments/l2f/operations_cpu.h>

#include "../common/models.h"

#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#if !defined(HYPERDRONE_ENV_NUM_ENVIRONMENTS) || !defined(HYPERDRONE_ENV_INSTANCES) || !defined(HYPERDRONE_ENV_CAM_WIDTH) || !defined(HYPERDRONE_ENV_CAM_HEIGHT) || !defined(HYPERDRONE_ENV_SHADING) || !defined(HYPERDRONE_ENV_MB_SAMPLES) || !defined(HYPERDRONE_ENV_HISTORY_LENGTH) || !defined(HYPERDRONE_ENV_PRESET) || !defined(HYPERDRONE_ENV_TASK) || !defined(HYPERDRONE_ENV_N_AGENTS) || !defined(HYPERDRONE_ENV_MAX_DRONES)
#error "every HYPERDRONE_ENV_* configuration define must be set (hyperdrone.env.EnvConfig.defines())"
#endif

#define HYPERDRONE_ENV_STRINGIFY_INNER(x) #x
#define HYPERDRONE_ENV_STRINGIFY(x) HYPERDRONE_ENV_STRINGIFY_INNER(x)

// the escape hatch: a user header defining hyperdrone_env_user::WORLD (a fully built
// World or task chain); preset/task/instance defines are superseded by the user type
#ifdef HYPERDRONE_ENV_SPEC_HEADER
#include HYPERDRONE_ENV_STRINGIFY(HYPERDRONE_ENV_SPEC_HEADER)
#endif

namespace rlt = rl_tools;

// Each configuration needs a distinct C++ type in nanobind's shared type registry.
namespace HYPERDRONE_MODULE_NAME {
    using DEVICE = rlt::devices::CPU<rlt::devices::DefaultCPUSpecification>;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    using T = float;
    using TI = DEVICE::index_t;

    namespace l2f = rlt::rl::environments::l2f;
#ifdef HYPERDRONE_ENV_SPEC_HEADER
    using WORLD = ::hyperdrone_env_user::WORLD;
#else
#if HYPERDRONE_ENV_PRESET == 0
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;
    struct DYNAMICS_STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI EPISODE_STEP_LIMIT = HYPERDRONE_MODULE_NAME::EPISODE_STEP_LIMIT;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
        using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
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
#elif HYPERDRONE_ENV_PRESET == 1
    using PRESET_SPEC = rlt::rl::environments::hyperdrone::presets::X500FPV<T, TI>;
#elif HYPERDRONE_ENV_PRESET == 2
    using PRESET_SPEC = rlt::rl::environments::hyperdrone::presets::X500FPVIMU<T, TI>;
#else
#error "unknown HYPERDRONE_ENV_PRESET"
#endif
    struct WORLD_SPEC: PRESET_SPEC {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = HYPERDRONE_ENV_INSTANCES;
        static constexpr TI N_AGENTS = HYPERDRONE_ENV_N_AGENTS;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE =
            HYPERDRONE_ENV_TASK == 2 && PRESET_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE == 0 ? 8 : PRESET_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE;
        // the visual-inertial localization benchmark flies without self-occlusion, like its
        // C++ harness (src/rl/environments/hyperdrone/tasks/visual_inertial_localization_harness.h)
        static constexpr bool SELF_VISIBLE = HYPERDRONE_ENV_TASK != 3 && PRESET_SPEC::SELF_VISIBLE;
        static constexpr TI CAM_WIDTH = HYPERDRONE_ENV_CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = HYPERDRONE_ENV_CAM_HEIGHT;
        static constexpr bool ENABLE_MOTION_BLUR = HYPERDRONE_ENV_MB_SAMPLES > 1;
        static constexpr TI MOTION_BLUR_SAMPLES = HYPERDRONE_ENV_MB_SAMPLES;
        static constexpr TI HISTORY_LENGTH = HYPERDRONE_ENV_HISTORY_LENGTH;
        static constexpr TI MAX_DRONES = HYPERDRONE_ENV_MAX_DRONES;
        using SHADING = rlt::utils::typing::conditional_t<HYPERDRONE_ENV_SHADING == 0, rlt::rendering::raytracing::Low,
                        rlt::utils::typing::conditional_t<HYPERDRONE_ENV_SHADING == 1, rlt::rendering::raytracing::Medium,
                                                                                      rlt::rendering::raytracing::High>>;
    };
    using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
#if HYPERDRONE_ENV_TASK == 0
    using WORLD = BASE_WORLD;
#elif HYPERDRONE_ENV_TASK == 1
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD> {};
    using WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
#elif HYPERDRONE_ENV_TASK == 2
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::moving_gate::Specification<BASE_WORLD> {};
    using WORLD = rlt::rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
#elif HYPERDRONE_ENV_TASK == 3
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::visual_inertial_localization::Specification<BASE_WORLD> {};
    using WORLD = rlt::rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
#else
#error "unknown HYPERDRONE_ENV_TASK"
#endif
#endif

    constexpr TI NUM_ENVIRONMENTS = HYPERDRONE_ENV_NUM_ENVIRONMENTS;
    using ENV = rlt::rl::environments::hyperdrone::MultiEnvironment<WORLD, NUM_ENVIRONMENTS>;
    constexpr TI TOTAL = NUM_ENVIRONMENTS * WORLD::INSTANCES;
    using DYNAMICS = typename WORLD::DYNAMICS_ENV::Parameters::Dynamics;

    struct Environment {
        DEVICE device{};
        ENV env{};
        RNG rng{};
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, TOTAL>>> parameters{};
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, TOTAL>>> states{}, next_states{};
        rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, TOTAL>>> reset_mask{}, terminated_flags{};
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL>>> rewards{};
        Environment();
        ~Environment();
        Environment(const Environment&) = delete;
        Environment& operator=(const Environment&) = delete;
        // drones: the shared table, one asset (SELF_VISIBLE) and one resolved parameter set per entry
        void init(const std::vector<std::string>& scenes, const std::vector<std::optional<std::string>>& drone_assets, const std::vector<std::string>& drone_parameters, const std::optional<std::string>& gate_asset, unsigned long long seed);
        std::string parameters_json();
        std::string drones_json();
        std::string instance_parameters_json();
        void set_instance_parameters(const bool* mask, const std::string& json);
        std::optional<std::string> model(const std::string& name);
        void reset(const bool* mask);
        void render(const bool* reset_mask);
        void observe(float* observations);
        void observe_privileged(float* observations);
        void step(const float* actions);
        void read_rewards(float* destination);
        void read_terminated(bool* destination);
        void rotate_scene();
    };

    // one schema for the nominal, the drone table and the per-instance parameters: {"dynamics":
    // <l2f parameters>, "camera": {...}}; instances also carry "drone", their table entry
    template <typename DYNAMICS_PARAMETERS>
    std::string parameters_to_json(Environment& environment, const DYNAMICS_PARAMETERS& dynamics, const rlt::rl::environments::hyperdrone::Camera<T>& camera, std::optional<TI> drone = std::nullopt){
        const auto& world = environment.env.environments[0];
        std::string value = "{\"dynamics\": " + rlt::json(environment.device, world.dynamics, dynamics) + ", \"camera\": " + rlt::json(environment.device, world, camera);
        if(drone){
            value += ", \"drone\": " + std::to_string(*drone);
        }
        return value + "}";
    }
    template <typename DYNAMICS_PARAMETERS>
    void parameters_from_json(Environment& environment, const nlohmann::json& parameters, DYNAMICS_PARAMETERS& dynamics, rlt::rl::environments::hyperdrone::Camera<T>& camera){
        auto& world = environment.env.environments[0];
        rlt::from_json(environment.device, world.dynamics, parameters.at("dynamics"), dynamics);
        rlt::from_json(environment.device, world, parameters.at("camera"), camera);
    }

    template <typename WORLD_TYPE, typename = void>
    struct HasGateAssetPath { static constexpr bool VALUE = false; };
    template <typename WORLD_TYPE>
    struct HasGateAssetPath<WORLD_TYPE, rlt::utils::typing::void_t<decltype(WORLD_TYPE::gate_asset_path)>> { static constexpr bool VALUE = true; };

    template <typename WORLD_TYPE>
    void set_gate_asset_path([[maybe_unused]] WORLD_TYPE& world, [[maybe_unused]] const std::string& path){
        if constexpr (HasGateAssetPath<WORLD_TYPE>::VALUE){
            world.gate_asset_path = path;
        }
    }

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

    template <typename WORLD_TYPE>
    void observe_imu(Environment& environment, float* observations){
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL, ObservationIMUDim<WORLD_TYPE>::VALUE>>> alias;
        alias._data = observations;
        rlt::observe(environment.device, environment.env, environment.parameters, environment.states, typename WORLD_TYPE::ObservationIMU{}, alias, environment.rng);
    }

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
#ifdef HYPERDRONE_ENV_SPEC_HEADER
        return {{WORLD::OBSERVATION_DIM}, "flat", {{"observation", 0, WORLD::OBSERVATION_DIM}}};
#else
        constexpr TI HEIGHT = WORLD::Observation::HEIGHT;
        constexpr TI WIDTH = WORLD::Observation::WIDTH;
        constexpr TI CHANNELS = WORLD::Observation::CHANNELS;
        ObservationLayout layout{{HEIGHT, WIDTH, CHANNELS}, "channel", {}};
#if HYPERDRONE_ENV_TASK == 1
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
#endif
    }

    inline ObservationLayout observation_layout_privileged(){
        ObservationLayout layout{{WORLD::OBSERVATION_DIM_PRIVILEGED}, "flat", {}};
#ifdef HYPERDRONE_ENV_SPEC_HEADER
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
#if HYPERDRONE_ENV_TASK == 2
        layout.blocks.push_back({"gate_state", WORLD::N_AGENTS * PER_AGENT, 8});
#endif
#if HYPERDRONE_ENV_TASK == 3
        layout.blocks.push_back({"waypoint", WORLD::N_AGENTS * PER_AGENT, 3});
#endif
#endif
        return layout;
    }

    inline ObservationLayout observation_layout_imu(){
#if HYPERDRONE_ENV_TASK == 3
        return {{OBSERVATION_DIM_IMU}, "flat", {{"accelerometer", 0, 3}, {"gyroscope", 3, 3}, {"frame_age", 6, 1}, {"new_frame", 7, 1}}};
#else
        return {{OBSERVATION_DIM_IMU}, "flat", {{"imu", 0, OBSERVATION_DIM_IMU}}};
#endif
    }

    inline Environment::Environment(){
        rlt::init(device);
        rlt::malloc(device, env);
        rlt::malloc(device, rng);
        rlt::init(device, rng, 0);
        rlt::malloc(device, parameters);
        rlt::malloc(device, states);
        rlt::malloc(device, next_states);
        rlt::malloc(device, reset_mask);
        rlt::malloc(device, terminated_flags);
        rlt::malloc(device, rewards);
        rlt::set_all(device, terminated_flags, false);
        rlt::set_all(device, rewards, (T)0);
    }
    inline Environment::~Environment(){
        rlt::free(device, parameters);
        rlt::free(device, states);
        rlt::free(device, next_states);
        rlt::free(device, reset_mask);
        rlt::free(device, terminated_flags);
        rlt::free(device, rewards);
        rlt::free(device, rng);
        rlt::free(device, env);
    }
    inline void Environment::init(const std::vector<std::string>& references, const std::vector<std::optional<std::string>>& drone_assets, const std::vector<std::string>& drone_parameters, const std::optional<std::string>& gate_asset_path, unsigned long long seed){
        rlt::init(device, rng, seed);
        if(drone_assets.size() != drone_parameters.size()){
            throw std::invalid_argument("hyperdrone: drone assets and parameters must pair up");
        }
        for(size_t drone_i = 0; drone_i < drone_assets.size(); drone_i++){
            typename ENV::SharedContext::DRONE drone;
            if(drone_assets[drone_i]){
                drone.asset = *drone_assets[drone_i];
            }
            parameters_from_json(*this, nlohmann::json::parse(drone_parameters[drone_i]), drone.parameters.dynamics, drone.parameters.camera);
            env.shared.drones.push_back(drone);
        }
        if(gate_asset_path){
            for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
                set_gate_asset_path(env.environments[environment_i], *gate_asset_path);
            }
        }
        rlt::rendering::datasets::procthor::GLB dataset{};
        if(references.size() == 1 && std::filesystem::is_directory(references[0])){
            dataset.directory = references[0];
        }
        else{
            dataset.references = references;
        }
        rlt::init(device, env, dataset);
    }
    inline std::string Environment::parameters_json(){
        const auto& world = env.environments[0];
        return parameters_to_json(*this, world.dynamics.parameters, world.parameters.camera);
    }
    inline std::string Environment::drones_json(){
        std::string value = "[";
        for(size_t drone_i = 0; drone_i < env.shared.drones.size(); drone_i++){
            const auto& drone = env.shared.drones[drone_i];
            const std::string asset = drone.asset.empty() ? "null" : nlohmann::json(drone.asset).dump();
            value += (drone_i > 0 ? ", " : "") + std::string("{\"asset\": ") + asset + ", \"parameters\": " + parameters_to_json(*this, drone.parameters.dynamics, drone.parameters.camera) + "}";
        }
        return value + "]";
    }
    inline std::string Environment::instance_parameters_json(){
        std::string value = "[";
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            const auto& instance = rlt::get_ref(device, parameters, instance_i);
            value += (instance_i > 0 ? ", " : "") + parameters_to_json(*this, instance.dynamics, instance.camera, instance.drone);
        }
        value += "]";
        return value;
    }
    inline void Environment::set_instance_parameters(const bool* mask, const std::string& parameters_json_array){
        const nlohmann::json entries = nlohmann::json::parse(parameters_json_array);
        TI entry_i = 0;
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            if(mask[instance_i]){
                auto& instance = rlt::get_ref(device, parameters, instance_i);
                parameters_from_json(*this, entries.at(entry_i++), instance.dynamics, instance.camera);
            }
        }
    }
    inline std::optional<std::string> Environment::model(const std::string& name){
        DYNAMICS dynamics;
        if(!hyperdrone::model_by_name<typename WORLD::SPEC>(name, dynamics)){
            return std::nullopt;
        }
        return rlt::json(device, env.environments[0].dynamics, dynamics);
    }
    inline void Environment::reset(const bool* mask){
        std::memcpy(rlt::data(reset_mask), mask, TOTAL * sizeof(bool));
        rlt::sample_initial_parameters(device, env, parameters, reset_mask, rng);
        rlt::sample_initial_state(device, env, parameters, states, reset_mask, rng);
    }
    inline void Environment::render(const bool* mask){
        std::memcpy(rlt::data(reset_mask), mask, TOTAL * sizeof(bool));
        rlt::render(device, env, parameters, states, reset_mask);
    }
    inline void Environment::observe(float* observations){
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL, WORLD::OBSERVATION_DIM>>> alias;
        alias._data = observations;
        rlt::observe(device, env, parameters, states, typename WORLD::Observation{}, alias, rng);
    }
    inline void Environment::observe_privileged(float* observations){
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL, WORLD::OBSERVATION_DIM_PRIVILEGED>>> alias;
        alias._data = observations;
        rlt::observe(device, env, parameters, states, typename WORLD::ObservationPrivileged{}, alias, rng);
    }
    inline void Environment::step(const float* actions){
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL, WORLD::ACTION_DIM>>> action_alias;
        action_alias._data = const_cast<float*>(actions);
        rlt::step(device, env, parameters, states, action_alias, next_states, rng);
        rlt::reward(device, env, parameters, states, action_alias, next_states, rewards, rng);
        rlt::copy(device, device, next_states, states);
        rlt::terminated(device, env, parameters, states, terminated_flags, rng);
    }
    inline void Environment::read_rewards(float* destination){
        std::memcpy(destination, rlt::data(rewards), TOTAL * sizeof(float));
    }
    inline void Environment::read_terminated(bool* destination){
        std::memcpy(destination, rlt::data(terminated_flags), TOTAL * sizeof(bool));
    }
    inline void Environment::rotate_scene(){
        rlt::rotate_scene(device, env);
    }
}
