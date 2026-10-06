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
#include <string>
#include <vector>

#ifndef HYPERDRONE_ENV_NUM_ENVIRONMENTS
#define HYPERDRONE_ENV_NUM_ENVIRONMENTS 1
#endif
#ifndef HYPERDRONE_ENV_INSTANCES
#error "HYPERDRONE_ENV_INSTANCES must be defined"
#endif
#ifndef HYPERDRONE_ENV_CAM_WIDTH
#define HYPERDRONE_ENV_CAM_WIDTH 32
#endif
#ifndef HYPERDRONE_ENV_CAM_HEIGHT
#define HYPERDRONE_ENV_CAM_HEIGHT 32
#endif
#ifndef HYPERDRONE_ENV_SHADING
#define HYPERDRONE_ENV_SHADING 0
#endif
#ifndef HYPERDRONE_ENV_HISTORY_LENGTH
#define HYPERDRONE_ENV_HISTORY_LENGTH 1
#endif
#ifndef HYPERDRONE_ENV_PRESET
#define HYPERDRONE_ENV_PRESET 0
#endif
#ifndef HYPERDRONE_ENV_TASK
#define HYPERDRONE_ENV_TASK 0
#endif
#ifndef HYPERDRONE_ENV_N_AGENTS
#define HYPERDRONE_ENV_N_AGENTS 1
#endif

#define HYPERDRONE_ENV_STRINGIFY_INNER(x) #x
#define HYPERDRONE_ENV_STRINGIFY(x) HYPERDRONE_ENV_STRINGIFY_INNER(x)

// the escape hatch: a user header defining hyperdrone_env_user::WORLD (a fully built
// World or task chain); preset/task/instance defines are superseded by the user type
#ifdef HYPERDRONE_ENV_SPEC_HEADER
#include HYPERDRONE_ENV_STRINGIFY(HYPERDRONE_ENV_SPEC_HEADER)
#endif

namespace rlt = rl_tools;

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
        static constexpr TI HISTORY_LENGTH = HYPERDRONE_ENV_HISTORY_LENGTH;
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
        void init(const std::vector<std::string>& scenes, const std::string& drone_asset, const std::string& gate_asset, unsigned long long seed);
        std::string parameters_json();
        bool set_parameters(const std::string& json);
        std::string instance_parameters_json();
        bool set_instance_parameters(const uint8_t* mask, const std::string& json);
        std::string model(const std::string& name);
        void reset(const uint8_t* mask);
        void render(const uint8_t* reset_mask);
        void observe(float* observations);
        void observe_privileged(float* observations);
        void observe_imu(float* observations);
        void step(const float* actions);
        void read_rewards(float* destination);
        void read_terminated(uint8_t* destination);
        void rotate_scene();
    };

    // one schema for the nominal and the per-instance parameters: {"dynamics": <l2f parameters>, "camera": {...}}
    template <typename DYNAMICS_PARAMETERS>
    std::string parameters_to_json(Environment& environment, const DYNAMICS_PARAMETERS& dynamics, const rlt::rl::environments::hyperdrone::Camera<T>& camera){
        const auto& world = environment.env.environments[0];
        return "{\"dynamics\": " + rlt::json(environment.device, world.dynamics, dynamics) + ", \"camera\": " + rlt::json(environment.device, world, camera) + "}";
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
    void set_gate_asset_path(WORLD_TYPE& world, const char* path){
        if constexpr (HasGateAssetPath<WORLD_TYPE>::VALUE){
            world.gate_asset_path = path;
        }else{
            (void)world;
            (void)path;
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
        if constexpr (ObservationIMUDim<WORLD_TYPE>::VALUE > 0){
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL, ObservationIMUDim<WORLD_TYPE>::VALUE>>> alias;
            alias._data = observations;
            rlt::observe(environment.device, environment.env, environment.parameters, environment.states, typename WORLD_TYPE::ObservationIMU{}, alias, environment.rng);
        }else{
            (void)environment;
            (void)observations;
        }
    }

    inline std::string flat_blocks_privileged(){
        std::string out;
#ifdef HYPERDRONE_ENV_SPEC_HEADER
        out += "block observation 0 " + std::to_string((unsigned long long)WORLD::OBSERVATION_DIM_PRIVILEGED) + "\n";
#else
        // the dynamics observation chain: Position(3), OrientationRotationMatrix(9),
        // LinearVelocity(3), AngularVelocity(3) — per agent, agents contiguous
        constexpr TI PER_AGENT = 18;
        constexpr TI N_AGENTS = WORLD::N_AGENTS;
        for(TI agent_i = 0; agent_i < N_AGENTS; agent_i++){
            std::string prefix = N_AGENTS > 1 ? ("agent" + std::to_string((unsigned long long)agent_i) + "/") : "";
            TI base = agent_i * PER_AGENT;
            out += "block " + prefix + "position " + std::to_string((unsigned long long)(base + 0)) + " 3\n";
            out += "block " + prefix + "orientation_rotation_matrix " + std::to_string((unsigned long long)(base + 3)) + " 9\n";
            out += "block " + prefix + "linear_velocity " + std::to_string((unsigned long long)(base + 12)) + " 3\n";
            out += "block " + prefix + "angular_velocity " + std::to_string((unsigned long long)(base + 15)) + " 3\n";
        }
#if HYPERDRONE_ENV_TASK == 2
        out += "block gate_state " + std::to_string((unsigned long long)(N_AGENTS * PER_AGENT)) + " 8\n";
#endif
#if HYPERDRONE_ENV_TASK == 3
        out += "block waypoint " + std::to_string((unsigned long long)(N_AGENTS * PER_AGENT)) + " 3\n";
#endif
#endif
        return out;
    }

    inline std::string observation_layout_imu(){
        if(OBSERVATION_DIM_IMU == 0){
            return "";
        }
        std::string out = "shape " + std::to_string((unsigned long long)OBSERVATION_DIM_IMU) + "\naxis flat\n";
#if HYPERDRONE_ENV_TASK == 3
        out += "block accelerometer 0 3\nblock gyroscope 3 3\nblock frame_age 6 1\nblock new_frame 7 1\n";
#else
        out += "block imu 0 " + std::to_string((unsigned long long)OBSERVATION_DIM_IMU) + "\n";
#endif
        return out;
    }

    inline std::string observation_layout_value(bool privileged){
        std::string out;
        if(privileged){
            out += "shape " + std::to_string((unsigned long long)WORLD::OBSERVATION_DIM_PRIVILEGED) + "\n";
            out += "axis flat\n";
            out += flat_blocks_privileged();
        }else{
#ifdef HYPERDRONE_ENV_SPEC_HEADER
            out += "shape " + std::to_string((unsigned long long)WORLD::OBSERVATION_DIM) + "\n";
            out += "axis flat\n";
            out += "block observation 0 " + std::to_string((unsigned long long)WORLD::OBSERVATION_DIM) + "\n";
#else
            constexpr TI HEIGHT = WORLD::Observation::HEIGHT;
            constexpr TI WIDTH = WORLD::Observation::WIDTH;
            constexpr TI CHANNELS = WORLD::Observation::CHANNELS;
            out += "shape " + std::to_string((unsigned long long)HEIGHT) + " " + std::to_string((unsigned long long)WIDTH) + " " + std::to_string((unsigned long long)CHANNELS) + "\n";
            out += "axis channel\n";
#if HYPERDRONE_ENV_TASK == 1
            constexpr TI STACK_CHANNELS = TASK_SPEC::IMAGE_STACK_N * BASE_WORLD::IMAGE_CHANNELS;
            out += "block image_stack 0 " + std::to_string((unsigned long long)STACK_CHANNELS) + "\n";
            out += "block target_image " + std::to_string((unsigned long long)STACK_CHANNELS) + " " + std::to_string((unsigned long long)BASE_WORLD::IMAGE_CHANNELS) + "\n";
            if constexpr (CHANNELS > STACK_CHANNELS + BASE_WORLD::IMAGE_CHANNELS){
                constexpr TI USED = STACK_CHANNELS + BASE_WORLD::IMAGE_CHANNELS;
                out += "block pad " + std::to_string((unsigned long long)USED) + " " + std::to_string((unsigned long long)(CHANNELS - USED)) + "\n";
            }
#else
            out += "block image 0 " + std::to_string((unsigned long long)CHANNELS) + "\n";
#endif
#endif
        }
        return out;
    }

    inline std::string config_string(){
        static const std::string value =
            std::string("num_environments=") + std::to_string(NUM_ENVIRONMENTS)
            + " instances=" + std::to_string((unsigned long long)WORLD::INSTANCES)
            + " cam=" + std::to_string((unsigned long long)WORLD::SPEC::CAM_WIDTH) + "x" + std::to_string((unsigned long long)WORLD::SPEC::CAM_HEIGHT)
            + " shading=" + std::to_string(HYPERDRONE_ENV_SHADING)
            + " history=" + std::to_string((unsigned long long)WORLD::SPEC::HISTORY_LENGTH)
            + " preset=" + std::to_string(HYPERDRONE_ENV_PRESET)
            + " task=" + std::to_string(HYPERDRONE_ENV_TASK)
            + " n_agents=" + std::to_string((unsigned long long)WORLD::N_AGENTS)
#ifdef HYPERDRONE_ENV_SPEC_HEADER
            + " spec_header=1"
#endif
            ;
        return value;
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
    inline void Environment::init(const std::vector<std::string>& references, const std::string& drone_asset_path, const std::string& gate_asset_path, unsigned long long seed){
        rlt::init(device, rng, seed);
        if(!drone_asset_path.empty()){
            for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
                env.environments[environment_i].drone_asset_path = drone_asset_path;
            }
        }
        if(!gate_asset_path.empty()){
            for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
                set_gate_asset_path(env.environments[environment_i], gate_asset_path.c_str());
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
    inline bool Environment::set_parameters(const std::string& parameters_json){
        auto dynamics = env.environments[0].dynamics.parameters;
        auto camera = env.environments[0].parameters.camera;
        try{
            parameters_from_json(*this, nlohmann::json::parse(parameters_json), dynamics, camera);
        }
        catch(const nlohmann::json::exception&){
            return false;
        }
        for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
            env.environments[environment_i].dynamics.parameters = dynamics;
            env.environments[environment_i].parameters.camera = camera;
        }
        return true;
    }
    inline std::string Environment::instance_parameters_json(){
        std::string value = "[";
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            const auto& instance = rlt::get_ref(device, parameters, instance_i);
            value += (instance_i > 0 ? ", " : "") + parameters_to_json(*this, instance.dynamics, instance.camera);
        }
        value += "]";
        return value;
    }
    inline bool Environment::set_instance_parameters(const uint8_t* mask, const std::string& parameters_json_array){
        std::vector<typename WORLD::Parameters> updated;
        try{
            const nlohmann::json entries = nlohmann::json::parse(parameters_json_array);
            for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
                if(mask[instance_i]){
                    updated.push_back(rlt::get_ref(device, parameters, instance_i));
                    parameters_from_json(*this, entries.at(updated.size() - 1), updated.back().dynamics, updated.back().camera);
                }
            }
            if(updated.size() != entries.size()){
                return false;
            }
        }
        catch(const nlohmann::json::exception&){
            return false;
        }
        TI entry_i = 0;
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            if(mask[instance_i]){
                rlt::get_ref(device, parameters, instance_i) = updated[entry_i++];
            }
        }
        return true;
    }
    inline std::string Environment::model(const std::string& name){
        DYNAMICS dynamics;
        if(!hyperdrone::model_by_name<typename WORLD::SPEC>(name, dynamics)){
            return {};
        }
        return rlt::json(device, env.environments[0].dynamics, dynamics);
    }
    inline void Environment::reset(const uint8_t* mask){
        static_assert(sizeof(bool) == sizeof(uint8_t));
        std::memcpy(rlt::data(reset_mask), mask, TOTAL * sizeof(bool));
        rlt::sample_initial_parameters(device, env, parameters, reset_mask, rng);
        rlt::sample_initial_state(device, env, parameters, states, reset_mask, rng);
    }
    inline void Environment::render(const uint8_t* mask){
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
    inline void Environment::observe_imu(float* observations){
        HYPERDRONE_MODULE_NAME::observe_imu<WORLD>(*this, observations);
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
    inline void Environment::read_terminated(uint8_t* destination){
        std::memcpy(destination, rlt::data(terminated_flags), TOTAL * sizeof(bool));
    }
    inline void Environment::rotate_scene(){
        rlt::rotate_scene(device, env);
    }
}
