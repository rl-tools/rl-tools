#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#if defined(HYPERDRONE_ENV_TASK) && HYPERDRONE_ENV_TASK == 1
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#endif
#if defined(HYPERDRONE_ENV_TASK) && HYPERDRONE_ENV_TASK == 2
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cpu.h>
#endif
#if defined(HYPERDRONE_ENV_PRESET) && HYPERDRONE_ENV_PRESET == 1
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#endif

#include "iface.h"

#include <cstring>
#include <string>

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

namespace hyperdrone_env_impl {
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
        static constexpr TI EPISODE_STEP_LIMIT = hyperdrone_env_impl::EPISODE_STEP_LIMIT;
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
#else
#error "unknown HYPERDRONE_ENV_PRESET"
#endif
    struct WORLD_SPEC: PRESET_SPEC {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = HYPERDRONE_ENV_INSTANCES;
        static constexpr TI N_AGENTS = HYPERDRONE_ENV_N_AGENTS;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE =
            HYPERDRONE_ENV_TASK == 2 && PRESET_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE == 0 ? 8 : PRESET_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE;
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
#else
#error "unknown HYPERDRONE_ENV_TASK"
#endif
#endif

    constexpr TI NUM_ENVIRONMENTS = HYPERDRONE_ENV_NUM_ENVIRONMENTS;
    using ENV = rlt::rl::environments::hyperdrone::MultiEnvironment<WORLD, NUM_ENVIRONMENTS>;
    constexpr TI TOTAL = NUM_ENVIRONMENTS * WORLD::INSTANCES;

    struct EnvImpl {
        DEVICE device;
        ENV env;
        RNG rng;
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, TOTAL>>> parameters;
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, TOTAL>>> states, next_states;
        rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, TOTAL>>> reset_mask, terminated_flags;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL>>> rewards;
    };

    EnvImpl* cast(void* handle){
        return reinterpret_cast<EnvImpl*>(handle);
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

    std::string flat_blocks_privileged(){
        std::string out;
#ifdef HYPERDRONE_ENV_SPEC_HEADER
        out += "block observation 0 " + std::to_string((unsigned long long)WORLD::OBSERVATION_DIM_PRIVILEGED) + "\n";
#else
        // the shim's dynamics observation chain: Position(3), OrientationRotationMatrix(9),
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
#endif
        return out;
    }

    std::string observation_layout_value(bool privileged){
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
}

using namespace hyperdrone_env_impl;

extern "C" {
    int hyperdrone_env_iface_version(){
        return HYPERDRONE_ENV_IFACE_VERSION;
    }
    const char* hyperdrone_env_config_string(){
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
        return value.c_str();
    }
    const char* hyperdrone_env_observation_layout(int privileged){
        static const std::string observation = observation_layout_value(false);
        static const std::string observation_privileged = observation_layout_value(true);
        return privileged ? observation_privileged.c_str() : observation.c_str();
    }
    void* hyperdrone_env_create(){
        EnvImpl* impl = new EnvImpl();
        rlt::init(impl->device);
        rlt::malloc(impl->device, impl->env);
        rlt::malloc(impl->device, impl->rng);
        rlt::init(impl->device, impl->rng, 0);
        rlt::malloc(impl->device, impl->parameters);
        rlt::malloc(impl->device, impl->states);
        rlt::malloc(impl->device, impl->next_states);
        rlt::malloc(impl->device, impl->reset_mask);
        rlt::malloc(impl->device, impl->terminated_flags);
        rlt::malloc(impl->device, impl->rewards);
        return impl;
    }
    void hyperdrone_env_destroy(void* handle){
        EnvImpl* impl = cast(handle);
        rlt::free(impl->device, impl->parameters);
        rlt::free(impl->device, impl->states);
        rlt::free(impl->device, impl->next_states);
        rlt::free(impl->device, impl->reset_mask);
        rlt::free(impl->device, impl->terminated_flags);
        rlt::free(impl->device, impl->rewards);
        rlt::free(impl->device, impl->rng);
        rlt::free(impl->device, impl->env);
        delete impl;
    }
    void hyperdrone_env_config(void* handle, hyperdrone::env::Config* config){
        (void)handle;
        config->num_environments = (uint32_t)NUM_ENVIRONMENTS;
        config->instances_per_environment = (uint32_t)WORLD::INSTANCES;
        config->total_instances = (uint32_t)TOTAL;
        config->n_agents = (uint32_t)WORLD::N_AGENTS;
        config->cam_width = (uint32_t)WORLD::SPEC::CAM_WIDTH;
        config->cam_height = (uint32_t)WORLD::SPEC::CAM_HEIGHT;
        config->image_channels = (uint32_t)WORLD::IMAGE_CHANNELS;
        config->observation_dim = (uint32_t)WORLD::OBSERVATION_DIM;
        config->observation_dim_privileged = (uint32_t)WORLD::OBSERVATION_DIM_PRIVILEGED;
        config->action_dim = (uint32_t)WORLD::ACTION_DIM;
        config->episode_step_limit = (uint32_t)WORLD::EPISODE_STEP_LIMIT;
    }
    void hyperdrone_env_init(void* handle, const char* scene_directory, const char* drone_asset_path, const char* gate_asset_path, unsigned long long seed){
        EnvImpl* impl = cast(handle);
        rlt::init(impl->device, impl->rng, seed);
        if(drone_asset_path != nullptr && drone_asset_path[0] != '\0'){
            for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
                impl->env.environments[environment_i].drone_asset_path = drone_asset_path;
            }
        }
        if(gate_asset_path != nullptr && gate_asset_path[0] != '\0'){
            for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
                set_gate_asset_path(impl->env.environments[environment_i], gate_asset_path);
            }
        }
        rlt::rendering::datasets::procthor::GLB dataset{scene_directory, {}};
        rlt::init(impl->device, impl->env, dataset);
    }
    void hyperdrone_env_reset(void* handle, const uint8_t* mask){
        EnvImpl* impl = cast(handle);
        static_assert(sizeof(bool) == sizeof(uint8_t));
        std::memcpy(rlt::data(impl->reset_mask), mask, TOTAL * sizeof(bool));
        rlt::sample_initial_parameters(impl->device, impl->env, impl->parameters, impl->reset_mask, impl->rng);
        rlt::sample_initial_state(impl->device, impl->env, impl->parameters, impl->states, impl->reset_mask, impl->rng);
    }
    void hyperdrone_env_render(void* handle, const uint8_t* reset_mask){
        EnvImpl* impl = cast(handle);
        std::memcpy(rlt::data(impl->reset_mask), reset_mask, TOTAL * sizeof(bool));
        rlt::render(impl->device, impl->env, impl->parameters, impl->states, impl->reset_mask);
    }
    void hyperdrone_env_observe(void* handle, float* observations){
        EnvImpl* impl = cast(handle);
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL, WORLD::OBSERVATION_DIM>>> alias;
        alias._data = observations;
        rlt::observe(impl->device, impl->env, impl->parameters, impl->states, typename WORLD::Observation{}, alias, impl->rng);
    }
    void hyperdrone_env_observe_privileged(void* handle, float* observations){
        EnvImpl* impl = cast(handle);
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL, WORLD::OBSERVATION_DIM_PRIVILEGED>>> alias;
        alias._data = observations;
        rlt::observe(impl->device, impl->env, impl->parameters, impl->states, typename WORLD::ObservationPrivileged{}, alias, impl->rng);
    }
    void hyperdrone_env_step(void* handle, const float* actions){
        EnvImpl* impl = cast(handle);
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL, WORLD::ACTION_DIM>>> action_alias;
        action_alias._data = const_cast<float*>(actions);
        rlt::step(impl->device, impl->env, impl->parameters, impl->states, action_alias, impl->next_states, impl->rng);
        rlt::reward(impl->device, impl->env, impl->parameters, impl->states, action_alias, impl->next_states, impl->rewards, impl->rng);
        rlt::copy(impl->device, impl->device, impl->next_states, impl->states);
        rlt::terminated(impl->device, impl->env, impl->parameters, impl->states, impl->terminated_flags, impl->rng);
    }
    void hyperdrone_env_rewards(void* handle, float* rewards){
        EnvImpl* impl = cast(handle);
        std::memcpy(rewards, rlt::data(impl->rewards), TOTAL * sizeof(float));
    }
    void hyperdrone_env_terminated(void* handle, uint8_t* terminated_flags){
        EnvImpl* impl = cast(handle);
        std::memcpy(terminated_flags, rlt::data(impl->terminated_flags), TOTAL * sizeof(bool));
    }
    void hyperdrone_env_rotate_scene(void* handle){
        EnvImpl* impl = cast(handle);
        rlt::rotate_scene(impl->device, impl->env);
    }
}
