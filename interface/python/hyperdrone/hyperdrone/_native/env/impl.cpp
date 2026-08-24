#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>

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

namespace rlt = rl_tools;

namespace hyperdrone_env_impl {
    using DEVICE = rlt::devices::CPU<rlt::devices::DefaultCPUSpecification>;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    using T = float;
    using TI = DEVICE::index_t;

    namespace l2f = rlt::rl::environments::l2f;
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

    struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS> {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = HYPERDRONE_ENV_INSTANCES;
        static constexpr TI CAM_WIDTH = HYPERDRONE_ENV_CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = HYPERDRONE_ENV_CAM_HEIGHT;
        static constexpr TI HISTORY_LENGTH = HYPERDRONE_ENV_HISTORY_LENGTH;
        using SHADING = rlt::utils::typing::conditional_t<HYPERDRONE_ENV_SHADING == 0, rlt::rendering::raytracing::Low,
                        rlt::utils::typing::conditional_t<HYPERDRONE_ENV_SHADING == 1, rlt::rendering::raytracing::Medium,
                                                                                      rlt::rendering::raytracing::High>>;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
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
        bool initialized = false;
    };

    EnvImpl* cast(void* handle){
        return reinterpret_cast<EnvImpl*>(handle);
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
            + " cam=" + std::to_string((unsigned long long)WORLD_SPEC::CAM_WIDTH) + "x" + std::to_string((unsigned long long)WORLD_SPEC::CAM_HEIGHT)
            + " shading=" + std::to_string(HYPERDRONE_ENV_SHADING)
            + " history=" + std::to_string((unsigned long long)WORLD_SPEC::HISTORY_LENGTH);
        return value.c_str();
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
        config->cam_width = (uint32_t)WORLD_SPEC::CAM_WIDTH;
        config->cam_height = (uint32_t)WORLD_SPEC::CAM_HEIGHT;
        config->image_channels = (uint32_t)WORLD::IMAGE_CHANNELS;
        config->observation_dim = (uint32_t)WORLD::OBSERVATION_DIM;
        config->observation_dim_privileged = (uint32_t)WORLD::OBSERVATION_DIM_PRIVILEGED;
        config->action_dim = (uint32_t)WORLD::ACTION_DIM;
        config->episode_step_limit = (uint32_t)WORLD::EPISODE_STEP_LIMIT;
    }
    void hyperdrone_env_init(void* handle, const char* scene_directory, unsigned long long seed){
        EnvImpl* impl = cast(handle);
        rlt::init(impl->device, impl->rng, seed);
        rlt::rl::environments::hyperdrone::datasets::Plain dataset{scene_directory};
        rlt::init(impl->device, impl->env, dataset);
        impl->initialized = true;
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
