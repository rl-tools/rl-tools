#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_cuda.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_cuda.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/unflatten/operations_generic.h>
#include <rl_tools/nn/layers/unflatten/operations_cuda.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cuda.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/operations_generic.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/operations_generic.h>
#include <rl_tools/nn/loss_functions/mse/operations_generic.h>
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>

#include <rl_tools/utils/extrack/operations_cpu.h>

#include <array>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace rlt = rl_tools;

// =========================================================================
// Device types
// =========================================================================
#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
using LOGGER = rlt::devices::logging::CPU_TENSORBOARD<>;
#else
using LOGGER = rlt::devices::logging::CPU;
#endif
using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<float>;
using TI = typename DEVICE::index_t;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;

// =========================================================================
// L2F dynamics configuration
// =========================================================================
namespace l2f = rlt::rl::environments::l2f;
namespace obs = l2f::observation;

using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
static constexpr TI SIMULATION_FREQUENCY = 100;
static constexpr TI EPISODE_STEP_LIMIT = 500;
using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie;

static constexpr REWARD_FUNCTION reward_function = {
    false, // non_negative
    1.00,  // scale
    1.00,  // constant (survival bonus: +1 per step)
    0.00,  // termination_penalty
    0.00,  // position
    0.00,  // position_clip
    0.00,  // orientation
    0.00,  // linear_velocity
    0.00,  // angular_velocity
    0.00,  // linear_acceleration
    0.00,  // angular_acceleration
    0.00,  // action
    0.00,  // d_action
    0.00   // position_error_integral
};
static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
    0.2, 1.0, 0.3, 1.0, 1.0, true, -1, +1,
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    true, 1.5, 10, 35, 10000, 50000,
};
static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::Integration integration = {
    static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)
};
static constexpr typename PARAMETERS_TYPE::MDP mdp = { init, reward_function, {}, {}, termination };
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = { {0, 0}, {0, 0} };
static constexpr PARAMETERS_TYPE nominal_parameters = { {dynamics, integration, mdp}, disturbances };

// =========================================================================
// Environment static parameters
// =========================================================================
static constexpr TI ACTION_HISTORY_LENGTH = 1;

struct STATIC_PARAMETERS {
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI CLOSED_FORM = false;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
    using OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI,
            obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
            obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
            obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI,
            obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
    using OBSERVATION_TYPE_PRIVILEGED = obs::Position<obs::PositionSpecificationPrivileged<T, TI,
            obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecificationPrivileged<T, TI,
            obs::LinearVelocity<obs::LinearVelocitySpecificationPrivileged<T, TI,
            obs::AngularVelocity<obs::AngularVelocitySpecificationPrivileged<T, TI,
            obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
    using PARAMETERS = PARAMETERS_TYPE;
    static constexpr auto PARAMETER_VALUES = nominal_parameters;
    static constexpr T STATE_LIMIT_POSITION = 100000;
    static constexpr T STATE_LIMIT_VELOCITY = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
};

// =========================================================================
// Visual environment specification
// =========================================================================
static constexpr TI NUM_ENVS = 64;
static constexpr TI CAM_WIDTH = 64;
static constexpr TI CAM_HEIGHT = 64;
static constexpr TI NUM_PROBES = 64;

using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, NUM_ENVS, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES>;
using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;

// =========================================================================
// PPO configuration
// =========================================================================
struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
    static constexpr T ALPHA = 3e-4;
    static constexpr T EPSILON = 1e-5;
    static constexpr T EPSILON_SQRT = 1e-5;
};

struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
    static constexpr TI BATCH_SIZE = 512;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;
    static constexpr TI N_ENVIRONMENTS = NUM_ENVS;
    static constexpr TI TOTAL_STEP_LIMIT = 10000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using ACTOR_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
    using CRITIC_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
    static constexpr bool NORMALIZE_OBSERVATIONS = true;
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.00;
        static constexpr TI N_EPOCHS = 1;
        static constexpr T GAMMA = 0.99;
        static constexpr T LAMBDA = 0.95;
        static constexpr T EPSILON_CLIP = 0.2;
        static constexpr T INITIAL_ACTION_STD = 0.5;
    };
};

// CNN actor + dense critic (asymmetric observations)
template<typename T_TYPE_POLICY, typename T_TI, typename T_ENVIRONMENT, typename PARAMETERS, bool T_DYNAMIC_ALLOCATION = true>
struct ConfigApproximatorsCNN{
    static constexpr T_TI STEPS = 1;
    static constexpr T_TI FORWARD_BATCH_SIZE = PARAMETERS::BATCH_SIZE;
    template <typename CAPABILITY>
    struct Actor{
        using OBS_SHAPE = typename T_ENVIRONMENT::Observation::SHAPE;
        using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
        static constexpr T_TI IMG_H = T_ENVIRONMENT::Observation::HEIGHT;
        static constexpr T_TI IMG_W = T_ENVIRONMENT::Observation::WIDTH;
        static constexpr T_TI IMG_C = T_ENVIRONMENT::Observation::CHANNELS;
        using INPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>;
        using INPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<INPUT_FLATTEN_CONFIG>;
        using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
        using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
        using UNFLATTEN_CONFIG = rlt::nn::layers::unflatten::Configuration<T_TYPE_POLICY, T_TI, IMG_H, IMG_W, IMG_C>;
        using UNFLATTEN = rlt::nn::layers::unflatten::BindConfiguration<UNFLATTEN_CONFIG>;
        using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 16, 3, 3, 2, 2, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
        using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 32, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
        using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>;
        using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
        using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, T_ENVIRONMENT::ACTION_DIM, 2, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
        using MLP = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_CONFIG>;
        using MODULE_CHAIN = rlt::nn_models::sequential::Module<INPUT_FLATTEN, STANDARDIZATION_LAYER, UNFLATTEN, CONV1, CONV2, OUTPUT_FLATTEN, MLP>;
        using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
    };
    template <typename CAPABILITY>
    struct Critic{
        using OBS_PRIV_SHAPE = typename T_ENVIRONMENT::ObservationPrivileged::SHAPE;
        using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
        using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
        using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
        using CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
        using TYPE = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;
        using MODULE_CHAIN = rlt::nn_models::sequential::Module<STANDARDIZATION_LAYER, rlt::nn_models::sequential::Module<TYPE>>;
        using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
    };
    using ACTOR_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T_TYPE_POLICY, T_TI, typename PARAMETERS::ACTOR_OPTIMIZER_PARAMETERS, T_DYNAMIC_ALLOCATION>;
    using CRITIC_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T_TYPE_POLICY, T_TI, typename PARAMETERS::CRITIC_OPTIMIZER_PARAMETERS, T_DYNAMIC_ALLOCATION>;
    using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
    using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
    using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, T_DYNAMIC_ALLOCATION>;
    using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
    using CRITIC_TYPE = typename Critic<CAPABILITY_ADAM>::MODEL;
};

using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximatorsCNN>;

// =========================================================================
// Derived types from config
// =========================================================================
using PPO_SPEC = typename LOOP_CORE_CONFIG::PPO_SPEC;
using PPO_TYPE = typename LOOP_CORE_CONFIG::PPO_TYPE;
using PPO_BUFFERS_TYPE = typename LOOP_CORE_CONFIG::PPO_BUFFERS_TYPE;
using ON_POLICY_RUNNER_SPEC = typename LOOP_CORE_CONFIG::ON_POLICY_RUNNER_SPEC;
using ON_POLICY_RUNNER_TYPE = typename LOOP_CORE_CONFIG::ON_POLICY_RUNNER_TYPE;
using ON_POLICY_RUNNER_DATASET_SPEC = typename LOOP_CORE_CONFIG::ON_POLICY_RUNNER_DATASET_SPEC;
using ON_POLICY_RUNNER_DATASET_TYPE = typename LOOP_CORE_CONFIG::ON_POLICY_RUNNER_DATASET_TYPE;
using ACTOR_OPTIMIZER = typename LOOP_CORE_CONFIG::NN::ACTOR_OPTIMIZER;
using CRITIC_OPTIMIZER = typename LOOP_CORE_CONFIG::NN::CRITIC_OPTIMIZER;
using ACTOR_BUFFERS = typename LOOP_CORE_CONFIG::ACTOR_BUFFERS;
using ACTOR_EVAL_BUFFERS = typename LOOP_CORE_CONFIG::ACTOR_EVAL_BUFFERS;
// Actor eval buffer sized for N_ENVIRONMENTS (not BATCH_SIZE)
using ACTOR_EVAL_TYPE = typename LOOP_CORE_CONFIG::NN::ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, LOOP_CORE_PARAMETERS::N_ENVIRONMENTS>;
using ACTOR_EVAL_BUFFERS_N_ENVS = typename ACTOR_EVAL_TYPE::template Buffer<true>;
using CRITIC_BUFFERS = typename LOOP_CORE_CONFIG::CRITIC_BUFFERS;
using CRITIC_BUFFERS_GAE = typename LOOP_CORE_CONFIG::CRITIC_BUFFERS_GAE;

// Constants
static constexpr TI N_ENVIRONMENTS = LOOP_CORE_PARAMETERS::N_ENVIRONMENTS;
static constexpr TI STEPS_PER_ENV = LOOP_CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV;
static constexpr TI BATCH_SIZE = LOOP_CORE_PARAMETERS::BATCH_SIZE;
static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
static constexpr TI OBS_PRIV_DIM = ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED;
static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
static constexpr TI STEPS_TOTAL = ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL;
static constexpr TI STEPS_TOTAL_ALL = ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL;
static constexpr TI N_EPOCHS = LOOP_CORE_PARAMETERS::PPO_PARAMETERS::N_EPOCHS;
static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
static constexpr TI IMG_H = ENVIRONMENT::Observation::HEIGHT;
static constexpr TI IMG_W = ENVIRONMENT::Observation::WIDTH;
static constexpr TI IMG_C = ENVIRONMENT::Observation::CHANNELS;

static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");

// =========================================================================
// Main
// =========================================================================
int main(int argc, char** argv){
    const char* scene_path = nullptr;
    TI seed = 0;
    if(argc > 1){
        scene_path = argv[1];
    }
    if(argc > 2){
        seed = std::atoi(argv[2]);
    }

    // Devices
    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device);

    // Extrack + tensorboard
    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = "l2f_visual_cuda";
    rlt::init(device, extrack_config, extrack_paths, seed);

    // RNGs
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);

    // CPU allocations
    PPO_TYPE ppo;
    PPO_BUFFERS_TYPE ppo_buffers;
    ON_POLICY_RUNNER_TYPE on_policy_runner;
    ON_POLICY_RUNNER_DATASET_TYPE dataset;
    ACTOR_OPTIMIZER actor_optimizer;
    CRITIC_OPTIMIZER critic_optimizer;
    ACTOR_EVAL_BUFFERS actor_eval_buffers_cpu;

    rlt::malloc(device, ppo);
    rlt::malloc(device, ppo_buffers);
    rlt::malloc(device, on_policy_runner);
    rlt::malloc(device, dataset);
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, critic_optimizer);
    rlt::malloc(device, actor_eval_buffers_cpu);

    // CPU transfer matrices
    rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ACTION_DIM>> cpu_actions_eval;
    rlt::malloc(device, cpu_actions_eval);

    // GPU allocations (deferred to after env init to avoid OptiX context issues)
    PPO_TYPE ppo_gpu;
    ACTOR_BUFFERS actor_buffers;
    ACTOR_EVAL_BUFFERS actor_eval_buffers_gpu;
    CRITIC_BUFFERS critic_buffers;
    CRITIC_BUFFERS_GAE critic_buffers_gae;
        // Extra rows so that BATCH_SIZE-padded evaluate from any collect step doesn't OOB
    static constexpr TI GPU_OBS_ROWS = STEPS_TOTAL + BATCH_SIZE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, OBSERVATION_DIM>>> gpu_all_observations;
    rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ACTION_DIM>> gpu_actions_eval;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_actions_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_d_action_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL_ALL, OBS_PRIV_DIM>> gpu_gae_obs;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL_ALL, 1>> gpu_gae_values;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, OBS_PRIV_DIM>> gpu_critic_obs;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, 1>> gpu_target_values;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, 1>> gpu_d_critic_output;
    RNG_GPU rng_gpu;

    // =========================================================================
    // Environment setup (shared renderer)
    // =========================================================================
    ENVIRONMENT envs[N_ENVIRONMENTS];
    typename ENVIRONMENT::Parameters env_parameters[N_ENVIRONMENTS];

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::malloc(device, envs[env_i]);
    }

    auto& env0 = envs[0];
    for(TI env_i = 1; env_i < N_ENVIRONMENTS; env_i++){
        if(envs[env_i].owns_renderer && envs[env_i].renderer != nullptr){
            rlt::free(device, *envs[env_i].renderer);
            delete envs[env_i].renderer;
        }
        if(envs[env_i].scene != nullptr){
            delete envs[env_i].scene;
        }
        envs[env_i].renderer = env0.renderer;
        envs[env_i].scene = env0.scene;
        envs[env_i].owns_renderer = false;
    }

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        envs[env_i].use_target_mode = true;
        envs[env_i].scene_path = scene_path;
        if(env_i > 0){
            envs[env_i].renderer_initialized = true;
        }
    }

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::init(device, envs[env_i]);
    }

    if(env0.scene->num_indoor_positions > 0){
        auto& target = env0.scene->indoor_positions[0];
        T target_translation[3] = {
            target.position[0],
            target.position[1] + env0.eye_height,
            target.position[2]
        };
        for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
            for(TI j = 0; j < 3; j++){
                envs[env_i].target_scene_translation[j] = target_translation[j];
            }
        }
    }

    // =========================================================================
    // GPU device init (after env init; must be after OptiX context creation)
    // PPO initialization (CPU)
    // =========================================================================
    rlt::init(device, ppo, actor_optimizer, critic_optimizer, rng);
    {
        rlt::set_all(device, on_policy_runner.episode_step, 0);
        rlt::set_all(device, on_policy_runner.episode_return, (T)0);
        rlt::set_all(device, on_policy_runner.truncated, true);
        for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::set(on_policy_runner.environments, 0, env_i, envs[env_i]);
            rlt::set(on_policy_runner.env_parameters, 0, env_i, env_parameters[env_i]);
        }
    }

    // Observation normalization warmup (CPU only, before GPU init)
    if(LOOP_CORE_PARAMETERS::NORMALIZE_OBSERVATIONS){
        ACTOR_BUFFERS actor_buffers_cpu;
        CRITIC_BUFFERS critic_buffers_cpu;
        rlt::malloc(device, actor_buffers_cpu);
        rlt::malloc(device, critic_buffers_cpu);
        std::cout << "Running observation normalization warmup..." << std::endl;
        rlt::collect(device, dataset, on_policy_runner, ppo.actor, actor_eval_buffers_cpu, rng);
        using OBS_SHAPE = typename ENVIRONMENT::Observation::SHAPE;
        using ACTOR_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_SHAPE, BATCH_SIZE>, (TI)1>;
        using OBS_PRIV_SHAPE = typename ENVIRONMENT::ObservationPrivileged::SHAPE;
        using CRITIC_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, BATCH_SIZE>, (TI)1>;
        rlt::Mode<rlt::nn::layers::standardize::AccumulateMode<>> accumulate_mode;
        for(TI batch_i = 0; batch_i < N_BATCHES; batch_i++){
            auto batch_observations = rlt::view_range(device, dataset.all_observations, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            auto batch_observations_reshaped = rlt::reshape_row_major(device, batch_observations, ACTOR_INPUT_SHAPE{});
            rlt::forward(device, ppo.actor, batch_observations_reshaped, actor_buffers_cpu, rng, accumulate_mode);
            auto batch_observations_privileged = rlt::view_range(device, dataset.all_observations_privileged, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            auto batch_observations_privileged_reshaped = rlt::reshape_row_major(device, batch_observations_privileged, CRITIC_INPUT_SHAPE{});
            rlt::forward(device, ppo.critic, batch_observations_privileged_reshaped, critic_buffers_cpu, rng, accumulate_mode);
        }
        std::cout << "Observation normalization warmup complete." << std::endl;
        rlt::free(device, actor_buffers_cpu);
        rlt::free(device, critic_buffers_cpu);
        rlt::set_all(device, on_policy_runner.episode_step, 0);
        rlt::set_all(device, on_policy_runner.episode_return, (T)0);
        rlt::set_all(device, on_policy_runner.truncated, true);
        on_policy_runner.step = 0;
    }

    // GPU device init AFTER warmup (warmup renders change CUDA context)
    rlt::init(device_gpu);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_gpu, rng_gpu, seed);
    rlt::malloc(device_gpu, ppo_gpu);
    rlt::malloc(device_gpu, actor_buffers);
    rlt::malloc(device_gpu, actor_eval_buffers_gpu);
    rlt::malloc(device_gpu, critic_buffers);
    rlt::malloc(device_gpu, critic_buffers_gae);
    rlt::malloc(device_gpu, gpu_all_observations);
    rlt::malloc(device_gpu, gpu_actions_eval);
    rlt::malloc(device_gpu, gpu_actions_train);
    rlt::malloc(device_gpu, gpu_d_action_train);
    rlt::malloc(device_gpu, gpu_gae_obs);
    rlt::malloc(device_gpu, gpu_gae_values);
    rlt::malloc(device_gpu, gpu_critic_obs);
    rlt::malloc(device_gpu, gpu_target_values);
    rlt::malloc(device_gpu, gpu_d_critic_output);
    // GPU optimizer copies (CUDA update kernel needs optimizer state on GPU)
    ACTOR_OPTIMIZER actor_optimizer_gpu;
    CRITIC_OPTIMIZER critic_optimizer_gpu;
    rlt::malloc(device_gpu, actor_optimizer_gpu);
    rlt::malloc(device_gpu, critic_optimizer_gpu);
    rlt::copy(device, device_gpu, ppo, ppo_gpu);
    rlt::copy(device, device_gpu, actor_optimizer, actor_optimizer_gpu);
    rlt::copy(device, device_gpu, critic_optimizer, critic_optimizer_gpu);
    rlt::reset_optimizer_state(device_gpu, actor_optimizer_gpu, ppo_gpu.actor);
    rlt::reset_optimizer_state(device_gpu, critic_optimizer_gpu, ppo_gpu.critic);

    // =========================================================================
    // Training loop
    // =========================================================================
    std::cout << "Starting PPO training (visual L2F hover, CUDA)" << std::endl;
    std::cout << "  N_ENVIRONMENTS: " << N_ENVIRONMENTS << std::endl;
    std::cout << "  STEPS_PER_ENV: " << STEPS_PER_ENV << std::endl;
    std::cout << "  BATCH_SIZE: " << BATCH_SIZE << std::endl;
    std::cout << "  N_BATCHES: " << N_BATCHES << std::endl;
    std::cout << "  OBSERVATION_DIM (image): " << OBSERVATION_DIM << std::endl;
    std::cout << "  OBSERVATION_DIM_PRIVILEGED: " << OBS_PRIV_DIM << std::endl;
    std::cout << "  N_EPOCHS: " << N_EPOCHS << std::endl;

    auto training_start = std::chrono::high_resolution_clock::now();
    std::array<rlt::CameraData, N_ENVIRONMENTS> cameras;
    static constexpr TI N_PPO_STEPS = LOOP_CORE_PARAMETERS::STEP_LIMIT;

    for(TI ppo_step_i = 0; ppo_step_i < N_PPO_STEPS; ppo_step_i++){
        auto step_start = std::chrono::high_resolution_clock::now();
        rlt::set_step(device, device.logger, on_policy_runner.step);

        // =================================================================
        // Collect visual
        // =================================================================
        for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
            // Episode management + privileged observations (CPU)
            for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                auto& env = rlt::get(on_policy_runner.environments, 0, env_i);
                auto& state = rlt::get(on_policy_runner.states, 0, env_i);
                auto& parameters = rlt::get(on_policy_runner.env_parameters, 0, env_i);

                if(rlt::get(on_policy_runner.truncated, 0, env_i)){
                    static constexpr TI EPISODE_LOG_CADENCE = 100;
                    rlt::add_scalar(device, device.logger, "episode/length", rlt::get(on_policy_runner.episode_step, 0, env_i), EPISODE_LOG_CADENCE);
                    rlt::add_scalar(device, device.logger, "episode/return", rlt::get(on_policy_runner.episode_return, 0, env_i), EPISODE_LOG_CADENCE);
                    rlt::set(on_policy_runner.truncated, 0, env_i, false);
                    rlt::set(on_policy_runner.episode_step, 0, env_i, (TI)0);
                    rlt::set(on_policy_runner.episode_return, 0, env_i, (T)0);
                    rlt::sample_initial_parameters(device, env, parameters, rng);
                    rlt::sample_initial_state(device, env, parameters, state, rng);
                }

                // Privileged observation
                TI obs_row = step_i * N_ENVIRONMENTS + env_i;
                auto obs_priv_slice = rlt::view(device, dataset.all_observations_privileged, obs_row);
                auto obs_priv_matrix = rlt::matrix_view(device, obs_priv_slice);
                rlt::observe(device, env, parameters, state, typename ENVIRONMENT::ObservationPrivileged{}, obs_priv_matrix, rng);

                // Camera
                cameras[env_i] = rlt::rl::environments::l2f_visual::make_camera_for_state(device, env, parameters, state);
            }

            // GPU: batch render + pixel conversion
            T* obs_ptr = rlt::data(gpu_all_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
            rlt::observe_batch_render_gpu(device, env0, cameras.data(), N_ENVIRONMENTS, obs_ptr);

            // GPU: actor evaluate (padded to BATCH_SIZE, same buffer as training)
            auto gpu_obs_slice = rlt::view_range(device_gpu, gpu_all_observations, (TI)(step_i * N_ENVIRONMENTS), rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            using EVAL_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<typename ENVIRONMENT::Observation::SHAPE, BATCH_SIZE>, (TI)1>;
            auto gpu_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_obs_slice, EVAL_INPUT_SHAPE{});
            auto gpu_actions_train_tensor_eval = rlt::to_tensor(device_gpu, gpu_actions_train);
            auto gpu_actions_train_reshaped_eval = rlt::reshape_row_major(device_gpu, gpu_actions_train_tensor_eval, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
            rlt::evaluate(device_gpu, ppo_gpu.actor, gpu_obs_reshaped, gpu_actions_train_reshaped_eval, actor_buffers, rng_gpu);
            cudaDeviceSynchronize();
            // Copy first N_ENVS actions
            auto gpu_actions_first_n = rlt::view(device_gpu, gpu_actions_train, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), 0, 0);
            rlt::copy(device_gpu, device_gpu, gpu_actions_first_n, gpu_actions_eval);

            // GPU->CPU: copy action means
            rlt::copy(device_gpu, device, gpu_actions_eval, cpu_actions_eval);

            // CPU: store action means + epilogue
            auto actions_mean_view = rlt::view(device, dataset.actions_mean, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
            auto actions_view = rlt::view(device, dataset.actions, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
            rlt::copy(device, device, cpu_actions_eval, actions_mean_view);

            auto& last_layer_cpu = rlt::get_last_layer(ppo.actor);
            auto log_std = rlt::matrix_view(device, last_layer_cpu.log_std.parameters);
            rlt::rl::components::on_policy_runner::epilogue(device, dataset, on_policy_runner, actions_mean_view, actions_view, log_std, rng, step_i);
        }

        // Final privileged observations
        for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
            auto& env = rlt::get(on_policy_runner.environments, 0, env_i);
            auto& state = rlt::get(on_policy_runner.states, 0, env_i);
            auto& parameters = rlt::get(on_policy_runner.env_parameters, 0, env_i);
            TI obs_row = STEPS_PER_ENV * N_ENVIRONMENTS + env_i;
            auto obs_priv_slice = rlt::view(device, dataset.all_observations_privileged, obs_row);
            auto obs_priv_matrix = rlt::matrix_view(device, obs_priv_slice);
            rlt::observe(device, env, parameters, state, typename ENVIRONMENT::ObservationPrivileged{}, obs_priv_matrix, rng);
        }
        on_policy_runner.step += N_ENVIRONMENTS * STEPS_PER_ENV;

        // Reset CUDA state after OptiX renders
        {
            cudaError_t err;
            do { err = cudaGetLastError(); } while(err != cudaSuccess);
            cudaDeviceSynchronize();
            cublasDestroy(device_gpu.handle);
            cublasCreate(&device_gpu.handle);
            if(device_gpu.stream != 0) cublasSetStream(device_gpu.handle, device_gpu.stream);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
            cudnnDestroy(device_gpu.cudnn_handle);
            cudnnCreate(&device_gpu.cudnn_handle);
            if(device_gpu.stream != 0) cudnnSetStream(device_gpu.cudnn_handle, device_gpu.stream);
#endif
        }

        // =================================================================
        // GAE
        // =================================================================
        {
            // GAE on CPU (privileged obs are small: 4160x22)
            CRITIC_BUFFERS_GAE critic_buffers_gae_cpu;
            rlt::malloc(device, critic_buffers_gae_cpu);
            using OBS_PRIV_SHAPE = typename ON_POLICY_RUNNER_DATASET_TYPE::OBS_PRIV_SHAPE;
            using CRITIC_GAE_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, STEPS_TOTAL_ALL>, (TI)1>;
            auto all_obs_priv_reshaped = rlt::reshape_row_major(device, dataset.all_observations_privileged, CRITIC_GAE_INPUT_SHAPE{});
            auto all_values_tensor = rlt::to_tensor(device, dataset.all_values);
            auto all_values_reshaped = rlt::reshape_row_major(device, all_values_tensor, rlt::tensor::Shape<TI, 1, STEPS_TOTAL_ALL, 1>{});
            rlt::evaluate(device, ppo.critic, all_obs_priv_reshaped, all_values_reshaped, critic_buffers_gae_cpu, rng);
            rlt::free(device, critic_buffers_gae_cpu);
        }
        rlt::estimate_generalized_advantages(device, dataset, typename PPO_TYPE::SPEC::PARAMETERS{});

        // =================================================================
        // Train
        // =================================================================
        // Sync GPU->CPU: only copy log_std (not full model, which can crash with conv2d workspace)
        {
            auto& ll_gpu = rlt::get_last_layer(ppo_gpu.actor);
            auto& ll_cpu = rlt::get_last_layer(ppo.actor);
            rlt::copy(device_gpu, device, ll_gpu.log_std.parameters, ll_cpu.log_std.parameters);
            rlt::copy(device_gpu, device, ll_gpu.log_std.gradient, ll_cpu.log_std.gradient);
        }

        for(TI epoch_i = 0; epoch_i < N_EPOCHS; epoch_i++){
            // Random batch order (instead of shuffling row data)
            TI batch_order[N_BATCHES];
            for(TI i = 0; i < N_BATCHES; i++) batch_order[i] = i;
            for(TI i = N_BATCHES - 1; i > 0; i--){
                TI j = rlt::random::uniform_int_distribution(device.random, (TI)0, i, rng);
                std::swap(batch_order[i], batch_order[j]);
            }

            for(TI batch_idx = 0; batch_idx < N_BATCHES; batch_idx++){
                TI batch_i = batch_order[batch_idx];
                TI batch_offset = batch_i * BATCH_SIZE;

                rlt::zero_gradient(device_gpu, ppo_gpu.critic);
                rlt::zero_gradient(device_gpu, ppo_gpu.actor);

                // --- Actor forward on GPU ---
                auto gpu_obs_batch = rlt::view_range(device_gpu, gpu_all_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                using ACTOR_INPUT_SHAPE2 = rlt::tensor::Prepend<rlt::tensor::Prepend<typename ENVIRONMENT::Observation::SHAPE, BATCH_SIZE>, (TI)1>;
                auto gpu_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_obs_batch, ACTOR_INPUT_SHAPE2{});

                auto gpu_actions_train_tensor = rlt::to_tensor(device_gpu, gpu_actions_train);
                auto gpu_actions_train_reshaped = rlt::reshape_row_major(device_gpu, gpu_actions_train_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                rlt::forward(device_gpu, ppo_gpu.actor, gpu_obs_batch_reshaped, gpu_actions_train_reshaped, actor_buffers, rng_gpu);
                cudaDeviceSynchronize();

                // GPU->CPU: copy action outputs
                rlt::copy(device_gpu, device, gpu_actions_train, ppo_buffers.current_batch_actions);

                // GPU->CPU: copy log_std
                auto& last_layer_gpu = rlt::get_last_layer(ppo_gpu.actor);
                auto& last_layer_cpu = rlt::get_last_layer(ppo.actor);
                rlt::copy(device_gpu, device, last_layer_gpu.log_std.parameters, last_layer_cpu.log_std.parameters);
                rlt::copy(device_gpu, device, last_layer_gpu.log_std.gradient, last_layer_cpu.log_std.gradient);

                // --- CPU: PPO loss computation ---
                auto batch_actions = rlt::view(device, dataset.actions, rlt::matrix::ViewSpec<BATCH_SIZE, ACTION_DIM>(), batch_offset, 0);
                auto batch_action_log_probs = rlt::view(device, dataset.action_log_probs, rlt::matrix::ViewSpec<BATCH_SIZE, 1>(), batch_offset, 0);
                auto batch_advantages = rlt::view(device, dataset.advantages, rlt::matrix::ViewSpec<BATCH_SIZE, 1>(), batch_offset, 0);
                auto batch_target_values = rlt::view(device, dataset.target_values, rlt::matrix::ViewSpec<BATCH_SIZE, 1>(), batch_offset, 0);

                T advantage_mean = 0, advantage_std = 0;
                if(PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE){
                    for(TI i = 0; i < BATCH_SIZE; i++){
                        T adv = rlt::get(batch_advantages, i, 0);
                        advantage_mean += adv;
                        advantage_std += adv * adv;
                    }
                    advantage_mean /= BATCH_SIZE;
                    advantage_std /= BATCH_SIZE;
                    advantage_std = rlt::math::sqrt(device.math, rlt::math::max(device.math, (T)0, advantage_std - advantage_mean * advantage_mean));
                }

                for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
                    T action_log_prob = 0;
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        T current_action = rlt::get(ppo_buffers.current_batch_actions, batch_step_i, action_i);
                        T rollout_action = rlt::get(batch_actions, batch_step_i, action_i);
                        T current_action_log_std = rlt::get(device, last_layer_cpu.log_std.parameters, action_i);
                        T current_action_std = rlt::math::exp(device.math, current_action_log_std);

                        action_log_prob += rlt::random::normal_distribution::log_prob(device.random, current_action, current_action_log_std, rollout_action);
                        rlt::set(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, rlt::random::normal_distribution::d_log_prob_d_mean(device.random, current_action, current_action_log_std, rollout_action));

                        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                            T d_entropy_loss_d_current_action_log_std = -(T)1/BATCH_SIZE * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT;
                            rlt::increment(device, last_layer_cpu.log_std.gradient, d_entropy_loss_d_current_action_log_std, action_i);
                            T d_action_log_prob_d_current_action_log_std = rlt::random::normal_distribution::d_log_prob_d_log_std(device.random, current_action, current_action_log_std, rollout_action);
                            rlt::set(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i, d_action_log_prob_d_current_action_log_std);
                        }
                    }
                    T rollout_action_log_prob = rlt::get(batch_action_log_probs, batch_step_i, 0);
                    T advantage = rlt::get(batch_advantages, batch_step_i, 0);
                    if(PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE){
                        advantage = (advantage - advantage_mean) / (advantage_std + PPO_SPEC::PARAMETERS::ADVANTAGE_EPSILON);
                    }
                    T log_ratio = action_log_prob - rollout_action_log_prob;
                    T ratio = rlt::math::exp(device.math, log_ratio);
                    T clipped_ratio = rlt::math::clamp(device.math, ratio, 1 - PPO_SPEC::PARAMETERS::EPSILON_CLIP, 1 + PPO_SPEC::PARAMETERS::EPSILON_CLIP);
                    bool clipped = ratio != clipped_ratio;
                    T normal_advantage = ratio * advantage;
                    T clipped_advantage = clipped_ratio * advantage;
                    bool ratio_min_switch = normal_advantage - clipped_advantage <= (T)0;
                    T d_loss_d_pessimistic_surrogate = -(T)1/BATCH_SIZE;
                    T d_pessimistic_surrogate_d_ratio = ratio_min_switch ? advantage : (clipped ? 0 : advantage);
                    T d_loss_d_action_log_prob = d_loss_d_pessimistic_surrogate * d_pessimistic_surrogate_d_ratio * ratio;
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        rlt::multiply(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, d_loss_d_action_log_prob);
                        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                            T current_d = rlt::get(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i);
                            rlt::increment(device, last_layer_cpu.log_std.gradient, d_loss_d_action_log_prob * current_d, action_i);
                        }
                    }
                }

                // CPU->GPU: copy log_std back
                rlt::copy(device, device_gpu, last_layer_cpu.log_std.parameters, last_layer_gpu.log_std.parameters);
                rlt::copy(device, device_gpu, last_layer_cpu.log_std.gradient, last_layer_gpu.log_std.gradient);

                // CPU->GPU: copy action gradient
                rlt::copy(device, device_gpu, ppo_buffers.d_action_log_prob_d_action, gpu_d_action_train);

                // --- Actor backward on GPU ---
                auto gpu_d_action_tensor = rlt::to_tensor(device_gpu, gpu_d_action_train);
                auto gpu_d_action_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                rlt::backward(device_gpu, ppo_gpu.actor, gpu_obs_batch_reshaped, gpu_d_action_reshaped, actor_buffers);
                cudaDeviceSynchronize();
                rlt::step(device_gpu, actor_optimizer_gpu, ppo_gpu.actor);
                cudaDeviceSynchronize();

                // --- Critic forward/backward on GPU ---
                auto batch_obs_priv = rlt::view_range(device, dataset.all_observations_privileged, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto batch_obs_priv_matrix = rlt::matrix_view(device, batch_obs_priv);
                rlt::copy(device, device_gpu, batch_obs_priv_matrix, gpu_critic_obs);
                auto gpu_critic_obs_tensor = rlt::to_tensor(device_gpu, gpu_critic_obs);
                auto gpu_critic_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_critic_obs_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, OBS_PRIV_DIM>{});

                rlt::forward(device_gpu, ppo_gpu.critic, gpu_critic_obs_reshaped, critic_buffers, rng_gpu);
                cudaDeviceSynchronize();

                // MSE gradient on CPU (avoids CUDA matrix get/set issues)
                {
                    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, 1>> cpu_critic_output, cpu_d_critic;
                    rlt::malloc(device, cpu_critic_output);
                    rlt::malloc(device, cpu_d_critic);
                    auto critic_output_tensor = rlt::output(device_gpu, ppo_gpu.critic);
                    auto critic_output_matrix = rlt::matrix_view(device_gpu, critic_output_tensor);
                    rlt::copy(device_gpu, device, critic_output_matrix, cpu_critic_output);
                    rlt::nn::loss_functions::mse::gradient(device, cpu_critic_output, batch_target_values, cpu_d_critic, (T)0.5);
                    rlt::copy(device, device_gpu, cpu_d_critic, gpu_d_critic_output);
                    rlt::free(device, cpu_critic_output);
                    rlt::free(device, cpu_d_critic);
                }
                auto gpu_d_critic_tensor = rlt::to_tensor(device_gpu, gpu_d_critic_output);
                auto gpu_d_critic_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_critic_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 1>{});
                rlt::backward(device_gpu, ppo_gpu.critic, gpu_critic_obs_reshaped, gpu_d_critic_reshaped, critic_buffers);
                cudaDeviceSynchronize();
                rlt::step(device_gpu, critic_optimizer_gpu, ppo_gpu.critic);
                cudaDeviceSynchronize();
            }
        }

        // Logging
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> training_elapsed = now - training_start;
        std::chrono::duration<T> step_elapsed = now - step_start;
        T steps_per_second_lifetime = on_policy_runner.step / training_elapsed.count();
        T steps_per_second_current = N_ENVIRONMENTS * STEPS_PER_ENV / step_elapsed.count();
        std::cout << "PPO step: " << std::setw(6) << ppo_step_i
                  << " env step: " << std::setw(10) << on_policy_runner.step
                  << " elapsed: " << std::setw(8) << std::setprecision(2) << training_elapsed.count() << "s"
                  << " (lifetime: " << std::setw(8) << std::setprecision(2) << steps_per_second_lifetime << " steps/s"
                  << ", current: " << std::setw(8) << std::setprecision(2) << steps_per_second_current << " steps/s)" << std::endl;

        // Log actor std
        {
            auto& last_layer = rlt::get_last_layer(ppo.actor);
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                T log_std_val = rlt::get(device, last_layer.log_std.parameters, action_i);
                rlt::add_scalar(device, device.logger, "actor/log_std", log_std_val, 100);
            }
        }
    }

    std::cout << "Training finished at step " << on_policy_runner.step << std::endl;

    // =========================================================================
    // Cleanup
    // =========================================================================
    for(TI env_i = 1; env_i < N_ENVIRONMENTS; env_i++){
        envs[env_i].renderer = nullptr;
        envs[env_i].scene = nullptr;
    }
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::free(device, envs[env_i]);
    }

    rlt::free(device, cpu_actions_eval);
    rlt::free(device, ppo);
    rlt::free(device, ppo_buffers);
    rlt::free(device, on_policy_runner);
    rlt::free(device, dataset);
    rlt::free(device, actor_optimizer);
    rlt::free(device, critic_optimizer);
    rlt::free(device_gpu, actor_optimizer_gpu);
    rlt::free(device_gpu, critic_optimizer_gpu);
    rlt::free(device, actor_eval_buffers_cpu);

    rlt::free(device_gpu, ppo_gpu);
    rlt::free(device_gpu, actor_buffers);
    rlt::free(device_gpu, actor_eval_buffers_gpu);
    rlt::free(device_gpu, critic_buffers);
    rlt::free(device_gpu, critic_buffers_gae);
    rlt::free(device_gpu, gpu_all_observations);
    rlt::free(device_gpu, gpu_actions_eval);
    rlt::free(device_gpu, gpu_actions_train);
    rlt::free(device_gpu, gpu_d_action_train);
    rlt::free(device_gpu, gpu_gae_obs);
    rlt::free(device_gpu, gpu_gae_values);
    rlt::free(device_gpu, gpu_critic_obs);
    rlt::free(device_gpu, gpu_target_values);
    rlt::free(device_gpu, gpu_d_critic_output);

    return 0;
}
