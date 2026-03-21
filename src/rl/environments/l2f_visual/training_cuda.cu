#define RL_TOOLS_DISABLE_VISUAL // comment out to enable rendering and image input
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_cuda.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_cuda.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cuda.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/unflatten/operations_generic.h>
#include <rl_tools/nn/layers/unflatten/operations_cuda.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_cuda.h>
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
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/utils/zlib/operations_cpu.h>

#include <array>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <vector>

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

// Actor state observation: position (3D) + orientation rotation matrix (9D) + linear velocity (3D) + angular velocity (3D) + action history (ACTION_HISTORY_LENGTH * 4D)
static constexpr TI ACTION_HISTORY_LENGTH = 8;
using ACTOR_STATE_OBS = obs::Position<obs::PositionSpecification<T, TI,
    obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
    obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
    obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI,
    obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
static constexpr TI STATE_OBS_DIM = ACTOR_STATE_OBS::DIM; // 18 + 8*4 = 50

using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
static constexpr TI SIMULATION_FREQUENCY = 100;
static constexpr TI EPISODE_STEP_LIMIT = 500;
using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::soft_rigid;

static constexpr REWARD_FUNCTION reward_function = {
    false,    // non_negative
    1.00,     // scale
    2.00,     // constant (survival bonus)
    -100.00,  // termination_penalty
    1.00,     // position
    0.00,     // position_clip
    0.20,     // orientation
    0.50,     // linear_velocity
    0.50,     // angular_velocity
    0.00,     // linear_acceleration
    0.00,     // angular_acceleration
    0.00,     // action
    0.50,     // d_action (action smoothness)
    0.00      // position_error_integral
};
static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
    0.1,                  // guidance (10% chance of spawning at origin)
    2.2,                  // max_position (~rotor_distance * 10)
    1.5707963267948966,   // max_angle (90 degrees)
    1.0,                  // max_linear_velocity
    1.0,                  // max_angular_velocity
    true,                 // relative_rpm
    -1, 0,               // min/max rpm
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    true, 4.4, 10, 35, 10000, 50000,
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
// Trajectory recording for extrack UI
// =========================================================================
static constexpr TI TRAJECTORY_SAVE_INTERVAL = 100; // save every N PPO steps
static constexpr TI TRAJECTORY_NUM_ENVS = 10; // record from first N environments
static constexpr TI TRAJECTORY_MAX_EPISODES = 10; // keep only last N completed episodes

struct TrajectoryStep {
    typename ENVIRONMENT::State state;
    T actions[ENVIRONMENT::ACTION_DIM];
    T reward;
    bool terminated;
};
struct EpisodeRecorder {
    std::vector<TrajectoryStep> current_episode;
    bool episode_started = false;
};

std::string trajectory_episodes_to_json(DEVICE& device, ENVIRONMENT& env, typename ENVIRONMENT::Parameters& parameters,
    const std::vector<std::vector<TrajectoryStep>>& episodes, T dt){
    if(episodes.empty()) return "[]";
    // The extrack UI iterates all episodes at the same step index, so all
    // episodes must have the same length.  Pad shorter ones by repeating
    // the last step with terminated=true.
    TI max_len = 0;
    for(auto& ep : episodes) if(ep.size() > max_len) max_len = ep.size();
    std::string json = "[";
    for(TI ep_i = 0; ep_i < episodes.size(); ep_i++){
        auto& episode = episodes[ep_i];
        json += "{\"parameters\": " + rlt::json(device, env.dynamics, parameters.dynamics) + ",\n";
        json += "\"trajectory\": [";
        for(TI step_i = 0; step_i < max_len; step_i++){
            auto& s = (step_i < episode.size()) ? episode[step_i] : episode.back();
            json += "{\"state\":" + rlt::json(device, env.dynamics, parameters.dynamics, s.state) + ",";
            json += "\"action\":[";
            for(TI a = 0; a < ENVIRONMENT::ACTION_DIM; a++){
                json += std::to_string(s.actions[a]);
                if(a < ENVIRONMENT::ACTION_DIM - 1) json += ",";
            }
            json += "],";
            json += "\"dt\":" + std::to_string(dt) + ",";
            json += "\"reward\":" + std::to_string(s.reward) + ",";
            bool terminated = (step_i < episode.size()) ? s.terminated : true;
            json += "\"terminated\":" + (terminated ? std::string("true") : std::string("false"));
            json += "}";
            if(step_i < max_len - 1) json += ",";
        }
        json += "]}";
        if(ep_i < episodes.size() - 1) json += ",";
    }
    json += "]";
    return json;
}

// =========================================================================
// PPO configuration
// =========================================================================
struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
    static constexpr T ALPHA = 1e-3;
};

struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
    static constexpr TI BATCH_SIZE = 2048;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 128;
    static constexpr TI N_ENVIRONMENTS = NUM_ENVS;
    static constexpr TI TOTAL_STEP_LIMIT = 15000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using ACTOR_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
    using CRITIC_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
    static constexpr bool NORMALIZE_OBSERVATIONS = true;
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
        static constexpr TI N_EPOCHS = 2;
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
#ifdef RL_TOOLS_DISABLE_VISUAL
        // State-only actor: Standardize → MLP → ACTION_DIM
        using INPUT_SHAPE = rlt::tensor::Shape<T_TI, STEPS, FORWARD_BATCH_SIZE, STATE_OBS_DIM>;
        using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
        using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
        using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, T_ENVIRONMENT::ACTION_DIM, 2, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
        using MLP = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_CONFIG>;
        using MODULE_CHAIN = rlt::nn_models::sequential::Module<STANDARDIZATION_LAYER, rlt::nn_models::sequential::Module<MLP>>;
        using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
#else
        // Image branch input shape
        using OBS_SHAPE = typename T_ENVIRONMENT::Observation::SHAPE;
        using IMAGE_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
        // State branch input shape
        using STATE_INPUT_SHAPE = rlt::tensor::Shape<T_TI, STEPS, FORWARD_BATCH_SIZE, STATE_OBS_DIM>;

        static constexpr T_TI IMG_H = T_ENVIRONMENT::Observation::HEIGHT;
        static constexpr T_TI IMG_W = T_ENVIRONMENT::Observation::WIDTH;
        static constexpr T_TI IMG_C = T_ENVIRONMENT::Observation::CHANNELS;

        // Image branch: Flatten→Standardize→Unflatten→Conv1→Conv2→Flatten→Dense(64)
        using INPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>;
        using INPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<INPUT_FLATTEN_CONFIG>;
        using IMAGE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
        using IMAGE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<IMAGE_STANDARDIZE_CONFIG>;
        using UNFLATTEN_CONFIG = rlt::nn::layers::unflatten::Configuration<T_TYPE_POLICY, T_TI, IMG_H, IMG_W, IMG_C>;
        using UNFLATTEN = rlt::nn::layers::unflatten::BindConfiguration<UNFLATTEN_CONFIG>;
        using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 16, 3, 3, 2, 2, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
        using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 32, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
        using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>;
        using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
        using IMAGE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, T_TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION>;
        using IMAGE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<IMAGE_DENSE_EMBED_CONFIG>;
        using IMAGE_BRANCH = rlt::nn_models::sequential::Module<INPUT_FLATTEN, IMAGE_STANDARDIZE, UNFLATTEN, CONV1, CONV2, OUTPUT_FLATTEN, IMAGE_DENSE_EMBED>;

        // State branch: Standardize→Dense(64)
        using STATE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
        using STATE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STATE_STANDARDIZE_CONFIG>;
        using STATE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, T_TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION>;
        using STATE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<STATE_DENSE_EMBED_CONFIG>;
        using STATE_BRANCH = rlt::nn_models::sequential::Module<STATE_STANDARDIZE, STATE_DENSE_EMBED>;

        // Head MLP: 128D → 64 → 64 → ACTION_DIM
        using MLP_HEAD_CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, T_ENVIRONMENT::ACTION_DIM, 2, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
        using MLP_HEAD = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_HEAD_CONFIG>;

        // Parallel model: two branches + head
        using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, IMAGE_BRANCH, STATE_BRANCH, IMAGE_INPUT_SHAPE, STATE_INPUT_SHAPE, MLP_HEAD>;
#endif
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

    rlt::malloc(device, ppo);
    rlt::malloc(device, ppo_buffers);
    rlt::malloc(device, on_policy_runner);
    rlt::malloc(device, dataset);
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, critic_optimizer);

    // CPU transfer matrices
    rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ACTION_DIM>> cpu_actions_eval;
    rlt::malloc(device, cpu_actions_eval);

    // CPU state observations buffer (per collect step, copied to GPU)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, STATE_OBS_DIM>>> cpu_state_obs_step;
    rlt::malloc(device, cpu_state_obs_step);

    // CPU-side accumulated state observations for training (populated during collection)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, STATE_OBS_DIM>>> cpu_all_state_observations;
    rlt::malloc(device, cpu_all_state_observations);

    // CPU-side actor/critic buffers for training
    ACTOR_BUFFERS actor_buffers_cpu;
    CRITIC_BUFFERS critic_buffers_cpu;
    rlt::malloc(device, actor_buffers_cpu);
    rlt::malloc(device, critic_buffers_cpu);

    // GPU allocations (deferred to after env init to avoid OptiX context issues)
    PPO_TYPE ppo_gpu;
    ACTOR_BUFFERS actor_buffers;
    CRITIC_BUFFERS critic_buffers;
    CRITIC_BUFFERS_GAE critic_buffers_gae;
        // Extra rows so that BATCH_SIZE-padded evaluate from any collect step doesn't OOB
    static constexpr TI GPU_OBS_ROWS = STEPS_TOTAL + BATCH_SIZE;
#ifndef RL_TOOLS_DISABLE_VISUAL
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, OBSERVATION_DIM>>> gpu_all_observations;
#endif
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, STATE_OBS_DIM>>> gpu_all_state_observations;
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
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        envs[env_i].use_target_mode = true;
    }
#ifdef RL_TOOLS_DISABLE_VISUAL
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::init(device, envs[env_i].dynamics);
    }
#else
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
        envs[env_i].scene_path = scene_path;
        if(env_i > 0){
            envs[env_i].renderer_initialized = true;
        }
    }

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::init(device, envs[env_i]);
    }

#endif

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

        // Warmup state observations tensor (CPU)
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, STATE_OBS_DIM>>> warmup_state_observations;
        rlt::malloc(device, warmup_state_observations);

        std::cout << "Running observation normalization warmup..." << std::endl;
        // Manual warmup collect: sample random states, observe, no actor evaluation needed
        for(TI obs_row = 0; obs_row < STEPS_TOTAL; obs_row++){
            TI env_i = obs_row % N_ENVIRONMENTS;
            typename ENVIRONMENT::State warmup_state;
            rlt::sample_initial_parameters(device, envs[env_i], env_parameters[env_i], rng);
            rlt::sample_initial_state(device, envs[env_i], env_parameters[env_i], warmup_state, rng);

#ifndef RL_TOOLS_DISABLE_VISUAL
            // Image observation
            auto obs_slice = rlt::view(device, dataset.all_observations, obs_row);
            auto obs_matrix = rlt::matrix_view(device, obs_slice);
            rlt::observe(device, envs[env_i], env_parameters[env_i], warmup_state, typename ENVIRONMENT::Observation{}, obs_matrix, rng);
#endif

            // State observation
            auto state_obs_slice = rlt::view(device, warmup_state_observations, obs_row);
            auto state_obs_matrix = rlt::matrix_view(device, state_obs_slice);
            rlt::observe(device, envs[env_i].dynamics, env_parameters[env_i].dynamics, warmup_state, ACTOR_STATE_OBS{}, state_obs_matrix, rng);

            // Privileged observation
            auto obs_priv_slice = rlt::view(device, dataset.all_observations_privileged, obs_row);
            auto obs_priv_matrix = rlt::matrix_view(device, obs_priv_slice);
            rlt::observe(device, envs[env_i], env_parameters[env_i], warmup_state, typename ENVIRONMENT::ObservationPrivileged{}, obs_priv_matrix, rng);
        }

        // Accumulate mode forward passes for standardize layers
        using STATE_INPUT_SHAPE_WARMUP = rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>;
        using OBS_PRIV_SHAPE = typename ENVIRONMENT::ObservationPrivileged::SHAPE;
        using CRITIC_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, BATCH_SIZE>, (TI)1>;
        rlt::Mode<rlt::nn::layers::standardize::AccumulateMode<>> accumulate_mode;
        for(TI batch_i = 0; batch_i < N_BATCHES; batch_i++){
            auto batch_state_observations = rlt::view_range(device, warmup_state_observations, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            auto batch_state_observations_reshaped = rlt::reshape_row_major(device, batch_state_observations, STATE_INPUT_SHAPE_WARMUP{});
#ifdef RL_TOOLS_DISABLE_VISUAL
            rlt::forward(device, ppo.actor, batch_state_observations_reshaped, actor_buffers_cpu, rng, accumulate_mode);
#else
            using IMAGE_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<typename ENVIRONMENT::Observation::SHAPE, BATCH_SIZE>, (TI)1>;
            auto batch_observations = rlt::view_range(device, dataset.all_observations, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            auto batch_observations_reshaped = rlt::reshape_row_major(device, batch_observations, IMAGE_INPUT_SHAPE{});
            rlt::forward(device, ppo.actor, batch_observations_reshaped, batch_state_observations_reshaped, actor_buffers_cpu, rng, accumulate_mode);
#endif
            auto batch_observations_privileged = rlt::view_range(device, dataset.all_observations_privileged, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            auto batch_observations_privileged_reshaped = rlt::reshape_row_major(device, batch_observations_privileged, CRITIC_INPUT_SHAPE{});
            rlt::forward(device, ppo.critic, batch_observations_privileged_reshaped, critic_buffers_cpu, rng, accumulate_mode);
        }
        std::cout << "Observation normalization warmup complete." << std::endl;
        rlt::free(device, actor_buffers_cpu);
        rlt::free(device, critic_buffers_cpu);
        rlt::free(device, warmup_state_observations);
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
    rlt::malloc(device_gpu, critic_buffers);
    rlt::malloc(device_gpu, critic_buffers_gae);
#ifndef RL_TOOLS_DISABLE_VISUAL
    rlt::malloc(device_gpu, gpu_all_observations);
#endif
    rlt::malloc(device_gpu, gpu_all_state_observations);
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
    std::cout << "  STATE_OBS_DIM: " << STATE_OBS_DIM << std::endl;
    std::cout << "  OBSERVATION_DIM_PRIVILEGED: " << OBS_PRIV_DIM << std::endl;
    std::cout << "  N_EPOCHS: " << N_EPOCHS << std::endl;

    auto training_start = std::chrono::high_resolution_clock::now();
    std::array<rlt::CameraData, N_ENVIRONMENTS> cameras;
    static constexpr TI N_PPO_STEPS = LOOP_CORE_PARAMETERS::STEP_LIMIT;

    // Trajectory recording for extrack UI
    EpisodeRecorder episode_recorders[TRAJECTORY_NUM_ENVS];
    std::vector<std::vector<TrajectoryStep>> completed_episodes;
    T simulation_dt = static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY);

    // Write ui.esm.js once
    {
        std::string ui = rlt::get_ui(device, envs[0].dynamics);
        if(!ui.empty()){
            std::filesystem::create_directories(extrack_paths.seed);
            std::ofstream ui_file(extrack_paths.seed / "ui.esm.js");
            ui_file << ui;
            std::cout << "UI written to: " << extrack_paths.seed / "ui.esm.js" << std::endl;
        }
    }

    // Curriculum state
    T curriculum_level = 0;
    static constexpr T CURRICULUM_FULL_MAX_ANGLE = 1.5707963267948966;
    static constexpr T CURRICULUM_FULL_MAX_POSITION = 2.199102;
    static constexpr T CURRICULUM_MIN_START_ANGLE = 0.3;
    static constexpr T CURRICULUM_MIN_START_POSITION = 0.1;
    static constexpr TI CURRICULUM_EVAL_INTERVAL = 50; // evaluate curriculum every N PPO steps
    T episode_length_accumulator = 0;
    TI episode_count = 0;

    // Apply initial curriculum (easy start)
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        auto& env = rlt::get(on_policy_runner.environments, 0, env_i);
        env.dynamics.parameters.mdp.init.max_angle = CURRICULUM_MIN_START_ANGLE;
        env.dynamics.parameters.mdp.init.max_position = CURRICULUM_MIN_START_POSITION;
    }

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
                    TI ep_len = rlt::get(on_policy_runner.episode_step, 0, env_i);
                    rlt::add_scalar(device, device.logger, "episode/length", ep_len, EPISODE_LOG_CADENCE);
                    rlt::add_scalar(device, device.logger, "episode/return", rlt::get(on_policy_runner.episode_return, 0, env_i), EPISODE_LOG_CADENCE);
                    episode_length_accumulator += ep_len;
                    episode_count++;
                    // Trajectory: finalize completed episode
                    if(env_i < TRAJECTORY_NUM_ENVS && episode_recorders[env_i].episode_started && !episode_recorders[env_i].current_episode.empty()){
                        completed_episodes.push_back(std::move(episode_recorders[env_i].current_episode));
                        episode_recorders[env_i].current_episode.clear();
                        if(completed_episodes.size() > TRAJECTORY_MAX_EPISODES){
                            completed_episodes.erase(completed_episodes.begin());
                        }
                    }
                    rlt::set(on_policy_runner.truncated, 0, env_i, false);
                    rlt::set(on_policy_runner.episode_step, 0, env_i, (TI)0);
                    rlt::set(on_policy_runner.episode_return, 0, env_i, (T)0);
                    rlt::sample_initial_parameters(device, env, parameters, rng);
                    rlt::sample_initial_state(device, env, parameters, state, rng);
                    if(env_i < TRAJECTORY_NUM_ENVS){
                        episode_recorders[env_i].episode_started = true;
                    }
                }

                // Privileged observation
                TI obs_row = step_i * N_ENVIRONMENTS + env_i;
                auto obs_priv_slice = rlt::view(device, dataset.all_observations_privileged, obs_row);
                auto obs_priv_matrix = rlt::matrix_view(device, obs_priv_slice);
                rlt::observe(device, env, parameters, state, typename ENVIRONMENT::ObservationPrivileged{}, obs_priv_matrix, rng);

                // State observation (for actor's state branch)
                auto state_obs_row = rlt::view(device, cpu_state_obs_step, env_i);
                auto state_obs_matrix = rlt::matrix_view(device, state_obs_row);
                rlt::observe(device, env.dynamics, parameters.dynamics, state, ACTOR_STATE_OBS{}, state_obs_matrix, rng);

#ifndef RL_TOOLS_DISABLE_VISUAL
                // Camera
                cameras[env_i] = rlt::rl::environments::l2f_visual::make_camera_for_state(device, env, parameters, state);
#endif
            }

            // CPU: accumulate state observations for training
            memcpy(
                rlt::data(cpu_all_state_observations) + (TI)(step_i * N_ENVIRONMENTS) * STATE_OBS_DIM,
                rlt::data(cpu_state_obs_step),
                N_ENVIRONMENTS * STATE_OBS_DIM * sizeof(T));

#ifdef RL_TOOLS_DISABLE_VISUAL
            // State-only actor on CPU (identical to zoo target)
            {
                auto cpu_obs_slice = rlt::view_range(device, cpu_all_state_observations, (TI)(step_i * N_ENVIRONMENTS), rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                auto cpu_obs_reshaped = rlt::reshape_row_major(device, cpu_obs_slice, rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, STATE_OBS_DIM>{});
                auto cpu_actions_tensor = rlt::to_tensor(device, cpu_actions_eval);
                auto cpu_actions_reshaped = rlt::reshape_row_major(device, cpu_actions_tensor, rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, ACTION_DIM>{});
                rlt::evaluate(device, ppo.actor, cpu_obs_reshaped, cpu_actions_reshaped, actor_buffers_cpu, rng);
            }
#else
            // CPU->GPU: copy state observations for this step
            cudaMemcpy(
                rlt::data(gpu_all_state_observations) + (TI)(step_i * N_ENVIRONMENTS) * STATE_OBS_DIM,
                rlt::data(cpu_state_obs_step),
                N_ENVIRONMENTS * STATE_OBS_DIM * sizeof(T),
                cudaMemcpyHostToDevice);

            auto gpu_state_obs_slice = rlt::view_range(device_gpu, gpu_all_state_observations, (TI)(step_i * N_ENVIRONMENTS), rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            auto gpu_state_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_state_obs_slice, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});
            auto gpu_actions_train_tensor_eval = rlt::to_tensor(device_gpu, gpu_actions_train);
            auto gpu_actions_train_reshaped_eval = rlt::reshape_row_major(device_gpu, gpu_actions_train_tensor_eval, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
            // GPU: batch render + pixel conversion
            T* obs_ptr = rlt::data(gpu_all_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
            rlt::observe_batch_render_gpu(device, env0, cameras.data(), N_ENVIRONMENTS, obs_ptr);

            // GPU: actor evaluate (parallel model, two inputs)
            auto gpu_obs_slice = rlt::view_range(device_gpu, gpu_all_observations, (TI)(step_i * N_ENVIRONMENTS), rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            using EVAL_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<typename ENVIRONMENT::Observation::SHAPE, BATCH_SIZE>, (TI)1>;
            auto gpu_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_obs_slice, EVAL_INPUT_SHAPE{});
            rlt::evaluate(device_gpu, ppo_gpu.actor, gpu_obs_reshaped, gpu_state_obs_reshaped, gpu_actions_train_reshaped_eval, actor_buffers, rng_gpu);
#endif
#ifndef RL_TOOLS_DISABLE_VISUAL
            cudaDeviceSynchronize();
            // Copy first N_ENVS actions
            auto gpu_actions_first_n = rlt::view(device_gpu, gpu_actions_train, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), 0, 0);
            rlt::copy(device_gpu, device_gpu, gpu_actions_first_n, gpu_actions_eval);

            // GPU->CPU: copy action means
            rlt::copy(device_gpu, device, gpu_actions_eval, cpu_actions_eval);
#endif
            // CPU: store action means + epilogue
            auto actions_mean_view = rlt::view(device, dataset.actions_mean, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
            auto actions_view = rlt::view(device, dataset.actions, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
            rlt::copy(device, device, cpu_actions_eval, actions_mean_view);

            // Trajectory: save pre-step states for recorded envs
            for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                if(episode_recorders[env_i].episode_started){
                    TrajectoryStep traj_step;
                    traj_step.state = rlt::get(on_policy_runner.states, 0, env_i);
                    for(TI a = 0; a < ACTION_DIM; a++){
                        traj_step.actions[a] = rlt::get(cpu_actions_eval, env_i, a);
                    }
                    traj_step.reward = 0;
                    traj_step.terminated = false;
                    episode_recorders[env_i].current_episode.push_back(traj_step);
                }
            }

            {
#ifdef RL_TOOLS_DISABLE_VISUAL
                auto& actor_log_std_layer = rlt::get_last_layer(ppo.actor);
#else
                auto& actor_log_std_layer = ppo.actor.head;
#endif
                auto log_std = rlt::matrix_view(device, actor_log_std_layer.log_std.parameters);
                rlt::rl::components::on_policy_runner::epilogue(device, dataset, on_policy_runner, actions_mean_view, actions_view, log_std, rng, step_i);
            }

            // Trajectory: fill in reward and terminated from epilogue results
            for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                if(episode_recorders[env_i].episode_started && !episode_recorders[env_i].current_episode.empty()){
                    TI pos = step_i * N_ENVIRONMENTS + env_i;
                    auto& last_step = episode_recorders[env_i].current_episode.back();
                    last_step.reward = rlt::get(dataset.rewards, pos, 0);
                    last_step.terminated = rlt::get(dataset.terminated, pos, 0);
                    // Use noisy actions from epilogue
                    for(TI a = 0; a < ACTION_DIM; a++){
                        last_step.actions[a] = rlt::get(actions_view, env_i, a);
                    }
                }
            }
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

#ifndef RL_TOOLS_DISABLE_VISUAL
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
#endif

        // =================================================================
        // GAE
        // =================================================================
        // GPU GAE
        {
            rlt::copy(device, device_gpu, ppo.critic, ppo_gpu.critic);
            auto all_obs_priv_matrix = rlt::matrix_view(device, dataset.all_observations_privileged);
            rlt::copy(device, device_gpu, all_obs_priv_matrix, gpu_gae_obs);
            auto gpu_gae_obs_tensor = rlt::to_tensor(device_gpu, gpu_gae_obs);
            auto gpu_gae_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_gae_obs_tensor, rlt::tensor::Shape<TI, 1, STEPS_TOTAL_ALL, OBS_PRIV_DIM>{});
            auto gpu_gae_values_tensor = rlt::to_tensor(device_gpu, gpu_gae_values);
            auto gpu_gae_values_reshaped = rlt::reshape_row_major(device_gpu, gpu_gae_values_tensor, rlt::tensor::Shape<TI, 1, STEPS_TOTAL_ALL, 1>{});
            rlt::evaluate(device_gpu, ppo_gpu.critic, gpu_gae_obs_reshaped, gpu_gae_values_reshaped, critic_buffers_gae, rng_gpu);
            cudaDeviceSynchronize();
            rlt::copy(device_gpu, device, gpu_gae_values, dataset.all_values);
        }
        rlt::estimate_generalized_advantages(device, dataset, typename PPO_TYPE::SPEC::PARAMETERS{});

        // =================================================================
        // Train
        // =================================================================
#ifdef RL_TOOLS_DISABLE_VISUAL
        // CPU training path — identical operations to library train()
        for(TI epoch_i = 0; epoch_i < N_EPOCHS; epoch_i++){
            // Row-level shuffle (same as library SHUFFLE_EPOCH)
            {
                auto obs_matrix = rlt::matrix_view(device, cpu_all_state_observations);
                auto obs_priv_matrix = rlt::matrix_view(device, dataset.all_observations_privileged);
                for(TI dataset_i = 0; dataset_i < STEPS_TOTAL; dataset_i++){
                    TI sample_index = rlt::random::uniform_int_distribution(device.random, dataset_i, STEPS_TOTAL - 1, rng);
                    { auto t = rlt::row(device, obs_matrix, dataset_i); auto s = rlt::row(device, obs_matrix, sample_index); rlt::swap(device, t, s); }
                    { auto t = rlt::row(device, obs_priv_matrix, dataset_i); auto s = rlt::row(device, obs_priv_matrix, sample_index); rlt::swap(device, t, s); }
                    { auto t = rlt::row(device, dataset.actions, dataset_i); auto s = rlt::row(device, dataset.actions, sample_index); rlt::swap(device, t, s); }
                    rlt::swap(device, dataset.advantages, dataset.advantages, dataset_i, 0, sample_index, 0);
                    rlt::swap(device, dataset.action_log_probs, dataset.action_log_probs, dataset_i, 0, sample_index, 0);
                    rlt::swap(device, dataset.target_values, dataset.target_values, dataset_i, 0, sample_index, 0);
                }
            }
            for(TI batch_i = 0; batch_i < N_BATCHES; batch_i++){
                TI batch_offset = batch_i * BATCH_SIZE;
                rlt::zero_gradient(device, ppo.critic);
                rlt::zero_gradient(device, ppo.actor);

                // Actor forward on CPU
                auto cpu_state_obs_batch = rlt::view_range(device, cpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto cpu_state_obs_batch_reshaped = rlt::reshape_row_major(device, cpu_state_obs_batch, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});
                auto current_batch_actions_tensor = rlt::to_tensor(device, ppo_buffers.current_batch_actions);
                auto current_batch_actions_reshaped = rlt::reshape_row_major(device, current_batch_actions_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                rlt::forward(device, ppo.actor, cpu_state_obs_batch_reshaped, current_batch_actions_reshaped, actor_buffers_cpu, rng);

                // PPO loss on CPU
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
                        auto& last_layer = rlt::get_last_layer(ppo.actor);
                        T current_action_log_std = rlt::get(device, last_layer.log_std.parameters, action_i);
                        action_log_prob += rlt::random::normal_distribution::log_prob(device.random, current_action, current_action_log_std, rollout_action);
                        rlt::set(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, rlt::random::normal_distribution::d_log_prob_d_mean(device.random, current_action, current_action_log_std, rollout_action));
                        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                            T d_entropy_loss_d_current_action_log_std = -(T)1/BATCH_SIZE * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT;
                            rlt::increment(device, last_layer.log_std.gradient, d_entropy_loss_d_current_action_log_std, action_i);
                            rlt::set(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i, rlt::random::normal_distribution::d_log_prob_d_log_std(device.random, current_action, current_action_log_std, rollout_action));
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
                            auto& last_layer = rlt::get_last_layer(ppo.actor);
                            rlt::increment(device, last_layer.log_std.gradient, d_loss_d_action_log_prob * current_d, action_i);
                        }
                    }
                }

                // Actor backward + step on CPU
                auto d_action_tensor = rlt::to_tensor(device, ppo_buffers.d_action_log_prob_d_action);
                auto d_action_reshaped = rlt::reshape_row_major(device, d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                rlt::backward(device, ppo.actor, cpu_state_obs_batch_reshaped, d_action_reshaped, actor_buffers_cpu);

                // Critic forward + backward + step on CPU
                using OBS_PRIV_SHAPE = typename ON_POLICY_RUNNER_DATASET_TYPE::OBS_PRIV_SHAPE;
                using CRITIC_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, BATCH_SIZE>, (TI)1>;
                auto batch_obs_priv = rlt::view_range(device, dataset.all_observations_privileged, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto batch_obs_priv_reshaped = rlt::reshape_row_major(device, batch_obs_priv, CRITIC_INPUT_SHAPE{});
                rlt::forward(device, ppo.critic, batch_obs_priv_reshaped, critic_buffers_cpu, rng);
                {
                    auto output_tensor = rlt::output(device, ppo.critic);
                    auto output_matrix = rlt::matrix_view(device, output_tensor);
                    rlt::nn::loss_functions::mse::gradient(device, output_matrix, batch_target_values, ppo_buffers.d_critic_output, (T)0.5);
                }
                auto d_critic_tensor = rlt::to_tensor(device, ppo_buffers.d_critic_output);
                auto d_critic_reshaped = rlt::reshape_row_major(device, d_critic_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 1>{});
                rlt::backward(device, ppo.critic, batch_obs_priv_reshaped, d_critic_reshaped, critic_buffers_cpu);

                // Optimizer steps on CPU (both at end, same as library)
                rlt::step(device, actor_optimizer, ppo.actor);
                rlt::step(device, critic_optimizer, ppo.critic);
            }
        }
#else
        // GPU training path for visual mode (hand-rolled PPO)
        // Sync GPU->CPU: only copy log_std (not full model, which can crash with conv2d workspace)
        {
            auto& ll_gpu = ppo_gpu.actor.head;
            auto& ll_cpu = ppo.actor.head;
            rlt::copy(device_gpu, device, ll_gpu.log_std.parameters, ll_cpu.log_std.parameters);
            rlt::copy(device_gpu, device, ll_gpu.log_std.gradient, ll_cpu.log_std.gradient);
        }

        for(TI epoch_i = 0; epoch_i < N_EPOCHS; epoch_i++){
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

                auto gpu_state_obs_batch = rlt::view_range(device_gpu, gpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_state_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_state_obs_batch, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});
                auto gpu_actions_train_tensor = rlt::to_tensor(device_gpu, gpu_actions_train);
                auto gpu_actions_train_reshaped = rlt::reshape_row_major(device_gpu, gpu_actions_train_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});

                auto gpu_obs_batch = rlt::view_range(device_gpu, gpu_all_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                using ACTOR_INPUT_SHAPE2 = rlt::tensor::Prepend<rlt::tensor::Prepend<typename ENVIRONMENT::Observation::SHAPE, BATCH_SIZE>, (TI)1>;
                auto gpu_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_obs_batch, ACTOR_INPUT_SHAPE2{});
                rlt::forward(device_gpu, ppo_gpu.actor, gpu_obs_batch_reshaped, gpu_state_obs_batch_reshaped, gpu_actions_train_reshaped, actor_buffers, rng_gpu);
                cudaDeviceSynchronize();

                rlt::copy(device_gpu, device, gpu_actions_train, ppo_buffers.current_batch_actions);
                auto& last_layer_gpu = ppo_gpu.actor.head;
                auto& last_layer_cpu = ppo.actor.head;
                rlt::copy(device_gpu, device, last_layer_gpu.log_std.parameters, last_layer_cpu.log_std.parameters);
                rlt::copy(device_gpu, device, last_layer_gpu.log_std.gradient, last_layer_cpu.log_std.gradient);

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
                        action_log_prob += rlt::random::normal_distribution::log_prob(device.random, current_action, current_action_log_std, rollout_action);
                        rlt::set(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, rlt::random::normal_distribution::d_log_prob_d_mean(device.random, current_action, current_action_log_std, rollout_action));
                        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                            T d_entropy_loss_d_current_action_log_std = -(T)1/BATCH_SIZE * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT;
                            rlt::increment(device, last_layer_cpu.log_std.gradient, d_entropy_loss_d_current_action_log_std, action_i);
                            rlt::set(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i, rlt::random::normal_distribution::d_log_prob_d_log_std(device.random, current_action, current_action_log_std, rollout_action));
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
                rlt::copy(device, device_gpu, last_layer_cpu.log_std.parameters, last_layer_gpu.log_std.parameters);
                rlt::copy(device, device_gpu, last_layer_cpu.log_std.gradient, last_layer_gpu.log_std.gradient);
                rlt::copy(device, device_gpu, ppo_buffers.d_action_log_prob_d_action, gpu_d_action_train);
                auto gpu_d_action_tensor = rlt::to_tensor(device_gpu, gpu_d_action_train);
                auto gpu_d_action_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                rlt::backward(device_gpu, ppo_gpu.actor, gpu_obs_batch_reshaped, gpu_state_obs_batch_reshaped, gpu_d_action_reshaped, actor_buffers);
                cudaDeviceSynchronize();
                rlt::step(device_gpu, actor_optimizer_gpu, ppo_gpu.actor);
                cudaDeviceSynchronize();

                auto batch_obs_priv = rlt::view_range(device, dataset.all_observations_privileged, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto batch_obs_priv_matrix = rlt::matrix_view(device, batch_obs_priv);
                rlt::copy(device, device_gpu, batch_obs_priv_matrix, gpu_critic_obs);
                auto gpu_critic_obs_tensor = rlt::to_tensor(device_gpu, gpu_critic_obs);
                auto gpu_critic_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_critic_obs_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, OBS_PRIV_DIM>{});
                rlt::forward(device_gpu, ppo_gpu.critic, gpu_critic_obs_reshaped, critic_buffers, rng_gpu);
                cudaDeviceSynchronize();
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
#endif

        // Curriculum update
        if(ppo_step_i % CURRICULUM_EVAL_INTERVAL == 0 && ppo_step_i > 0 && episode_count > 0){
            T mean_episode_length = episode_length_accumulator / episode_count;
            T survival_share = mean_episode_length / (T)EPISODE_STEP_LIMIT;
            rlt::add_scalar(device, device.logger, "curriculum/survival_share", survival_share);
            rlt::add_scalar(device, device.logger, "curriculum/level", curriculum_level);
            rlt::add_scalar(device, device.logger, "curriculum/mean_episode_length", mean_episode_length);
            if(survival_share > (T)0.95){
                curriculum_level = rlt::math::min(device.math, (T)1, curriculum_level + (T)0.05);
            }
            T current_max_angle = CURRICULUM_MIN_START_ANGLE + curriculum_level * (CURRICULUM_FULL_MAX_ANGLE - CURRICULUM_MIN_START_ANGLE);
            T current_max_position = CURRICULUM_MIN_START_POSITION + curriculum_level * (CURRICULUM_FULL_MAX_POSITION - CURRICULUM_MIN_START_POSITION);
            for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                auto& env = rlt::get(on_policy_runner.environments, 0, env_i);
                env.dynamics.parameters.mdp.init.max_angle = current_max_angle;
                env.dynamics.parameters.mdp.init.max_position = current_max_position;
            }
            std::cout << "  Curriculum: level=" << std::setprecision(2) << curriculum_level
                      << " survival=" << std::setprecision(3) << survival_share
                      << " angle=" << std::setprecision(3) << current_max_angle
                      << " pos=" << std::setprecision(3) << current_max_position << std::endl;
            episode_length_accumulator = 0;
            episode_count = 0;
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
#ifdef RL_TOOLS_DISABLE_VISUAL
            auto& actor_log_std = rlt::get_last_layer(ppo.actor);
#else
            auto& actor_log_std = ppo.actor.head;
#endif
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                T log_std_val = rlt::get(device, actor_log_std.log_std.parameters, action_i);
                rlt::add_scalar(device, device.logger, "actor/log_std", log_std_val, 100);
            }
        }

        // Save trajectories to extrack
        if(ppo_step_i % TRAJECTORY_SAVE_INTERVAL == 0 && !completed_episodes.empty()){
            auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, on_policy_runner.step);
            auto& parameters_ref = rlt::get(on_policy_runner.env_parameters, 0, (TI)0);
            std::string trajectories_json = trajectory_episodes_to_json(device, envs[0], parameters_ref, completed_episodes, simulation_dt);
#ifdef RL_TOOLS_ENABLE_ZLIB
            std::vector<uint8_t> compressed;
            if(rlt::compress_zlib(trajectories_json, compressed)){
                std::ofstream f(step_folder / "trajectories.json.gz", std::ios::binary);
                f.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
            }
#else
            {
                std::ofstream f(step_folder / "trajectories.json");
                f << trajectories_json;
            }
#endif
            std::cout << "Saved " << completed_episodes.size() << " episodes to " << step_folder << std::endl;
            completed_episodes.clear();
        }
    }

    std::cout << "Training finished at step " << on_policy_runner.step << std::endl;

    // =========================================================================
    // Cleanup
    // =========================================================================
#ifndef RL_TOOLS_DISABLE_VISUAL
    for(TI env_i = 1; env_i < N_ENVIRONMENTS; env_i++){
        envs[env_i].renderer = nullptr;
        envs[env_i].scene = nullptr;
    }
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::free(device, envs[env_i]);
    }
#endif

    rlt::free(device, cpu_actions_eval);
    rlt::free(device, cpu_state_obs_step);
    rlt::free(device, ppo);
    rlt::free(device, ppo_buffers);
    rlt::free(device, on_policy_runner);
    rlt::free(device, dataset);
    rlt::free(device, actor_optimizer);
    rlt::free(device, critic_optimizer);
    rlt::free(device_gpu, actor_optimizer_gpu);
    rlt::free(device_gpu, critic_optimizer_gpu);

    rlt::free(device_gpu, ppo_gpu);
    rlt::free(device_gpu, actor_buffers);
    rlt::free(device_gpu, critic_buffers);
    rlt::free(device_gpu, critic_buffers_gae);
#ifndef RL_TOOLS_DISABLE_VISUAL
    rlt::free(device_gpu, gpu_all_observations);
#endif
    rlt::free(device_gpu, gpu_all_state_observations);
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
