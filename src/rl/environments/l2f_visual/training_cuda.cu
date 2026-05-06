// PPO asserts actor's INPUT_SHAPE last dim matches env's Observation last dim, but our
// actor takes a stacked-frame+target combined image (24 channels) while the env's
// Observation is the per-step student frame (3 channels). The macro name is misleading
// — it only disables the one assertion in ppo.h, not visual processing itself.
#define RL_TOOLS_DISABLE_VISUAL
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/random/operations_generic_array.h>
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

#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cuda.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/operations_generic.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cuda.h>
#include <rl_tools/nn/loss_functions/mse/operations_generic.h>
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>

#include <rl_tools/utils/extrack/operations_cpu.h>
#include <rl_tools/utils/zlib/operations_cpu.h>

#include <rl_tools/persist/backends/tar/operations_cpu.h>
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
#include <rl_tools/persist/backends/hdf5/hdf5.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#endif
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/flatten/persist.h>
#include <rl_tools/nn/layers/unflatten/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>
#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/matrix/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/conv2d/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/flatten/persist_code.h>
#include <rl_tools/nn/layers/unflatten/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/parallel/persist_code.h>

#include <array>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <string>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <vector>
#include <numeric>
#include <random>
#include <mutex>
#include <sstream>

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
using RNG = rlt::devices::generic::random::ArrayENGINE<rlt::devices::generic::random::ArraySpecification<TI, 1024>>;
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
struct DOMAIN_RANDOMIZATION_OPTIONS {
    static constexpr bool THRUST_TO_WEIGHT = true;
    static constexpr bool MASS = false;
    static constexpr bool TORQUE_TO_INERTIA = false;
    static constexpr bool MASS_SIZE_DEVIATION = false;
    static constexpr bool ROTOR_TORQUE_CONSTANT = false;
    static constexpr bool DISTURBANCE_FORCE = false;
    static constexpr bool ROTOR_TIME_CONSTANT = false;
};
using PARAMETERS_TYPE = l2f::ParametersDomainRandomization<l2f::ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>>>;

static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie;

static constexpr REWARD_FUNCTION reward_function = {
    false,
    0.10,
    1.00,
    -1.00,
    10.00,
    0.00,
    1.00,
    0.05,
    0.00,
    0.00,
    0.00,
    0.10,
    0.00,
    0.00
};
static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
    1.0, 0.0, 0.3, 1.0, 1.0, true, -1, +1,
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    true, 1.0, 0, 10, 35, 10000, 50000,
};
static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::Integration integration = {
    static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)
};
static constexpr typename PARAMETERS_TYPE::MDP mdp = { init, reward_function, {}, {}, termination };
static constexpr T DISTURBANCE_FORCE_STD = 0;
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = { {0, DISTURBANCE_FORCE_STD}, {0, 0} };
static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
    1.5, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static constexpr PARAMETERS_TYPE nominal_parameters = { {{dynamics, integration, mdp}, disturbances}, domain_randomization };

// =========================================================================
// Environment static parameters
// =========================================================================
static constexpr TI ACTION_HISTORY_LENGTH = 64;

struct STATIC_PARAMETERS {
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI CLOSED_FORM = false;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, STATE_BASE>>>>>>>>;
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

using ACTOR_STATE_OBS = obs::OrientationWorldZ<obs::OrientationWorldZSpecification<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI, obs::LinearAccelerationBodyFrameHistory<obs::LinearAccelerationBodyFrameHistorySpecification<T, TI, 1, obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>;
static constexpr TI STATE_OBS_DIM = ACTOR_STATE_OBS::DIM;

// =========================================================================
// Visual environment specification (multi-scene)
// =========================================================================
static constexpr TI N_TOTAL_SCENES = 25;
static constexpr TI N_ACTIVE_SCENES = 2;
static constexpr TI N_ENVIRONMENTS_PER_SCENE = 64;
static constexpr TI N_ENVIRONMENTS = N_ACTIVE_SCENES * N_ENVIRONMENTS_PER_SCENE;
static constexpr TI CAM_WIDTH = 80;
static constexpr TI CAM_HEIGHT = 50;
static constexpr TI NUM_PROBES = 64;
static constexpr T CAMERA_FOV = static_cast<T>(63.8) / static_cast<T>(180) * rlt::math::PI<T>;
static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = static_cast<T>(5.0) / static_cast<T>(180) * rlt::math::PI<T>;
static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = static_cast<T>(10.0) / static_cast<T>(180) * rlt::math::PI<T>;
constexpr bool HIGH_FIDELITY_SHADING = true;
static constexpr bool RENDER_ENABLE_MOTION_BLUR = false;
static constexpr TI RENDER_MOTION_BLUR_SAMPLES = 1;
static constexpr bool RENDER_ENABLE_ANTI_ALIASING = true;
static constexpr TI RENDER_ANTI_ALIASING_GRID_SIZE = 2;
static constexpr T RENDER_SHUTTER_FRACTION_MIN = static_cast<T>(0.25);
static constexpr T RENDER_SHUTTER_FRACTION_MAX = static_cast<T>(1);
static_assert(RENDER_SHUTTER_FRACTION_MIN >= static_cast<T>(0) && RENDER_SHUTTER_FRACTION_MIN <= RENDER_SHUTTER_FRACTION_MAX && RENDER_SHUTTER_FRACTION_MAX <= static_cast<T>(1), "Invalid l2f_visual training shutter fraction range");

using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, N_ENVIRONMENTS_PER_SCENE, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES, HIGH_FIDELITY_SHADING, RENDER_ENABLE_MOTION_BLUR, RENDER_MOTION_BLUR_SAMPLES, RENDER_ENABLE_ANTI_ALIASING, RENDER_ANTI_ALIASING_GRID_SIZE>;
using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;
using CAMERA_DATA = rlt::rendering::raytracing::CameraData<T>;
static constexpr bool RENDER_MOTION_BLUR_ACTIVE = ENVIRONMENT::SPEC::RENDERER_SPEC::ENABLE_MOTION_BLUR;
static constexpr bool RENDER_ANTI_ALIASING_ACTIVE = ENVIRONMENT::SPEC::RENDERER_SPEC::ENABLE_ANTI_ALIASING;

// Mosaic layout: each env cell shows (target | actual) pair, arranged in an ENV_GRID_SIDE×ENV_GRID_SIDE grid per active scene.
static constexpr TI ENV_GRID_SIDE = 8;
static constexpr TI SCENE_GRID_COLS = N_ACTIVE_SCENES;
static constexpr TI SCENE_GRID_ROWS = (N_ACTIVE_SCENES + SCENE_GRID_COLS - 1) / SCENE_GRID_COLS;
static_assert(ENV_GRID_SIDE * ENV_GRID_SIDE == N_ENVIRONMENTS_PER_SCENE, "ENV_GRID_SIDE^2 must equal N_ENVIRONMENTS_PER_SCENE for the mosaic layout");

// =========================================================================
// Frame stacking + target-channel concatenation
// =========================================================================
static constexpr TI FRAME_STACK_N = 10;
static constexpr TI FRAME_STACK_STRIDE = 10;
static constexpr TI ROLLOUT_STEPS_PER_ENV = 512;
static constexpr TI ROLLOUTS_PER_SCENE_SET = 1;
static constexpr TI FRAME_STACK_HISTORY_LENGTH = FRAME_STACK_STRIDE * (FRAME_STACK_N - 1) + ROLLOUT_STEPS_PER_ENV;
static constexpr TI STACKED_IMG_C = ENVIRONMENT::Observation::CHANNELS * FRAME_STACK_N;
static constexpr TI COMBINED_IMG_C_LOGICAL = STACKED_IMG_C + ENVIRONMENT::Observation::CHANNELS;
// Pad to multiple of 8 for cuDNN tensor-core fast path
static constexpr TI COMBINED_IMG_C = (COMBINED_IMG_C_LOGICAL + 7) & ~((TI)7);
static constexpr TI COMBINED_OBS_DIM = ENVIRONMENT::Observation::HEIGHT * ENVIRONMENT::Observation::WIDTH * COMBINED_IMG_C;
static constexpr TI INDOOR_POSITION_DIM = 3;
static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.5;
static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0.25;
static constexpr T OBSERVATION_NOISE_STD = 0.0;

// =========================================================================
// Trajectory recording for extrack UI
// =========================================================================
static constexpr TI EXTRACK_SAVE_INTERVAL_PPO_STEPS = 500;
static constexpr TI EXTRACK_SAVE_INTERVAL_SCENE_SETS_BASE = (EXTRACK_SAVE_INTERVAL_PPO_STEPS + ROLLOUTS_PER_SCENE_SET - 1) / ROLLOUTS_PER_SCENE_SET;
static constexpr TI EXTRACK_SAVE_INTERVAL_SCENE_SETS = 2 * EXTRACK_SAVE_INTERVAL_SCENE_SETS_BASE;
static constexpr TI VIDEO_SAVE_INTERVAL_SCENE_SETS = EXTRACK_SAVE_INTERVAL_SCENE_SETS;
static constexpr TI CHECKPOINT_CADENCE_SCENE_SETS = EXTRACK_SAVE_INTERVAL_SCENE_SETS;
static constexpr TI REWARD_COMPONENT_LOG_INTERVAL_PPO_STEPS = 100;
static constexpr bool EXPORT_CHECKPOINT_TAR = true;
static constexpr bool EXPORT_CHECKPOINT_CODE = false;
static constexpr TI N_EXAMPLES = 512;
static constexpr TI REDUCED_BATCH_SIZE = 2;
static_assert(REDUCED_BATCH_SIZE <= N_EXAMPLES);
static constexpr TI TRAJECTORY_NUM_ENVS = 10;
static constexpr TI TRAJECTORY_MAX_EPISODES = 10;

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
    TI max_len = 0;
    for(auto& ep : episodes) if(ep.size() > max_len) max_len = ep.size();
    std::string json = "[";
    for(TI ep_i = 0; ep_i < episodes.size(); ep_i++){
        auto& episode = episodes[ep_i];
        json += "{\"parameters\": " + rlt::json(device, env, parameters) + ",\n";
        json += "\"trajectory\": [";
        for(TI step_i = 0; step_i < max_len; step_i++){
            auto& s = (step_i < episode.size()) ? episode[step_i] : episode.back();
            json += "{\"state\":" + rlt::json(device, env, parameters, s.state) + ",";
            json += "\"action\":[";
            for(TI a = 0; a < ENVIRONMENT::ACTION_DIM; a++){
                json += std::to_string(s.actions[a]);
                if(a < ENVIRONMENT::ACTION_DIM - 1) json += ",";
            }
            json += "],";
            json += "\"dt\":" + std::to_string(dt) + ",";
            json += "\"reward\":" + std::to_string(s.reward) + ",";
            bool term = (step_i < episode.size()) ? s.terminated : true;
            json += "\"terminated\":" + (term ? std::string("true") : std::string("false"));
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
    static constexpr T ALPHA = 1e-4;
};

struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
    static constexpr TI BATCH_SIZE = 1024;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI ACTOR_CNN_CHANNEL_MULTIPLIER = 2;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = ROLLOUT_STEPS_PER_ENV;
    static constexpr TI N_ENVIRONMENTS = ::N_ENVIRONMENTS;
    static constexpr TI TOTAL_STEP_LIMIT = 1000000000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT / (N_ENVIRONMENTS * ON_POLICY_RUNNER_STEPS_PER_ENV) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using ACTOR_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
    using CRITIC_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
    static constexpr bool NORMALIZE_OBSERVATIONS = false; // standardize layers handle their own warmup
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.005;
        static constexpr TI N_EPOCHS = 2;
        static constexpr T GAMMA = 0.99;
        static constexpr T LAMBDA = 0.95;
        static constexpr T EPSILON_CLIP = 0.2;
        static constexpr T INITIAL_ACTION_STD = 0.5;
    };
};

// CNN actor (image branch + state branch + mlp head with unconditional log_std)
// dense critic on full privileged observation (asymmetric)
template<typename T_TYPE_POLICY, typename T_TI, typename T_ENVIRONMENT, typename PARAMETERS, bool T_DYNAMIC_ALLOCATION = true>
struct ConfigApproximators{
    static constexpr T_TI STEPS = 1;
    static constexpr T_TI FORWARD_BATCH_SIZE = PARAMETERS::BATCH_SIZE;

    static constexpr T_TI IMG_H = T_ENVIRONMENT::Observation::HEIGHT;
    static constexpr T_TI IMG_W = T_ENVIRONMENT::Observation::WIDTH;

    template <typename CAPABILITY>
    struct Actor{
        using IMAGE_INPUT_SHAPE = rlt::tensor::Shape<T_TI, STEPS, FORWARD_BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
        using STATE_INPUT_SHAPE = rlt::tensor::Shape<T_TI, STEPS, FORWARD_BATCH_SIZE, STATE_OBS_DIM>;

        // Image branch: Conv(32) -> Conv(64) -> Conv(128) -> Conv(256) stride-2 + Flatten + Dense(64)
        using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 16 * PARAMETERS::ACTOR_CNN_CHANNEL_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
        using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 32 * PARAMETERS::ACTOR_CNN_CHANNEL_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
        using CONV3_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 64 * PARAMETERS::ACTOR_CNN_CHANNEL_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV3 = rlt::nn::layers::conv2d::BindConfiguration<CONV3_CONFIG>;
        using CONV4_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 128 * PARAMETERS::ACTOR_CNN_CHANNEL_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV4 = rlt::nn::layers::conv2d::BindConfiguration<CONV4_CONFIG>;
        using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>;
        using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
        using IMAGE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, T_TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION>;
        using IMAGE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<IMAGE_DENSE_EMBED_CONFIG>;
        using IMAGE_BRANCH = rlt::nn_models::sequential::Module<CONV1, CONV2, CONV3, CONV4, OUTPUT_FLATTEN, IMAGE_DENSE_EMBED>;

        // State branch: Standardize + Dense(64)
        using STATE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
        using STATE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STATE_STANDARDIZE_CONFIG>;
        using STATE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, T_TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION>;
        using STATE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<STATE_DENSE_EMBED_CONFIG>;
        using STATE_BRANCH = rlt::nn_models::sequential::Module<STATE_STANDARDIZE, STATE_DENSE_EMBED>;

        // Head: MLP with unconditional stddev
        using MLP_HEAD_CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, T_ENVIRONMENT::ACTION_DIM, 3, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
        using MLP_HEAD = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_HEAD_CONFIG>;

        using BRANCH_IMAGE = rlt::nn_models::parallel::Branch<IMAGE_BRANCH, IMAGE_INPUT_SHAPE>;
        using BRANCH_STATE = rlt::nn_models::parallel::Branch<STATE_BRANCH, STATE_INPUT_SHAPE>;
        using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, MLP_HEAD, BRANCH_IMAGE, BRANCH_STATE>;
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

using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximators>;

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
using ACTOR_TYPE = typename LOOP_CORE_CONFIG::NN::ACTOR_TYPE;

// Rollout actor: forward-only with batch size N_ENVIRONMENTS
using CAPABILITY_ROLLOUT = rlt::nn::capability::Forward<true>;
using ROLLOUT_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<CAPABILITY_ROLLOUT>::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>;
using ROLLOUT_ACTOR_BUFFERS = typename ROLLOUT_ACTOR_TYPE::template Buffer<true>;
using CHECKPOINT_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<CAPABILITY_ROLLOUT>::template CHANGE_BATCH_SIZE<TI, N_EXAMPLES>;

// Constants
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
static_assert(N_EXAMPLES <= BATCH_SIZE, "N_EXAMPLES must fit the reusable combined-observation batch buffer");
static_assert(N_EXAMPLES <= STEPS_TOTAL, "N_EXAMPLES must fit one PPO rollout dataset");

static constexpr T EPISODE_END_REASON_NONE = 0;
static constexpr T EPISODE_END_REASON_TERMINATED = 1;
static constexpr T EPISODE_END_REASON_TIME_LIMIT = 2;
static constexpr T EPISODE_END_REASON_SCENE_BOUNDARY = 3;

// =========================================================================
// Custom CUDA kernels
// =========================================================================
namespace ppo_visual {
    using namespace rl_tools;

    __device__ CAMERA_DATA interpolate_camera(const CAMERA_DATA& from, const CAMERA_DATA& to, T alpha){
        CAMERA_DATA out;
        for(TI i = 0; i < 3; i++){
            out.pos[i] = from.pos[i] + (to.pos[i] - from.pos[i]) * alpha;
            out.dir_00[i] = from.dir_00[i] + (to.dir_00[i] - from.dir_00[i]) * alpha;
            out.dir_du[i] = from.dir_du[i] + (to.dir_du[i] - from.dir_du[i]) * alpha;
            out.dir_dv[i] = from.dir_dv[i] + (to.dir_dv[i] - from.dir_dv[i]) * alpha;
        }
        return out;
    }

    template<bool ENABLE_MOTION_BLUR, typename RENDERER>
    void set_active_scene_camera_open_buffer(void*& camera_open_buffer, RENDERER* renderer){
        if constexpr(ENABLE_MOTION_BLUR){
            camera_open_buffer = (void*)owlBufferGetPointer((OWLBuffer)renderer->backend.owl_cameras_open_buffer, 0);
        }
    }

    // Per-step prologue: handle episode reset (sample initial state, sample indoor position + scene yaw,
    // randomize brightness), observe state branch and privileged observations, record episode_start_step.
    template<typename DEVICE, typename OBS_PRIV_SPEC, typename STATE_OBS_SPEC, typename EPISODE_STAT_SPEC, typename RNG>
    __global__
    void prologue_kernel(
        DEVICE device,
        ENVIRONMENT* envs, typename ENVIRONMENT::Parameters* env_params, typename ENVIRONMENT::State* states,
        bool* truncated_arr, TI* episode_step_arr, T* episode_return_arr,
        T* episode_end_reason_arr,
        bool* render_reset_arr,
        T* shutter_fraction_arr,
        Tensor<OBS_PRIV_SPEC> observations_privileged,
        Tensor<STATE_OBS_SPEC> state_observations,
        Matrix<EPISODE_STAT_SPEC> episode_lengths_log,
        Matrix<EPISODE_STAT_SPEC> episode_returns_log,
        Matrix<EPISODE_STAT_SPEC> episode_end_reasons_log,
        T* brightness_scale_arr,
        T* target_brightness_scale_arr,
        T* target_frame_roll_arr,
        T* target_frame_pitch_arr,
        T* scene_translation_arr,
        T* scene_yaw_arr,
        T* scene_yaw_cos_arr,
        T* scene_yaw_sin_arr,
        T* indoor_positions_ptr, TI* num_indoor_positions_ptr, TI* env_scene_ptr, TI max_indoor_pos,
        TI* episode_start_step,
        RNG rng, TI step_i, TI frame_step_i
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& rng_state = get(rng.states, 0, env_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        TI pos = step_i * N_ENVIRONMENTS + env_i;
        bool reset_for_render = truncated_arr[env_i];
        render_reset_arr[env_i] = reset_for_render;
        if(reset_for_render){
            if(episode_step_arr[env_i] > 0){
                set(episode_lengths_log, pos, 0, (T)episode_step_arr[env_i]);
                set(episode_returns_log, pos, 0, episode_return_arr[env_i]);
                set(episode_end_reasons_log, pos, 0, episode_end_reason_arr[env_i]);
            } else {
                set(episode_lengths_log, pos, 0, (T)-1);
                set(episode_returns_log, pos, 0, (T)0);
                set(episode_end_reasons_log, pos, 0, EPISODE_END_REASON_NONE);
            }
            sample_initial_parameters(device, env, params, rng_state);
            sample_initial_state(device, env, params, state, rng_state);
            TI scene_idx = env_scene_ptr[env_i];
            TI num_pos = num_indoor_positions_ptr[scene_idx];
            TI pos_idx = random::uniform_int_distribution(device.random, (TI)0, num_pos - 1, rng_state);
            T* p = indoor_positions_ptr + (scene_idx * max_indoor_pos + pos_idx) * INDOOR_POSITION_DIM;
            scene_translation_arr[env_i * 3 + 0] = p[0];
            scene_translation_arr[env_i * 3 + 1] = p[1];
            scene_translation_arr[env_i * 3 + 2] = p[2];
            T scene_yaw = random::uniform_real_distribution(device.random, (T)0, (T)(2.0 * 3.14159265358979323846), rng_state);
            scene_yaw_arr[env_i] = scene_yaw;
            scene_yaw_cos_arr[env_i] = math::cos(device.math, scene_yaw);
            scene_yaw_sin_arr[env_i] = math::sin(device.math, scene_yaw);
            episode_step_arr[env_i] = 0;
            episode_return_arr[env_i] = (T)0;
            truncated_arr[env_i] = false;
            episode_end_reason_arr[env_i] = EPISODE_END_REASON_NONE;
            brightness_scale_arr[env_i] = (T)1 + (random::uniform_real_distribution(device.random, (T)0, (T)1, rng_state) * (T)2 - (T)1) * BRIGHTNESS_RANDOMIZATION_RANGE;
            if constexpr(TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE > static_cast<T>(0)){
                T mismatch = (T)1 + (random::uniform_real_distribution(device.random, (T)0, (T)1, rng_state) * (T)2 - (T)1) * TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE;
                target_brightness_scale_arr[env_i] = brightness_scale_arr[env_i] * mismatch;
            } else {
                target_brightness_scale_arr[env_i] = brightness_scale_arr[env_i];
            }
            if constexpr(RENDER_MOTION_BLUR_ACTIVE){
                shutter_fraction_arr[env_i] = random::uniform_real_distribution(device.random, RENDER_SHUTTER_FRACTION_MIN, RENDER_SHUTTER_FRACTION_MAX, rng_state);
            }
            if constexpr(TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE > static_cast<T>(0)){
                target_frame_roll_arr[env_i] = random::uniform_real_distribution(device.random, -TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, rng_state);
                target_frame_pitch_arr[env_i] = random::uniform_real_distribution(device.random, -TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, rng_state);
            } else {
                target_frame_roll_arr[env_i] = (T)0;
                target_frame_pitch_arr[env_i] = (T)0;
            }
            episode_start_step[env_i] = frame_step_i;
        } else {
            set(episode_lengths_log, pos, 0, (T)-1);
            set(episode_returns_log, pos, 0, (T)0);
            set(episode_end_reasons_log, pos, 0, EPISODE_END_REASON_NONE);
        }
        // Privileged observation (full state)
        {
            auto obs_priv_slice = view(device, observations_privileged, env_i);
            auto obs_priv_flat = view_memory<tensor::Shape<TI, ENVIRONMENT::ObservationPrivileged::DIM>>(device, obs_priv_slice);
            auto obs_priv_matrix = matrix_view(device, obs_priv_flat);
            observe(device, env.dynamics, params.dynamics, state, typename ENVIRONMENT::ObservationPrivileged{}, obs_priv_matrix, rng_state);
        }
        // State observation (reduced — actor's state branch)
        {
            auto state_obs_slice = view(device, state_observations, env_i);
            auto state_obs_matrix = matrix_view(device, state_obs_slice);
            observe(device, env.dynamics, params.dynamics, state, ACTOR_STATE_OBS{}, state_obs_matrix, rng_state);
        }
    }

    // Per-step epilogue: sample noisy action from N(action_mean, exp(log_std)), store action_log_prob,
    // step env, compute reward, terminated/truncated.
    template<typename DEVICE, typename ACTIONS_MEAN_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RUNNER_SPEC, typename RNG>
    __global__
    void epilogue_kernel(
        DEVICE device,
        ENVIRONMENT* envs, typename ENVIRONMENT::Parameters* env_params, typename ENVIRONMENT::State* states,
        bool* truncated_arr, TI* episode_step_arr, T* episode_return_arr,
        T* episode_end_reason_arr,
        Matrix<ACTIONS_MEAN_SPEC> actions_mean,
        Matrix<ACTIONS_SPEC> actions,
        Matrix<ACTION_LOG_STD_SPEC> action_log_std,
        rl::components::on_policy_runner::Dataset<RUNNER_SPEC> dataset,
        RNG rng, TI step_i, TI episode_step_limit
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& rng_state = get(rng.states, 0, env_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        TI pos = step_i * N_ENVIRONMENTS + env_i;

        T action_log_prob = 0;
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T action_mean = get(actions_mean, env_i, action_i);
            T current_action_log_std = get(action_log_std, 0, action_i);
            T action_std = math::exp(device.math, current_action_log_std);
            T action_noisy = random::normal_distribution::sample(device.random, action_mean, action_std, rng_state);
            action_log_prob += random::normal_distribution::log_prob(device.random, action_mean, current_action_log_std, action_noisy);
            set(actions, env_i, action_i, action_noisy);
        }
        set(dataset.action_log_probs, pos, 0, action_log_prob);

        typename ENVIRONMENT::State next_state;
        auto action_row = row(device, actions, env_i);
        step(device, env.dynamics, params.dynamics, state, action_row, next_state, rng_state);
        bool terminated_flag = terminated(device, env.dynamics, params.dynamics, next_state, rng_state);
        T reward_value = reward(device, env.dynamics, params.dynamics, state, action_row, next_state, rng_state);
        episode_return_arr[env_i] += reward_value;
        episode_step_arr[env_i]++;
        bool time_limit_flag = episode_step_limit > 0 && episode_step_arr[env_i] >= episode_step_limit;
        bool trunc = terminated_flag || time_limit_flag;
        if(trunc){
            episode_end_reason_arr[env_i] = terminated_flag ? EPISODE_END_REASON_TERMINATED : EPISODE_END_REASON_TIME_LIMIT;
        }
        truncated_arr[env_i] = trunc;
        set(dataset.terminated, pos, 0, terminated_flag);
        set(dataset.rewards, pos, 0, reward_value);
        set(dataset.truncated, pos, 0, trunc);
        TI pos_reset = pos + N_ENVIRONMENTS;
        set(dataset.all_reset, pos_reset, 0, trunc);
        state = next_state;
    }

    __global__ void force_scene_boundary_reset_kernel(bool* truncated_arr, TI* episode_step_arr, T* episode_end_reason_arr){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        if(!truncated_arr[env_i] && episode_step_arr[env_i] > 0){
            episode_end_reason_arr[env_i] = EPISODE_END_REASON_SCENE_BOUNDARY;
        }
        truncated_arr[env_i] = true;
    }

    template<typename DEVICE, typename OBS_PRIV_SPEC, typename RNG>
    __global__
    void final_priv_obs_kernel(
        DEVICE device,
        ENVIRONMENT* envs, typename ENVIRONMENT::Parameters* env_params, typename ENVIRONMENT::State* states,
        Tensor<OBS_PRIV_SPEC> observations_privileged,
        RNG rng
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& rng_state = get(rng.states, 0, env_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        auto obs_priv_slice = view(device, observations_privileged, env_i);
        auto obs_priv_flat = view_memory<tensor::Shape<TI, ENVIRONMENT::ObservationPrivileged::DIM>>(device, obs_priv_slice);
        auto obs_priv_matrix = matrix_view(device, obs_priv_flat);
        observe(device, env.dynamics, params.dynamics, state, typename ENVIRONMENT::ObservationPrivileged{}, obs_priv_matrix, rng_state);
    }

    template<typename DEVICE>
    __global__
    void make_cameras_kernel(
        DEVICE device,
        typename ENVIRONMENT::Parameters* env_params, typename ENVIRONMENT::State* states,
        CAMERA_DATA* gpu_cameras,
        CAMERA_DATA* gpu_cameras_open,
        CAMERA_DATA* gpu_prev_cameras,
        const bool* render_reset_arr,
        const T* shutter_fraction_arr,
        TI frame_step_i,
        T aspect,
        T* scene_translation_arr, T* scene_yaw_cos_arr, T* scene_yaw_sin_arr
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& state = states[env_i];
        const auto& params = env_params[env_i];
        CAMERA_DATA close_camera = rl::environments::l2f_visual::cuda::make_camera_for_state<DEVICE, VISUAL_SPEC>(
            device, params, state, aspect,
            scene_translation_arr + env_i * 3,
            scene_yaw_cos_arr[env_i], scene_yaw_sin_arr[env_i]
        );
        gpu_cameras[env_i] = close_camera;
        if constexpr(RENDER_MOTION_BLUR_ACTIVE){
            CAMERA_DATA open_camera = close_camera;
            if(frame_step_i > 0 && !render_reset_arr[env_i]){
                T shutter_fraction = shutter_fraction_arr[env_i];
                open_camera = interpolate_camera(close_camera, gpu_prev_cameras[env_i], shutter_fraction);
            }
            gpu_cameras_open[env_i] = open_camera;
            gpu_prev_cameras[env_i] = close_camera;
        }
    }

    template<typename DEVICE>
    __global__
    void make_target_cameras_kernel(
        DEVICE device,
        typename ENVIRONMENT::Parameters* env_params,
        CAMERA_DATA* target_cameras,
        T aspect,
        const T* target_frame_roll_arr,
        const T* target_frame_pitch_arr,
        T* scene_translation_arr, T* scene_yaw_cos_arr, T* scene_yaw_sin_arr
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        const auto& params = env_params[env_i];
        target_cameras[env_i] = rl::environments::l2f_visual::cuda::make_target_camera<DEVICE, VISUAL_SPEC>(
            device, params, aspect,
            scene_translation_arr + env_i * 3,
            scene_yaw_cos_arr[env_i], scene_yaw_sin_arr[env_i],
            target_frame_roll_arr[env_i], target_frame_pitch_arr[env_i]
        );
    }
}

// =========================================================================
// Pixel scatter / brightness / noise kernels
// =========================================================================
template <bool APPLY_BRIGHTNESS>
__global__ void scatter_pixel_to_float_kernel(
    const uint32_t* __restrict__ fb, float* __restrict__ output,
    const float* __restrict__ brightness_scales,
    int base_env, int n_envs,
    int pixels_per_camera, int obs_dim
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_envs * pixels_per_camera;
    if(tid >= total) return;
    int local_env = tid / pixels_per_camera;
    int pixel_idx = tid % pixels_per_camera;
    int fb_pixel = local_env * pixels_per_camera + pixel_idx;
    uint32_t rgba = fb[fb_pixel];
    float r = static_cast<float>((rgba >>  0) & 0xFF) / 255.0f;
    float g = static_cast<float>((rgba >>  8) & 0xFF) / 255.0f;
    float b = static_cast<float>((rgba >> 16) & 0xFF) / 255.0f;
    int global_env = base_env + local_env;
    if constexpr(APPLY_BRIGHTNESS){
        float scale = brightness_scales[global_env];
        r = fminf(fmaxf(r * scale, 0.0f), 1.0f);
        g = fminf(fmaxf(g * scale, 0.0f), 1.0f);
        b = fminf(fmaxf(b * scale, 0.0f), 1.0f);
    }
    int out_base = global_env * obs_dim + pixel_idx * 3;
    output[out_base + 0] = r;
    output[out_base + 1] = g;
    output[out_base + 2] = b;
}

// Build combined (stacked frames + target frame) batch for the rollout step (N_ENVIRONMENTS rows).
__global__ void build_frame_stacked_with_target_from_history_kernel(
    const float* __restrict__ history_obs,
    const float* __restrict__ target_obs,
    const TI* __restrict__ episode_start_step,
    TI step_i,
    float* __restrict__ combined_out,
    int obs_dim, int img_c, int n_frames, int frame_stride, int combined_img_c, int combined_obs_dim, int num_envs
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx >= num_envs * combined_obs_dim) return;
    int sample = global_idx / combined_obs_dim;
    int offset = global_idx % combined_obs_dim;
    int pixel = offset / combined_img_c;
    int frame_channel = offset % combined_img_c;
    int logical_channels = n_frames * img_c + img_c;
    if(frame_channel < n_frames * img_c){
        int frame = frame_channel / img_c;
        int channel = frame_channel % img_c;
        TI episode_start = episode_start_step[sample];
        TI back = static_cast<TI>(frame) * static_cast<TI>(frame_stride);
        TI desired_step = step_i >= back ? step_i - back : episode_start;
        if(desired_step < episode_start){
            desired_step = episode_start;
        }
        TI history_slot = desired_step % FRAME_STACK_HISTORY_LENGTH;
        TI src_row = history_slot * num_envs + sample;
        combined_out[global_idx] = history_obs[src_row * obs_dim + pixel * img_c + channel];
    } else if(frame_channel < logical_channels) {
        int channel = frame_channel - n_frames * img_c;
        combined_out[global_idx] = target_obs[sample * obs_dim + pixel * img_c + channel];
    } else {
        combined_out[global_idx] = 0.0f;
    }
}

// Build combined batch for training using row indices: each batch row corresponds
// to (step_i, env_i). For each frame slot we look up the prior frame in the
// student observation history (clamped to episode start). Target frame is taken at
// the current row's step.
__global__ void build_frame_stacked_with_target_from_dataset_kernel(
    const float* __restrict__ history_obs,
    const float* __restrict__ all_target_obs,
    const TI* __restrict__ episode_start_step_per_row,
    float* __restrict__ combined_out,
    int obs_dim, int img_c, int n_frames, int frame_stride, int combined_img_c, int combined_obs_dim,
    TI frame_step_start, int batch_offset, int batch_size, int n_envs
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx >= batch_size * combined_obs_dim) return;
    int sample = global_idx / combined_obs_dim;
    int offset = global_idx % combined_obs_dim;
    int pixel = offset / combined_img_c;
    int frame_channel = offset % combined_img_c;
    int row = batch_offset + sample;
    int env_i = row % n_envs;
    int step_i_local = row / n_envs;
    int logical_channels = n_frames * img_c + img_c;
    if(frame_channel < n_frames * img_c){
        int frame = frame_channel / img_c;
        int channel = frame_channel % img_c;
        TI episode_start = episode_start_step_per_row[row];
        TI back = static_cast<TI>(frame) * static_cast<TI>(frame_stride);
        TI frame_step_i = frame_step_start + static_cast<TI>(step_i_local);
        TI desired_step = frame_step_i >= back ? frame_step_i - back : episode_start;
        if(desired_step < episode_start) desired_step = episode_start;
        TI history_slot = desired_step % FRAME_STACK_HISTORY_LENGTH;
        TI src_row = history_slot * n_envs + env_i;
        combined_out[global_idx] = history_obs[src_row * obs_dim + pixel * img_c + channel];
    } else if(frame_channel < logical_channels) {
        int channel = frame_channel - n_frames * img_c;
        TI src_row = (TI)row;
        combined_out[global_idx] = all_target_obs[src_row * obs_dim + pixel * img_c + channel];
    } else {
        combined_out[global_idx] = 0.0f;
    }
}

__global__ void record_episode_start_kernel(TI* episode_start_step, TI* episode_start_step_per_row, TI step_i){
    TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
    if(env_i >= N_ENVIRONMENTS) return;
    episode_start_step_per_row[step_i * N_ENVIRONMENTS + env_i] = episode_start_step[env_i];
}

// =========================================================================
// Main
// =========================================================================
static bool parse_hex_hash(const char* hex, unsigned char* out, unsigned len){
    for(unsigned i = 0; i < len; i++){
        unsigned byte = 0;
        for(int nibble = 0; nibble < 2; nibble++){
            char c = hex[i * 2 + nibble];
            if(c >= '0' && c <= '9') byte = (byte << 4) | (c - '0');
            else if(c >= 'a' && c <= 'f') byte = (byte << 4) | (c - 'a' + 10);
            else if(c >= 'A' && c <= 'F') byte = (byte << 4) | (c - 'A' + 10);
            else return false;
        }
        out[i] = static_cast<unsigned char>(byte);
    }
    return true;
}

int main(int argc, char** argv){
    TI seed = 0;
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <scene_directory or scene.glb> [seed]" << std::endl;
        return 1;
    }
    if(argc > 2){
        seed = std::atoi(argv[2]);
    }

    // ---------------------------------------------------------------------
    // Resolve scenes
    // ---------------------------------------------------------------------
    std::vector<std::string> scene_paths;
    std::mt19937 scene_rng(seed);
    const char* scene_arg = argv[1];
    if(std::filesystem::is_directory(scene_arg)){
        std::vector<std::string> all_glbs;
        for(auto& entry : std::filesystem::directory_iterator(scene_arg)){
            if(entry.path().extension() == ".glb"){
                all_glbs.push_back(entry.path().string());
            }
        }
        std::sort(all_glbs.begin(), all_glbs.end(), [](const std::string& a, const std::string& b){
            auto extract_number = [](const std::string& path) -> int {
                auto filename = std::filesystem::path(path).stem().string();
                auto pos = filename.rfind('-');
                if(pos != std::string::npos){
                    try { return std::stoi(filename.substr(pos + 1)); } catch(...) {}
                }
                return 0;
            };
            return extract_number(a) < extract_number(b);
        });
        if(static_cast<TI>(all_glbs.size()) < N_TOTAL_SCENES){
            std::cerr << "Need at least " << N_TOTAL_SCENES << " GLB scenes, found " << all_glbs.size() << std::endl;
            return 1;
        }
        for(TI i = 0; i < N_TOTAL_SCENES; i++) scene_paths.push_back(all_glbs[i]);
        std::cout << "Selected " << scene_paths.size() << " scenes from " << scene_arg << std::endl;
    } else {
        scene_paths.resize(N_TOTAL_SCENES, scene_arg);
        std::cout << "Replicating single scene across " << N_TOTAL_SCENES << " renderers: " << scene_arg << std::endl;
    }

    // ---------------------------------------------------------------------
    // Devices + extrack
    // ---------------------------------------------------------------------
    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device);

    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = "l2f_visual_training_cuda";
    rlt::init(device, extrack_config, extrack_paths, seed);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    RNG reward_log_rng;
    rlt::malloc(device, reward_log_rng);
    rlt::init(device, reward_log_rng, seed + 0xBADC0DE);

    // ---------------------------------------------------------------------
    // PPO components (CPU)
    // ---------------------------------------------------------------------
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

    // ---------------------------------------------------------------------
    // Load scenes (per-scene renderer + procthor scene)
    // ---------------------------------------------------------------------
    using RENDERER_TYPE = rlt::rendering::raytracing::Renderer<typename ENVIRONMENT::SPEC::RENDERER_SPEC>;
    using SCENE_TYPE = rlt::rendering::raytracing::scene::procthor::Scene<typename ENVIRONMENT::SPEC::SCENE_SPEC>;

    std::vector<ENVIRONMENT> envs(N_ENVIRONMENTS);
    std::vector<typename ENVIRONMENT::Parameters> env_parameters(N_ENVIRONMENTS);

    std::array<RENDERER_TYPE*, N_TOTAL_SCENES> renderers{};
    std::array<SCENE_TYPE*, N_TOTAL_SCENES> scenes{};
    {
        ENVIRONMENT loader_env;
        rlt::malloc(device, loader_env);
        for(TI s = 0; s < N_TOTAL_SCENES; s++){
            std::cout << "Loading scene [" << s << "]: " << std::filesystem::path(scene_paths[s]).filename().string() << std::flush;
            if(s > 0){
                loader_env.renderer = new RENDERER_TYPE{};
                rlt::malloc(device, *loader_env.renderer);
                loader_env.owns_renderer = true;
                loader_env.scene = new SCENE_TYPE{};
            }
            loader_env.scene_path = scene_paths[s].c_str();
            loader_env.renderer_initialized = false;
            rlt::init(device, loader_env);
            TI num_pos = loader_env.scene->num_indoor_positions;
            std::cout << " — " << num_pos << " indoor positions" << std::endl;
            if(num_pos == 0){
                std::cerr << "Scene has no valid positions with 1m clearance: " << scene_paths[s] << std::endl;
                return 1;
            }
            renderers[s] = loader_env.renderer;
            scenes[s] = loader_env.scene;
            loader_env.renderer = nullptr;
            loader_env.scene = nullptr;
            loader_env.owns_renderer = false;
        }
        std::cout << "Loaded " << N_TOTAL_SCENES << " scenes" << std::endl;
    }

    // GPU buffers for indoor positions
    static constexpr TI MAX_INDOOR_POS = 256;
    T* gpu_indoor_positions = nullptr;
    TI* gpu_num_indoor_positions = nullptr;
    TI* gpu_env_scene = nullptr;
    cudaMalloc(&gpu_indoor_positions, N_TOTAL_SCENES * MAX_INDOOR_POS * INDOOR_POSITION_DIM * sizeof(T));
    cudaMalloc(&gpu_num_indoor_positions, N_TOTAL_SCENES * sizeof(TI));
    cudaMalloc(&gpu_env_scene, N_ENVIRONMENTS * sizeof(TI));
    {
        std::vector<T> all_positions(N_TOTAL_SCENES * MAX_INDOOR_POS * INDOOR_POSITION_DIM, 0);
        std::vector<TI> all_counts(N_TOTAL_SCENES);
        for(TI s = 0; s < N_TOTAL_SCENES; s++){
            all_counts[s] = scenes[s]->num_indoor_positions;
            for(TI i = 0; i < all_counts[s]; i++){
                TI base = (s * MAX_INDOOR_POS + i) * INDOOR_POSITION_DIM;
                all_positions[base + 0] = scenes[s]->indoor_positions[i].position[0];
                all_positions[base + 1] = scenes[s]->indoor_positions[i].position[1];
                all_positions[base + 2] = scenes[s]->indoor_positions[i].position[2];
            }
        }
        cudaMemcpy(gpu_indoor_positions, all_positions.data(), all_positions.size() * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_num_indoor_positions, all_counts.data(), N_TOTAL_SCENES * sizeof(TI), cudaMemcpyHostToDevice);
    }

    std::array<TI, N_ACTIVE_SCENES> active_scene_indices{};
    std::vector<TI> scene_permutation(N_TOTAL_SCENES);
    std::iota(scene_permutation.begin(), scene_permutation.end(), 0);
    auto upload_active_scenes = [&](){
        std::array<TI, N_ENVIRONMENTS> env_scene{};
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            TI actual_scene_i = active_scene_indices[active_scene_i];
            for(TI local_env_i = 0; local_env_i < N_ENVIRONMENTS_PER_SCENE; local_env_i++){
                env_scene[active_scene_i * N_ENVIRONMENTS_PER_SCENE + local_env_i] = actual_scene_i;
            }
        }
        cudaMemcpy(gpu_env_scene, env_scene.data(), N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
    };
    for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
        active_scene_indices[active_scene_i] = active_scene_i;
    }
    upload_active_scenes();
    for(TI scene_i = 0; scene_i < N_TOTAL_SCENES; scene_i++){
        rlt::set_cameras(device, *renderers[scene_i], renderers[scene_i]->cameras);
    }

    // Set up envs (scene_i = env_i / N_ENVIRONMENTS_PER_SCENE for initial assignment)
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        TI scene_i = env_i / N_ENVIRONMENTS_PER_SCENE;
        rlt::malloc(device, envs[env_i].dynamics);
        envs[env_i].renderer = renderers[scene_i];
        envs[env_i].scene = scenes[scene_i];
        envs[env_i].owns_renderer = false;
        envs[env_i].renderer_initialized = true;
        envs[env_i].use_target_mode = true;
        envs[env_i].parameters.fov = CAMERA_FOV;
        envs[env_i].parameters.camera_randomization.fov_range = CAMERA_FOV_RANDOMIZATION_RANGE;
        rlt::initial_parameters(device, envs[env_i], env_parameters[env_i]);
        env_parameters[env_i].scene_translation[0] = 0;
        env_parameters[env_i].scene_translation[1] = 0;
        env_parameters[env_i].scene_translation[2] = 0;
    }

    // Write the env UI for extrack visualization
    {
        std::string ui = rlt::get_ui(device, envs[0].dynamics);
        if(!ui.empty()){
            std::filesystem::create_directories(extrack_paths.seed);
            std::ofstream ui_file(extrack_paths.seed / "ui.esm.js");
            ui_file << ui;
        }
    }

    // ---------------------------------------------------------------------
    // PPO init (CPU)
    // ---------------------------------------------------------------------
    rlt::init(device, ppo, actor_optimizer, critic_optimizer, rng);
    rlt::set_all(device, on_policy_runner.episode_step, 0);
    rlt::set_all(device, on_policy_runner.episode_return, (T)0);
    rlt::set_all(device, on_policy_runner.truncated, true);
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::set(on_policy_runner.environments, 0, env_i, envs[env_i]);
        rlt::set(on_policy_runner.env_parameters, 0, env_i, env_parameters[env_i]);
    }

    // ---------------------------------------------------------------------
    // GPU init
    // ---------------------------------------------------------------------
    rlt::init(device_gpu);
    RNG_GPU rng_gpu;
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_gpu, rng_gpu, seed);

    PPO_TYPE ppo_gpu;
    ACTOR_BUFFERS actor_buffers;
    CRITIC_BUFFERS critic_buffers;
    CRITIC_BUFFERS_GAE critic_buffers_gae;
    ON_POLICY_RUNNER_TYPE on_policy_runner_gpu;
    ON_POLICY_RUNNER_DATASET_TYPE dataset_gpu;
    rlt::malloc(device_gpu, ppo_gpu);
    rlt::malloc(device_gpu, actor_buffers);
    rlt::malloc(device_gpu, critic_buffers);
    rlt::malloc(device_gpu, critic_buffers_gae);
    rlt::malloc(device_gpu, on_policy_runner_gpu);
    rlt::malloc(device_gpu, dataset_gpu);

    // Rollout actor (forward only, batch = N_ENVIRONMENTS)
    ROLLOUT_ACTOR_TYPE rollout_actor_gpu;
    ROLLOUT_ACTOR_BUFFERS rollout_actor_buffers;
    rlt::malloc(device_gpu, rollout_actor_gpu);
    rlt::malloc(device_gpu, rollout_actor_buffers);

    // GPU optimizers
    ACTOR_OPTIMIZER actor_optimizer_gpu;
    CRITIC_OPTIMIZER critic_optimizer_gpu;
    rlt::malloc(device_gpu, actor_optimizer_gpu);
    rlt::malloc(device_gpu, critic_optimizer_gpu);
    rlt::copy(device, device_gpu, ppo, ppo_gpu);
    rlt::copy(device, device_gpu, actor_optimizer, actor_optimizer_gpu);
    rlt::copy(device, device_gpu, critic_optimizer, critic_optimizer_gpu);
    rlt::copy(device, device_gpu, ppo.actor, rollout_actor_gpu);
    rlt::reset_optimizer_state(device_gpu, actor_optimizer_gpu, ppo_gpu.actor);
    rlt::reset_optimizer_state(device_gpu, critic_optimizer_gpu, ppo_gpu.critic);

    // Initialize GPU on_policy_runner state
    rlt::set_all(device_gpu, on_policy_runner_gpu.truncated, true);
    rlt::set_all(device_gpu, on_policy_runner_gpu.episode_step, (TI)0);
    rlt::set_all(device_gpu, on_policy_runner_gpu.episode_return, (T)0);
    cudaMemcpy(on_policy_runner_gpu.environments._data, on_policy_runner.environments._data,
               N_ENVIRONMENTS * sizeof(ENVIRONMENT), cudaMemcpyHostToDevice);
    cudaMemcpy(on_policy_runner_gpu.env_parameters._data, on_policy_runner.env_parameters._data,
               N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::Parameters), cudaMemcpyHostToDevice);
    cudaMemcpy(on_policy_runner_gpu.states._data, on_policy_runner.states._data,
               N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::State), cudaMemcpyHostToDevice);
    on_policy_runner_gpu.step = on_policy_runner.step;

    // ---------------------------------------------------------------------
    // GPU-resident environment state (separate from on_policy_runner so kernels can mutate freely)
    // ---------------------------------------------------------------------
    ENVIRONMENT* gpu_envs_arr = nullptr;
    typename ENVIRONMENT::Parameters* gpu_params_arr = nullptr;
    typename ENVIRONMENT::State* gpu_states_arr = nullptr;
    bool* gpu_truncated_arr = nullptr;
    TI* gpu_episode_step_arr = nullptr;
    T* gpu_episode_return_arr = nullptr;
    T* gpu_episode_end_reason_arr = nullptr;
    T* gpu_brightness_scale_arr = nullptr;
    T* gpu_target_brightness_scale_arr = nullptr;
    T* gpu_target_frame_roll_arr = nullptr;
    T* gpu_target_frame_pitch_arr = nullptr;
    T* gpu_scene_translation_arr = nullptr;
    T* gpu_scene_yaw_arr = nullptr;
    T* gpu_scene_yaw_cos_arr = nullptr;
    T* gpu_scene_yaw_sin_arr = nullptr;
    bool* gpu_render_reset_arr = nullptr;
    T* gpu_shutter_fraction_arr = nullptr;
    cudaMalloc(&gpu_envs_arr, N_ENVIRONMENTS * sizeof(ENVIRONMENT));
    cudaMalloc(&gpu_params_arr, N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::Parameters));
    cudaMalloc(&gpu_states_arr, N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::State));
    cudaMalloc(&gpu_truncated_arr, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_step_arr, N_ENVIRONMENTS * sizeof(TI));
    cudaMalloc(&gpu_episode_return_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_episode_end_reason_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_brightness_scale_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_target_brightness_scale_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_target_frame_roll_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_target_frame_pitch_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_scene_translation_arr, N_ENVIRONMENTS * 3 * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_cos_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_sin_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_render_reset_arr, N_ENVIRONMENTS * sizeof(bool));
    if constexpr(RENDER_MOTION_BLUR_ACTIVE){
        cudaMalloc(&gpu_shutter_fraction_arr, N_ENVIRONMENTS * sizeof(T));
    }
    {
        cudaMemcpy(gpu_envs_arr, envs.data(), N_ENVIRONMENTS * sizeof(ENVIRONMENT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_params_arr, env_parameters.data(), N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::Parameters), cudaMemcpyHostToDevice);
        std::vector<unsigned char> init_truncated(N_ENVIRONMENTS, 1);
        std::vector<TI> init_step(N_ENVIRONMENTS, 0);
        std::vector<T> init_return(N_ENVIRONMENTS, (T)0);
        std::vector<T> init_end_reason(N_ENVIRONMENTS, EPISODE_END_REASON_NONE);
        std::vector<T> init_brightness(N_ENVIRONMENTS, (T)1);
        std::vector<T> init_target_brightness(N_ENVIRONMENTS, (T)1);
        std::vector<T> init_target_roll(N_ENVIRONMENTS, (T)0);
        std::vector<T> init_target_pitch(N_ENVIRONMENTS, (T)0);
        std::vector<T> init_translation(N_ENVIRONMENTS * 3, (T)0);
        std::vector<T> init_yaw(N_ENVIRONMENTS, (T)0);
        std::vector<T> init_cos(N_ENVIRONMENTS, (T)1);
        std::vector<T> init_sin(N_ENVIRONMENTS, (T)0);
        cudaMemcpy(gpu_truncated_arr, init_truncated.data(), N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_step_arr, init_step.data(), N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_return_arr, init_return.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_end_reason_arr, init_end_reason.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_brightness_scale_arr, init_brightness.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_target_brightness_scale_arr, init_target_brightness.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_target_frame_roll_arr, init_target_roll.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_target_frame_pitch_arr, init_target_pitch.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_scene_translation_arr, init_translation.data(), N_ENVIRONMENTS * 3 * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_scene_yaw_arr, init_yaw.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_scene_yaw_cos_arr, init_cos.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_scene_yaw_sin_arr, init_sin.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
    }

    // ---------------------------------------------------------------------
    // Auxiliary GPU buffers for visual pipeline
    // ---------------------------------------------------------------------
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, OBSERVATION_DIM>>> gpu_all_target_observations;
    rlt::malloc(device_gpu, gpu_all_target_observations);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, FRAME_STACK_HISTORY_LENGTH * N_ENVIRONMENTS, OBSERVATION_DIM>>> gpu_frame_stack_history;
    rlt::malloc(device_gpu, gpu_frame_stack_history);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, COMBINED_OBS_DIM>>> gpu_rollout_combined;
    rlt::malloc(device_gpu, gpu_rollout_combined);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, COMBINED_OBS_DIM>>> gpu_combined_batch;
    rlt::malloc(device_gpu, gpu_combined_batch);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, STATE_OBS_DIM>>> gpu_all_state_observations;
    rlt::malloc(device_gpu, gpu_all_state_observations);

    rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ACTION_DIM>> gpu_actions_eval;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_actions_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_d_action_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, OBS_PRIV_DIM>> gpu_critic_obs;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, 1>> gpu_d_critic_output;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL_ALL, OBS_PRIV_DIM>> gpu_gae_obs;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL_ALL, 1>> gpu_gae_values;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> gpu_episode_lengths_log;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> gpu_episode_returns_log;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> gpu_episode_end_reasons_log;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> cpu_episode_lengths_log;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> cpu_episode_returns_log;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> cpu_episode_end_reasons_log;
    rlt::malloc(device_gpu, gpu_actions_eval);
    rlt::malloc(device_gpu, gpu_actions_train);
    rlt::malloc(device_gpu, gpu_d_action_train);
    rlt::malloc(device_gpu, gpu_critic_obs);
    rlt::malloc(device_gpu, gpu_d_critic_output);
    rlt::malloc(device_gpu, gpu_gae_obs);
    rlt::malloc(device_gpu, gpu_gae_values);
    rlt::malloc(device_gpu, gpu_episode_lengths_log);
    rlt::malloc(device_gpu, gpu_episode_returns_log);
    rlt::malloc(device_gpu, gpu_episode_end_reasons_log);
    rlt::malloc(device, cpu_episode_lengths_log);
    rlt::malloc(device, cpu_episode_returns_log);
    rlt::malloc(device, cpu_episode_end_reasons_log);

    TI* gpu_episode_start_step = nullptr;
    TI* gpu_episode_start_step_per_row = nullptr;
    cudaMalloc(&gpu_episode_start_step, N_ENVIRONMENTS * sizeof(TI));
    cudaMalloc(&gpu_episode_start_step_per_row, STEPS_TOTAL * sizeof(TI));
    cudaMemset(gpu_episode_start_step, 0, N_ENVIRONMENTS * sizeof(TI));

    CAMERA_DATA* gpu_cameras = nullptr;
    CAMERA_DATA* gpu_cameras_open = nullptr;
    CAMERA_DATA* gpu_prev_cameras = nullptr;
    CAMERA_DATA* gpu_target_cameras = nullptr;
    cudaMalloc(&gpu_cameras, N_ENVIRONMENTS * sizeof(CAMERA_DATA));
    if constexpr(RENDER_MOTION_BLUR_ACTIVE){
        cudaMalloc(&gpu_cameras_open, N_ENVIRONMENTS * sizeof(CAMERA_DATA));
        cudaMalloc(&gpu_prev_cameras, N_ENVIRONMENTS * sizeof(CAMERA_DATA));
    }
    cudaMalloc(&gpu_target_cameras, N_ENVIRONMENTS * sizeof(CAMERA_DATA));

    cudaEvent_t cameras_ready_event;
    cudaEventCreateWithFlags(&cameras_ready_event, cudaEventDisableTiming);
    cudaEvent_t target_cameras_ready_event;
    cudaEventCreateWithFlags(&target_cameras_ready_event, cudaEventDisableTiming);
    std::array<cudaEvent_t, N_ACTIVE_SCENES> render_scatter_done_events;
    std::array<cudaEvent_t, N_ACTIVE_SCENES> target_render_scatter_done_events;
    for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
        cudaEventCreateWithFlags(&render_scatter_done_events[active_scene_i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&target_render_scatter_done_events[active_scene_i], cudaEventDisableTiming);
    }

    EpisodeRecorder episode_recorders[TRAJECTORY_NUM_ENVS];
    std::vector<std::vector<TrajectoryStep>> completed_episodes;
    T simulation_dt = static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY);
    typename ENVIRONMENT::State reward_log_state;
    typename ENVIRONMENT::State reward_log_next_state;
    typename ENVIRONMENT::Parameters reward_log_parameters;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_DIM>> reward_log_action;
    rlt::malloc(device, reward_log_action);

    // Trajectory state buffer for post-collect reconstruction
    std::vector<typename ENVIRONMENT::State> trajectory_states(STEPS_PER_ENV * TRAJECTORY_NUM_ENVS);

    // Video mosaic: each env cell shows (target | actual), ENV_GRID_SIDE^2 per active scene.
    static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    static constexpr TI MOSAIC_W = SCENE_GRID_COLS * ENV_GRID_SIDE * CAM_WIDTH * 2;
    static constexpr TI MOSAIC_H = SCENE_GRID_ROWS * ENV_GRID_SIDE * CAM_HEIGHT;
    std::vector<uint8_t> mosaic_frame(MOSAIC_W * MOSAIC_H * 3);
    std::vector<float> cpu_obs_video(N_ENVIRONMENTS * OBSERVATION_DIM);
    std::vector<float> cpu_target_obs_video(N_ENVIRONMENTS * OBSERVATION_DIM);

    // ---------------------------------------------------------------------
    // Training loop
    // ---------------------------------------------------------------------
    std::cout << "Starting PPO training (visual L2F target navigation, CUDA)" << std::endl;
    std::cout << "  N_ENVIRONMENTS:   " << N_ENVIRONMENTS << std::endl;
    std::cout << "  STEPS_PER_ENV:    " << STEPS_PER_ENV << std::endl;
    std::cout << "  SCENE_SET_STEPS:  " << STEPS_PER_ENV * ROLLOUTS_PER_SCENE_SET << std::endl;
    std::cout << "  STEPS_TOTAL:      " << STEPS_TOTAL << std::endl;
    std::cout << "  BATCH_SIZE:       " << BATCH_SIZE << std::endl;
    std::cout << "  N_BATCHES:        " << N_BATCHES << std::endl;
    std::cout << "  COMBINED_IMG_C:   " << COMBINED_IMG_C << " (logical " << COMBINED_IMG_C_LOGICAL << ")" << std::endl;
    std::cout << "  STATE_OBS_DIM:    " << STATE_OBS_DIM << std::endl;
    std::cout << "  OBS_PRIV_DIM:     " << OBS_PRIV_DIM << std::endl;
    std::cout << "  PPO STEP_LIMIT:   " << LOOP_CORE_PARAMETERS::STEP_LIMIT << std::endl;
    std::cout << "  RENDER_AA:        " << (RENDER_ANTI_ALIASING_ACTIVE ? "on" : "off") << " grid=" << (RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : (TI)1) << std::endl;
    std::cout << "  RENDER_MOTION_BLUR: " << (RENDER_MOTION_BLUR_ACTIVE ? "on" : "off") << " samples=" << (RENDER_MOTION_BLUR_ACTIVE ? RENDER_MOTION_BLUR_SAMPLES : (TI)1) << " shutter=[" << RENDER_SHUTTER_FRACTION_MIN << ", " << RENDER_SHUTTER_FRACTION_MAX << "]" << std::endl;

    auto training_start = std::chrono::high_resolution_clock::now();
    static constexpr TI N_PPO_STEPS = LOOP_CORE_PARAMETERS::STEP_LIMIT;

    constexpr TI BLOCKSIZE = 32;
    constexpr TI N_BLOCKS = (N_ENVIRONMENTS + BLOCKSIZE - 1) / BLOCKSIZE;
    dim3 grid(N_BLOCKS);
    dim3 block(BLOCKSIZE);
    rlt::devices::cuda::TAG<DEVICE_GPU, true> tag_device{};
    FILE* ffmpeg_pipe = nullptr;
    bool record_video_scene_set = false;
    std::filesystem::path current_video_path;
    TI current_video_step = 0;
    auto close_video_pipe = [&](){
        if(ffmpeg_pipe){
            pclose(ffmpeg_pipe);
            if(!current_video_path.empty()){
                auto latest_folder = rlt::get_latest_folder(device, extrack_paths);
                rlt::link_latest_artifact(device, latest_folder, current_video_path, current_video_path.parent_path(), current_video_step);
                current_video_path.clear();
                current_video_step = 0;
            }
            ffmpeg_pipe = nullptr;
            record_video_scene_set = false;
        }
    };

    for(TI ppo_step_i = 0; ppo_step_i < N_PPO_STEPS; ppo_step_i++){
        auto step_start = std::chrono::high_resolution_clock::now();
        rlt::set_step(device, device.logger, on_policy_runner_gpu.step);
        T rollout_episode_length_mean = 0;
        T rollout_episode_length_std = 0;
        T rollout_return_mean = 0;
        T rollout_return_std = 0;
        T rollout_reward_mean = 0;
        T rollout_reward_std = 0;
        T rollout_terminated_share = 0;
        TI rollout_episode_count = 0;
        TI rollout_terminated_count = 0;
        TI rollout_truncated_count = 0;
        TI rollout_done_count = 0;

        TI rollout_in_scene_set = ppo_step_i % ROLLOUTS_PER_SCENE_SET;
        TI scene_set_i = ppo_step_i / ROLLOUTS_PER_SCENE_SET;
        bool scene_set_boundary = ppo_step_i % ROLLOUTS_PER_SCENE_SET == 0;
        bool scene_set_end = rollout_in_scene_set + 1 == ROLLOUTS_PER_SCENE_SET;
        bool save_extrack_step = scene_set_end && scene_set_i % CHECKPOINT_CADENCE_SCENE_SETS == 0;
        bool log_reward_components_this_step = ppo_step_i % REWARD_COMPONENT_LOG_INTERVAL_PPO_STEPS == 0;
        if(scene_set_boundary){
            close_video_pipe();
            ppo_visual::force_scene_boundary_reset_kernel<<<grid, block, 0, device_gpu.stream>>>(gpu_truncated_arr, gpu_episode_step_arr, gpu_episode_end_reason_arr);
            rlt::check_status(device_gpu);
            cudaStreamSynchronize(device_gpu.stream);
            rlt::check_status(device_gpu);
            std::shuffle(scene_permutation.begin(), scene_permutation.end(), scene_rng);
            for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                active_scene_indices[active_scene_i] = scene_permutation[active_scene_i];
            }
            upload_active_scenes();
        }

        std::array<RENDERER_TYPE*, N_ACTIVE_SCENES> active_renderers{};
        std::array<cudaStream_t, N_ACTIVE_SCENES> active_scene_render_streams{};
        std::array<void*, N_ACTIVE_SCENES> active_scene_camera_buffers{};
        std::array<void*, N_ACTIVE_SCENES> active_scene_camera_open_buffers{};
        std::array<const uint32_t*, N_ACTIVE_SCENES> active_scene_framebuffer_ptrs{};
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            TI actual_scene_i = active_scene_indices[active_scene_i];
            auto* renderer = renderers[actual_scene_i];
            active_renderers[active_scene_i] = renderer;
            OWLParams rgb_lp = (OWLParams)renderer->backend.rgb_launch_params;
            active_scene_render_streams[active_scene_i] = (cudaStream_t)owlParamsGetCudaStream(rgb_lp, 0);
            active_scene_camera_buffers[active_scene_i] = (void*)owlBufferGetPointer((OWLBuffer)renderer->backend.owl_cameras_buffer, 0);
            ppo_visual::set_active_scene_camera_open_buffer<RENDER_MOTION_BLUR_ACTIVE>(active_scene_camera_open_buffers[active_scene_i], renderer);
            active_scene_framebuffer_ptrs[active_scene_i] = rlt::get_framebuffer_device_ptr(device, *renderer);
        }

        // Episodes flow across PPO steps inside a scene set. At scene-set boundaries all envs
        // are marked truncated before shuffling, so the next prologue samples states in the new scenes.
        {
            std::vector<unsigned char> truncated_host(N_ENVIRONMENTS);
            cudaMemcpy(truncated_host.data(), gpu_truncated_arr, N_ENVIRONMENTS * sizeof(bool), cudaMemcpyDeviceToHost);
            std::vector<T> reset_host(N_ENVIRONMENTS);
            for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++) reset_host[env_i] = truncated_host[env_i] ? (T)1 : (T)0;
            cudaMemcpy(dataset_gpu.reset._data, reset_host.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        }

        // =================================================================
        // Optional video recording (mosaic of target | actual frames)
        // =================================================================
        if(scene_set_boundary){
            record_video_scene_set = scene_set_i % VIDEO_SAVE_INTERVAL_SCENE_SETS == 0;
        }
        if(record_video_scene_set && ffmpeg_pipe == nullptr){
            TI video_step = on_policy_runner_gpu.step + N_ENVIRONMENTS * STEPS_PER_ENV * ROLLOUTS_PER_SCENE_SET;
            auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, video_step);
            std::filesystem::create_directories(step_folder);
            auto video_path = step_folder / "video.mp4";
            current_video_path = video_path;
            current_video_step = video_step;
            char ffmpeg_cmd[1024];
            std::snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
                "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %lux%lu -framerate %lu -i - "
                "-c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast -loglevel warning %s",
                (unsigned long)MOSAIC_W, (unsigned long)MOSAIC_H, (unsigned long)SIMULATION_FREQUENCY, video_path.c_str());
            ffmpeg_pipe = popen(ffmpeg_cmd, "w");
            if(!ffmpeg_pipe){
                std::cerr << "Failed to open ffmpeg pipe for " << video_path << std::endl;
                record_video_scene_set = false;
                current_video_path.clear();
            }
        }
        bool record_video = record_video_scene_set && ffmpeg_pipe != nullptr;

        // =================================================================
        // Data collection
        // =================================================================
        T cam_aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
        TI frame_step_start = ppo_step_i * STEPS_PER_ENV;
        for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
            TI frame_step_i = frame_step_start + step_i;
            // 1. Prologue: episode reset, sample indoor pos+yaw, observe state + privileged
            auto observations_privileged = rlt::view_range(device_gpu, dataset_gpu.all_observations_privileged, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            auto state_observations = rlt::view_range(device_gpu, gpu_all_state_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            ppo_visual::prologue_kernel<<<grid, block, 0, device_gpu.stream>>>(
                tag_device, gpu_envs_arr, gpu_params_arr, gpu_states_arr,
                gpu_truncated_arr, gpu_episode_step_arr, gpu_episode_return_arr,
                gpu_episode_end_reason_arr,
                gpu_render_reset_arr,
                gpu_shutter_fraction_arr,
                observations_privileged, state_observations,
                gpu_episode_lengths_log, gpu_episode_returns_log, gpu_episode_end_reasons_log,
                gpu_brightness_scale_arr,
                gpu_target_brightness_scale_arr,
                gpu_target_frame_roll_arr,
                gpu_target_frame_pitch_arr,
                gpu_scene_translation_arr,
                gpu_scene_yaw_arr,
                gpu_scene_yaw_cos_arr,
                gpu_scene_yaw_sin_arr,
                gpu_indoor_positions, gpu_num_indoor_positions, gpu_env_scene, MAX_INDOOR_POS,
                gpu_episode_start_step,
                rng_gpu, step_i, frame_step_i);
            rlt::check_status(device_gpu);
            record_episode_start_kernel<<<grid, block, 0, device_gpu.stream>>>(gpu_episode_start_step, gpu_episode_start_step_per_row, step_i);
            rlt::check_status(device_gpu);

            // 2. Build cameras + render student frame per active scene
            ppo_visual::make_cameras_kernel<<<grid, block, 0, device_gpu.stream>>>(
                tag_device, gpu_params_arr, gpu_states_arr,
                gpu_cameras, gpu_cameras_open, gpu_prev_cameras,
                gpu_render_reset_arr,
                gpu_shutter_fraction_arr,
                frame_step_i, cam_aspect,
                gpu_scene_translation_arr, gpu_scene_yaw_cos_arr, gpu_scene_yaw_sin_arr);
            rlt::check_status(device_gpu);

            T* obs_ptr = rlt::data(dataset_gpu.all_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
            cudaEventRecord(cameras_ready_event, device_gpu.stream);
            for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                auto& renderer = *active_renderers[active_scene_i];
                constexpr TI n_envs_s = N_ENVIRONMENTS_PER_SCENE;
                TI base_env = active_scene_i * N_ENVIRONMENTS_PER_SCENE;
                cudaStream_t optix_stream = active_scene_render_streams[active_scene_i];
                cudaStreamWaitEvent(optix_stream, cameras_ready_event, 0);
                if constexpr(RENDER_MOTION_BLUR_ACTIVE){
                    cudaMemcpyAsync(active_scene_camera_open_buffers[active_scene_i], gpu_cameras_open + base_env,
                                    n_envs_s * sizeof(CAMERA_DATA),
                                    cudaMemcpyDeviceToDevice, optix_stream);
                }
                cudaMemcpyAsync(active_scene_camera_buffers[active_scene_i], gpu_cameras + base_env,
                                n_envs_s * sizeof(CAMERA_DATA),
                                cudaMemcpyDeviceToDevice, optix_stream);
                int total_scatter = n_envs_s * CAM_WIDTH * CAM_HEIGHT;
                int pf_block = 256;
                int pf_grid = (total_scatter + pf_block - 1) / pf_block;
                rlt::render_rgb_only_launch(device, renderer);
                const uint32_t* fb_ptr = active_scene_framebuffer_ptrs[active_scene_i];
                if constexpr(BRIGHTNESS_RANDOMIZATION_RANGE > 0){
                    scatter_pixel_to_float_kernel<true><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, obs_ptr, gpu_brightness_scale_arr, base_env, n_envs_s, CAM_WIDTH * CAM_HEIGHT, OBSERVATION_DIM);
                } else {
                    scatter_pixel_to_float_kernel<false><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, obs_ptr, nullptr, base_env, n_envs_s, CAM_WIDTH * CAM_HEIGHT, OBSERVATION_DIM);
                }
                cudaEventRecord(render_scatter_done_events[active_scene_i], optix_stream);
            }
            for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                cudaStreamWaitEvent(device_gpu.stream, render_scatter_done_events[active_scene_i], 0);
            }
            // Copy student frame into circular history buffer
            {
                TI history_slot = frame_step_i % FRAME_STACK_HISTORY_LENGTH;
                T* history_slot_ptr = rlt::data(gpu_frame_stack_history) + (TI)(history_slot * N_ENVIRONMENTS) * OBSERVATION_DIM;
                cudaMemcpyAsync(history_slot_ptr, obs_ptr, N_ENVIRONMENTS * OBSERVATION_DIM * sizeof(T), cudaMemcpyDeviceToDevice, device_gpu.stream);
            }

            // 3. Build target cameras + render target frame
            ppo_visual::make_target_cameras_kernel<<<grid, block, 0, device_gpu.stream>>>(
                tag_device, gpu_params_arr, gpu_target_cameras,
                cam_aspect,
                gpu_target_frame_roll_arr,
                gpu_target_frame_pitch_arr,
                gpu_scene_translation_arr, gpu_scene_yaw_cos_arr, gpu_scene_yaw_sin_arr);
            rlt::check_status(device_gpu);

            T* target_obs_ptr = rlt::data(gpu_all_target_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
            cudaEventRecord(target_cameras_ready_event, device_gpu.stream);
            for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                auto& renderer = *active_renderers[active_scene_i];
                constexpr TI n_envs_s = N_ENVIRONMENTS_PER_SCENE;
                TI base_env = active_scene_i * N_ENVIRONMENTS_PER_SCENE;
                cudaStream_t optix_stream = active_scene_render_streams[active_scene_i];
                cudaStreamWaitEvent(optix_stream, target_cameras_ready_event, 0);
                if constexpr(RENDER_MOTION_BLUR_ACTIVE){
                    cudaMemcpyAsync(active_scene_camera_open_buffers[active_scene_i], gpu_target_cameras + base_env,
                                    n_envs_s * sizeof(CAMERA_DATA),
                                    cudaMemcpyDeviceToDevice, optix_stream);
                }
                cudaMemcpyAsync(active_scene_camera_buffers[active_scene_i], gpu_target_cameras + base_env,
                                n_envs_s * sizeof(CAMERA_DATA),
                                cudaMemcpyDeviceToDevice, optix_stream);
                int total_scatter = n_envs_s * CAM_WIDTH * CAM_HEIGHT;
                int pf_block = 256;
                int pf_grid = (total_scatter + pf_block - 1) / pf_block;
                rlt::render_rgb_only_launch(device, renderer);
                const uint32_t* fb_ptr = active_scene_framebuffer_ptrs[active_scene_i];
                if constexpr(BRIGHTNESS_RANDOMIZATION_RANGE > 0 || TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE > 0){
                    scatter_pixel_to_float_kernel<true><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, target_obs_ptr, gpu_target_brightness_scale_arr, base_env, n_envs_s, CAM_WIDTH * CAM_HEIGHT, OBSERVATION_DIM);
                } else {
                    scatter_pixel_to_float_kernel<false><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, target_obs_ptr, nullptr, base_env, n_envs_s, CAM_WIDTH * CAM_HEIGHT, OBSERVATION_DIM);
                }
                cudaEventRecord(target_render_scatter_done_events[active_scene_i], optix_stream);
            }
            for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                cudaStreamWaitEvent(device_gpu.stream, target_render_scatter_done_events[active_scene_i], 0);
            }

            // Video mosaic write: pull both frames to CPU and place each env cell as (target | actual).
            if(record_video && ffmpeg_pipe){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(cpu_obs_video.data(), obs_ptr, N_ENVIRONMENTS * OBSERVATION_DIM * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(cpu_target_obs_video.data(), target_obs_ptr, N_ENVIRONMENTS * OBSERVATION_DIM * sizeof(float), cudaMemcpyDeviceToHost);
                for(TI scene_row = 0; scene_row < SCENE_GRID_ROWS; scene_row++){
                    for(TI scene_col = 0; scene_col < SCENE_GRID_COLS; scene_col++){
                        TI active_scene_i = scene_row * SCENE_GRID_COLS + scene_col;
                        if(active_scene_i >= N_ACTIVE_SCENES) continue;
                        for(TI local_row = 0; local_row < ENV_GRID_SIDE; local_row++){
                            for(TI local_col = 0; local_col < ENV_GRID_SIDE; local_col++){
                                TI local_env = local_row * ENV_GRID_SIDE + local_col;
                                TI env_i = active_scene_i * N_ENVIRONMENTS_PER_SCENE + local_env;
                                const float* env_obs = cpu_obs_video.data() + env_i * OBSERVATION_DIM;
                                const float* env_target_obs = cpu_target_obs_video.data() + env_i * OBSERVATION_DIM;
                                TI cell_x = (scene_col * ENV_GRID_SIDE + local_col) * CAM_WIDTH * 2;
                                TI cell_y = (scene_row * ENV_GRID_SIDE + local_row) * CAM_HEIGHT;
                                for(TI py = 0; py < CAM_HEIGHT; py++){
                                    for(TI px = 0; px < CAM_WIDTH; px++){
                                        TI pixel_i = py * CAM_WIDTH + px;
                                        TI mosaic_y = cell_y + py;
                                        TI target_x = cell_x + px;
                                        TI actual_x = cell_x + CAM_WIDTH + px;
                                        TI target_idx = (mosaic_y * MOSAIC_W + target_x) * 3;
                                        TI actual_idx = (mosaic_y * MOSAIC_W + actual_x) * 3;
                                        mosaic_frame[target_idx + 0] = static_cast<uint8_t>(std::clamp(env_target_obs[pixel_i * 3 + 0] * 255.0f, 0.0f, 255.0f));
                                        mosaic_frame[target_idx + 1] = static_cast<uint8_t>(std::clamp(env_target_obs[pixel_i * 3 + 1] * 255.0f, 0.0f, 255.0f));
                                        mosaic_frame[target_idx + 2] = static_cast<uint8_t>(std::clamp(env_target_obs[pixel_i * 3 + 2] * 255.0f, 0.0f, 255.0f));
                                        mosaic_frame[actual_idx + 0] = static_cast<uint8_t>(std::clamp(env_obs[pixel_i * 3 + 0] * 255.0f, 0.0f, 255.0f));
                                        mosaic_frame[actual_idx + 1] = static_cast<uint8_t>(std::clamp(env_obs[pixel_i * 3 + 1] * 255.0f, 0.0f, 255.0f));
                                        mosaic_frame[actual_idx + 2] = static_cast<uint8_t>(std::clamp(env_obs[pixel_i * 3 + 2] * 255.0f, 0.0f, 255.0f));
                                    }
                                }
                            }
                        }
                    }
                }
                std::fwrite(mosaic_frame.data(), 1, mosaic_frame.size(), ffmpeg_pipe);
            }

            // 4. Build combined rollout input from history + target
            {
                int total_elements = N_ENVIRONMENTS * COMBINED_OBS_DIM;
                build_frame_stacked_with_target_from_history_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                    rlt::data(gpu_frame_stack_history),
                    target_obs_ptr,
                    gpu_episode_start_step,
                    frame_step_i,
                    rlt::data(gpu_rollout_combined),
                    OBSERVATION_DIM, IMG_C, FRAME_STACK_N, FRAME_STACK_STRIDE, COMBINED_IMG_C, COMBINED_OBS_DIM, N_ENVIRONMENTS);
            }
            rlt::check_status(device_gpu);

            // 5. Actor evaluate (rollout) → action means
            {
                using ROLLOUT_IMG_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, IMG_H, IMG_W, COMBINED_IMG_C>;
                auto step_combined_reshaped = rlt::reshape_row_major(device_gpu, gpu_rollout_combined, ROLLOUT_IMG_SHAPE{});
                auto step_state_obs = rlt::view_range(device_gpu, gpu_all_state_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                using ROLLOUT_STATE_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, STATE_OBS_DIM>;
                auto step_state_reshaped = rlt::reshape_row_major(device_gpu, step_state_obs, ROLLOUT_STATE_SHAPE{});
                auto inputs = rlt::nn_models::parallel::pack_inputs(step_combined_reshaped, step_state_reshaped);
                auto gpu_actions_eval_tensor = rlt::to_tensor(device_gpu, gpu_actions_eval);
                auto gpu_actions_eval_reshaped = rlt::reshape_row_major(device_gpu, gpu_actions_eval_tensor, rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, ACTION_DIM>{});
                rlt::evaluate(device_gpu, rollout_actor_gpu, inputs, gpu_actions_eval_reshaped, rollout_actor_buffers, rng_gpu);
            }

            // 6. Copy action means into dataset (the actions_mean view + actions sample slot)
            {
                auto actions_mean_view = rlt::view(device_gpu, dataset_gpu.actions_mean, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
                rlt::copy(device_gpu, device_gpu, gpu_actions_eval, actions_mean_view);
            }

            // 7. Epilogue: sample noisy action, log_prob, env step, reward, store into dataset
            if(log_reward_components_this_step && step_i == STEPS_PER_ENV - 1){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(&reward_log_state, gpu_states_arr, sizeof(typename ENVIRONMENT::State), cudaMemcpyDeviceToHost);
            }
            {
                auto& last_layer_gpu = ppo_gpu.actor.head;
                auto log_std_gpu = rlt::matrix_view(device_gpu, last_layer_gpu.log_std.parameters);
                auto actions_mean_view = rlt::view(device_gpu, dataset_gpu.actions_mean, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
                auto actions_view = rlt::view(device_gpu, dataset_gpu.actions, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
                ppo_visual::epilogue_kernel<<<grid, block, 0, device_gpu.stream>>>(
                    tag_device, gpu_envs_arr, gpu_params_arr, gpu_states_arr,
                    gpu_truncated_arr, gpu_episode_step_arr, gpu_episode_return_arr,
                    gpu_episode_end_reason_arr,
                    actions_mean_view, actions_view, log_std_gpu,
                    dataset_gpu, rng_gpu, step_i, EPISODE_STEP_LIMIT);
                rlt::check_status(device_gpu);
            }
            if(log_reward_components_this_step && step_i == STEPS_PER_ENV - 1){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(&reward_log_next_state, gpu_states_arr, sizeof(typename ENVIRONMENT::State), cudaMemcpyDeviceToHost);
                cudaMemcpy(&reward_log_parameters, gpu_params_arr, sizeof(typename ENVIRONMENT::Parameters), cudaMemcpyDeviceToHost);
            }

            // 8. Pull state for trajectory recording
            if(save_extrack_step){
                cudaStreamSynchronize(device_gpu.stream);
                std::vector<typename ENVIRONMENT::State> tmp_states(TRAJECTORY_NUM_ENVS);
                cudaMemcpy(tmp_states.data(), gpu_states_arr, TRAJECTORY_NUM_ENVS * sizeof(typename ENVIRONMENT::State), cudaMemcpyDeviceToHost);
                for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                    trajectory_states[step_i * TRAJECTORY_NUM_ENVS + env_i] = tmp_states[env_i];
                }
            }
        }

        if(ffmpeg_pipe && rollout_in_scene_set + 1 == ROLLOUTS_PER_SCENE_SET){
            close_video_pipe();
        }

        // Final privileged observations for value bootstrap
        {
            auto final_obs_priv = rlt::view_range(device_gpu, dataset_gpu.all_observations_privileged, STEPS_PER_ENV * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            ppo_visual::final_priv_obs_kernel<<<grid, block, 0, device_gpu.stream>>>(
                tag_device, gpu_envs_arr, gpu_params_arr, gpu_states_arr, final_obs_priv, rng_gpu);
            rlt::check_status(device_gpu);
        }
        on_policy_runner_gpu.step += N_ENVIRONMENTS * STEPS_PER_ENV;
        rlt::set_step(device, device.logger, on_policy_runner_gpu.step);

        // =================================================================
        // GPU→CPU: copy dataset for GAE + training
        // =================================================================
        cudaDeviceSynchronize();
        {
            auto gpu_scalar = rlt::matrix_view(device_gpu, dataset_gpu.scalar_data);
            auto cpu_scalar = rlt::matrix_view(device, dataset.scalar_data);
            rlt::copy(device_gpu, device, gpu_scalar, cpu_scalar);
        }
        {
            auto gpu_obs_priv = rlt::matrix_view(device_gpu, dataset_gpu.all_observations_privileged);
            auto cpu_obs_priv = rlt::matrix_view(device, dataset.all_observations_privileged);
            rlt::copy(device_gpu, device, gpu_obs_priv, cpu_obs_priv);
        }
        if(log_reward_components_this_step){
            static constexpr TI REWARD_LOG_POS = (STEPS_PER_ENV - 1) * N_ENVIRONMENTS;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                rlt::set(reward_log_action, 0, action_i, rlt::get(dataset.actions, REWARD_LOG_POS, action_i));
            }
            rlt::log_reward(device, envs[0].dynamics, reward_log_parameters.dynamics, reward_log_state, reward_log_action, reward_log_next_state, reward_log_rng);
        }

        // Episode statistics + log
        {
            rlt::copy(device_gpu, device, gpu_episode_lengths_log, cpu_episode_lengths_log);
            rlt::copy(device_gpu, device, gpu_episode_returns_log, cpu_episode_returns_log);
            rlt::copy(device_gpu, device, gpu_episode_end_reasons_log, cpu_episode_end_reasons_log);
            T length_sum = 0;
            T length_sq_sum = 0;
            T length_sum_terminated = 0;
            T length_sum_time_limit = 0;
            T length_sum_scene_boundary = 0;
            T length_sum_task = 0;
            T return_sum = 0;
            T return_sq_sum = 0;
            T reward_sum = 0;
            T reward_sq_sum = 0;
            TI count = 0;
            TI episode_end_terminated_count = 0;
            TI episode_end_time_limit_count = 0;
            TI episode_end_scene_boundary_count = 0;
            for(TI pos = 0; pos < STEPS_TOTAL; pos++){
                T reward_value = rlt::get(dataset.rewards, pos, 0);
                reward_sum += reward_value;
                reward_sq_sum += reward_value * reward_value;
                bool terminated_event = rlt::get(dataset.terminated, pos, 0) > (T)0.5;
                bool done_event = rlt::get(dataset.truncated, pos, 0) > (T)0.5;
                if(terminated_event){
                    rollout_terminated_count++;
                }
                if(done_event){
                    rollout_done_count++;
                    if(!terminated_event){
                        rollout_truncated_count++;
                    }
                }
                T ep_len = rlt::get(cpu_episode_lengths_log, pos, 0);
                if(ep_len >= (T)0){
                    rlt::add_scalar(device, device.logger, "episode/length", ep_len, 100);
                    T ep_return = rlt::get(cpu_episode_returns_log, pos, 0);
                    rlt::add_scalar(device, device.logger, "episode/return", ep_return, 100);
                    length_sum += ep_len;
                    length_sq_sum += ep_len * ep_len;
                    return_sum += ep_return;
                    return_sq_sum += ep_return * ep_return;
                    TI ep_reason = static_cast<TI>(rlt::get(cpu_episode_end_reasons_log, pos, 0) + (T)0.5);
                    if(ep_reason == static_cast<TI>(EPISODE_END_REASON_TERMINATED)){
                        episode_end_terminated_count++;
                        length_sum_terminated += ep_len;
                        length_sum_task += ep_len;
                        rlt::add_scalar(device, device.logger, "episode/length/terminated", ep_len, 100);
                    } else if(ep_reason == static_cast<TI>(EPISODE_END_REASON_TIME_LIMIT)){
                        episode_end_time_limit_count++;
                        length_sum_time_limit += ep_len;
                        length_sum_task += ep_len;
                        rlt::add_scalar(device, device.logger, "episode/length/time_limit", ep_len, 100);
                    } else if(ep_reason == static_cast<TI>(EPISODE_END_REASON_SCENE_BOUNDARY)){
                        episode_end_scene_boundary_count++;
                        length_sum_scene_boundary += ep_len;
                        rlt::add_scalar(device, device.logger, "episode/length/scene_boundary", ep_len, 100);
                    }
                    count++;
                }
            }
            rollout_reward_mean = reward_sum / static_cast<T>(STEPS_TOTAL);
            rollout_reward_std = rlt::math::sqrt(device.math, rlt::math::max(device.math, (T)0, reward_sq_sum / static_cast<T>(STEPS_TOTAL) - rollout_reward_mean * rollout_reward_mean));
            rollout_terminated_share = rollout_done_count > 0 ? static_cast<T>(rollout_terminated_count) / static_cast<T>(rollout_done_count) : (T)0;
            if(count > 0){
                rollout_episode_count = count;
                rollout_episode_length_mean = length_sum / static_cast<T>(count);
                rollout_episode_length_std = rlt::math::sqrt(device.math, rlt::math::max(device.math, (T)0, length_sq_sum / static_cast<T>(count) - rollout_episode_length_mean * rollout_episode_length_mean));
                rollout_return_mean = return_sum / static_cast<T>(count);
                rollout_return_std = rlt::math::sqrt(device.math, rlt::math::max(device.math, (T)0, return_sq_sum / static_cast<T>(count) - rollout_return_mean * rollout_return_mean));
                std::cout << std::defaultfloat << std::setprecision(6)
                          << "  episodes finished: " << count
                          << "  mean length: " << rollout_episode_length_mean
                          << "  mean return: " << rollout_return_mean << std::endl;
                rlt::add_scalar(device, device.logger, "training/episode_length", rollout_episode_length_mean);
                rlt::add_scalar(device, device.logger, "training/episode_length/mean", rollout_episode_length_mean);
                rlt::add_scalar(device, device.logger, "training/episode_length/std", rollout_episode_length_std);
                rlt::add_scalar(device, device.logger, "training/return/mean", rollout_return_mean);
                rlt::add_scalar(device, device.logger, "training/return/std", rollout_return_std);
                rlt::add_scalar(device, device.logger, "training/episodes", static_cast<T>(rollout_episode_count));
            }
            TI episode_end_task_count = episode_end_terminated_count + episode_end_time_limit_count;
            TI episode_end_count = episode_end_task_count + episode_end_scene_boundary_count;
            rlt::add_scalar(device, device.logger, "training/episode_end/terminated", static_cast<T>(episode_end_terminated_count));
            rlt::add_scalar(device, device.logger, "training/episode_end/time_limit", static_cast<T>(episode_end_time_limit_count));
            rlt::add_scalar(device, device.logger, "training/episode_end/scene_boundary", static_cast<T>(episode_end_scene_boundary_count));
            rlt::add_scalar(device, device.logger, "training/episode_end/task", static_cast<T>(episode_end_task_count));
            rlt::add_scalar(device, device.logger, "training/time_limit_episodes", static_cast<T>(episode_end_time_limit_count));
            rlt::add_scalar(device, device.logger, "training/scene_boundary_resets", static_cast<T>(episode_end_scene_boundary_count));
            rlt::add_scalar(device, device.logger, "training/task_episodes", static_cast<T>(episode_end_task_count));
            if(episode_end_terminated_count > 0){
                rlt::add_scalar(device, device.logger, "training/episode_length/terminated", length_sum_terminated / static_cast<T>(episode_end_terminated_count));
            }
            if(episode_end_time_limit_count > 0){
                rlt::add_scalar(device, device.logger, "training/episode_length/time_limit", length_sum_time_limit / static_cast<T>(episode_end_time_limit_count));
            }
            if(episode_end_scene_boundary_count > 0){
                rlt::add_scalar(device, device.logger, "training/episode_length/scene_boundary", length_sum_scene_boundary / static_cast<T>(episode_end_scene_boundary_count));
            }
            if(episode_end_task_count > 0){
                T task_episode_count = static_cast<T>(episode_end_task_count);
                T termination_rate_task = static_cast<T>(episode_end_terminated_count) / task_episode_count;
                rlt::add_scalar(device, device.logger, "training/episode_length/task", length_sum_task / task_episode_count);
                rlt::add_scalar(device, device.logger, "training/episode_length/excluding_scene_boundary", length_sum_task / task_episode_count);
                rlt::add_scalar(device, device.logger, "training/termination_rate/task_episodes", termination_rate_task);
                rlt::add_scalar(device, device.logger, "training/termination_rate/excluding_scene_boundary", termination_rate_task);
                rlt::add_scalar(device, device.logger, "training/time_limit_rate/task_episodes", static_cast<T>(episode_end_time_limit_count) / task_episode_count);
            }
            if(episode_end_count > 0){
                T all_episode_end_count = static_cast<T>(episode_end_count);
                rlt::add_scalar(device, device.logger, "training/termination_rate/all_episode_ends", static_cast<T>(episode_end_terminated_count) / all_episode_end_count);
                rlt::add_scalar(device, device.logger, "training/scene_boundary_rate/all_episode_ends", static_cast<T>(episode_end_scene_boundary_count) / all_episode_end_count);
            }
            rlt::add_scalar(device, device.logger, "training/reward/mean", rollout_reward_mean);
            rlt::add_scalar(device, device.logger, "training/reward/std", rollout_reward_std);
            rlt::add_scalar(device, device.logger, "training/terminated_share", rollout_terminated_share);
            rlt::add_scalar(device, device.logger, "training/terminated_episodes", static_cast<T>(rollout_terminated_count));
            rlt::add_scalar(device, device.logger, "training/truncated_episodes", static_cast<T>(rollout_truncated_count));
            rlt::add_scalar(device, device.logger, "training/complete_episodes", static_cast<T>(rollout_done_count));
        }

        // Save trajectories to extrack
        if(save_extrack_step){
            // Reconstruct episodes from trajectory_states + dataset
            for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                episode_recorders[env_i].current_episode.clear();
                for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
                    TI pos = step_i * N_ENVIRONMENTS + env_i;
                    TrajectoryStep ts;
                    ts.state = trajectory_states[step_i * TRAJECTORY_NUM_ENVS + env_i];
                    for(TI a = 0; a < ACTION_DIM; a++){
                        ts.actions[a] = rlt::get(dataset.actions, pos, a);
                    }
                    ts.reward = rlt::get(dataset.rewards, pos, 0);
                    ts.terminated = (rlt::get(dataset.terminated, pos, 0) > (T)0.5);
                    episode_recorders[env_i].current_episode.push_back(ts);
                    if(ts.terminated || rlt::get(dataset.truncated, pos, 0) > (T)0.5){
                        completed_episodes.push_back(std::move(episode_recorders[env_i].current_episode));
                        episode_recorders[env_i].current_episode.clear();
                        if(completed_episodes.size() > TRAJECTORY_MAX_EPISODES){
                            completed_episodes.erase(completed_episodes.begin());
                        }
                    }
                }
            }
            if(!completed_episodes.empty()){
                auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, on_policy_runner_gpu.step);
                auto& parameters_ref = rlt::get(on_policy_runner.env_parameters, 0, (TI)0);
                std::string trajectories_json = trajectory_episodes_to_json(device, envs[0], parameters_ref, completed_episodes, simulation_dt);
                std::vector<uint8_t> compressed;
                if(rlt::compress_zlib(trajectories_json, compressed)){
                    std::filesystem::path trajectories_path = step_folder / "trajectories.json.gz";
                    std::ofstream f(trajectories_path, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
                    f.close();
                    auto latest_folder = rlt::get_latest_folder(device, extrack_paths);
                    rlt::link_latest_artifact(device, latest_folder, trajectories_path, step_folder, on_policy_runner_gpu.step);
                }
                std::cout << "  Saved " << completed_episodes.size() << " trajectory episodes to " << step_folder << std::endl;
                completed_episodes.clear();
            }
        }
        on_policy_runner.step = on_policy_runner_gpu.step;

        // =================================================================
        // GAE
        // =================================================================
        {
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
        // Train (per minibatch: gather combined obs → forward+PPO loss+backward+step)
        // =================================================================
        {
            // Sync log_std GPU→CPU once per PPO step
            auto& last_layer_gpu = ppo_gpu.actor.head;
            auto& last_layer_cpu = ppo.actor.head;
            rlt::copy(device_gpu, device, last_layer_gpu.log_std.parameters, last_layer_cpu.log_std.parameters);
            rlt::copy(device_gpu, device, last_layer_gpu.log_std.gradient, last_layer_cpu.log_std.gradient);
        }

        T ppo_actor_loss_sum = 0;
        T ppo_entropy_sum = 0;
        T ppo_approx_kl_sum = 0;
        T ppo_ratio_sum = 0;
        T ppo_advantage_mean_sum = 0;
        T ppo_advantage_std_sum = 0;
        T ppo_critic_loss_sum = 0;
        TI ppo_update_samples = 0;
        TI ppo_clipped_samples = 0;
        TI ppo_update_batches = 0;
        TI ppo_critic_batches = 0;

        for(TI epoch_i = 0; epoch_i < N_EPOCHS; epoch_i++){
            // Random batch order (no within-batch shuffle so frame stack indices stay coherent)
            TI batch_order[N_BATCHES];
            for(TI i = 0; i < N_BATCHES; i++) batch_order[i] = i;
            for(TI i = N_BATCHES - 1; i > 0; i--){
                TI j = rlt::random::uniform_int_distribution(device.random, (TI)0, i, rng);
                std::swap(batch_order[i], batch_order[j]);
            }

            for(TI batch_idx = 0; batch_idx < N_BATCHES; batch_idx++){
                TI batch_i = batch_order[batch_idx];
                TI batch_offset = batch_i * BATCH_SIZE;

                rlt::zero_gradient(device_gpu, ppo_gpu.actor);
                rlt::zero_gradient(device_gpu, ppo_gpu.critic);

                // Build combined batch observation via gather kernel
                {
                    int total_elements = BATCH_SIZE * COMBINED_OBS_DIM;
                    build_frame_stacked_with_target_from_dataset_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                        rlt::data(gpu_frame_stack_history),
                        rlt::data(gpu_all_target_observations),
                        gpu_episode_start_step_per_row,
                        rlt::data(gpu_combined_batch),
                        OBSERVATION_DIM, IMG_C, FRAME_STACK_N, FRAME_STACK_STRIDE, COMBINED_IMG_C, COMBINED_OBS_DIM,
                        frame_step_start, (int)batch_offset, (int)BATCH_SIZE, (int)N_ENVIRONMENTS);
                    rlt::check_status(device_gpu);
                }

                // Slice state observations for this batch
                auto batch_state_obs = rlt::view_range(device_gpu, gpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                using TRAIN_IMG_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
                using TRAIN_STATE_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>;
                auto batch_combined_reshaped = rlt::reshape_row_major(device_gpu, gpu_combined_batch, TRAIN_IMG_SHAPE{});
                auto batch_state_reshaped = rlt::reshape_row_major(device_gpu, batch_state_obs, TRAIN_STATE_SHAPE{});
                auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(batch_combined_reshaped, batch_state_reshaped);

                auto gpu_actions_train_tensor = rlt::to_tensor(device_gpu, gpu_actions_train);
                auto gpu_actions_train_reshaped = rlt::reshape_row_major(device_gpu, gpu_actions_train_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                rlt::forward(device_gpu, ppo_gpu.actor, fwd_inputs, gpu_actions_train_reshaped, actor_buffers, rng_gpu);
                cudaDeviceSynchronize();

                // PPO loss on CPU using freshly-forwarded action means + log_std
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
                for(TI i = 0; i < BATCH_SIZE; i++){
                    T adv = rlt::get(batch_advantages, i, 0);
                    advantage_mean += adv;
                    advantage_std += adv * adv;
                }
                advantage_mean /= BATCH_SIZE;
                advantage_std /= BATCH_SIZE;
                advantage_std = rlt::math::sqrt(device.math, rlt::math::max(device.math, (T)0, advantage_std - advantage_mean * advantage_mean));
                ppo_advantage_mean_sum += advantage_mean;
                ppo_advantage_std_sum += advantage_std;

                for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
                    T action_log_prob = 0;
                    T action_entropy = 0;
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        T current_action = rlt::get(ppo_buffers.current_batch_actions, batch_step_i, action_i);
                        T rollout_action = rlt::get(batch_actions, batch_step_i, action_i);
                        T current_action_log_std = rlt::get(device, last_layer_cpu.log_std.parameters, action_i);
                        action_log_prob += rlt::random::normal_distribution::log_prob(device.random, current_action, current_action_log_std, rollout_action);
                        action_entropy += current_action_log_std + rlt::math::log(device.math, static_cast<T>(2) * rlt::math::PI<T>) / static_cast<T>(2) + static_cast<T>(0.5);
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
                    T pessimistic_surrogate = ratio_min_switch ? normal_advantage : clipped_advantage;
                    ppo_actor_loss_sum += -pessimistic_surrogate;
                    ppo_entropy_sum += action_entropy;
                    ppo_approx_kl_sum += (ratio - (T)1) - log_ratio;
                    ppo_ratio_sum += ratio;
                    ppo_clipped_samples += clipped ? 1 : 0;
                    ppo_update_samples++;
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
                // Sync log_std + d_action gradients CPU→GPU, actor backward + optimizer step on GPU
                rlt::copy(device, device_gpu, last_layer_cpu.log_std.parameters, last_layer_gpu.log_std.parameters);
                rlt::copy(device, device_gpu, last_layer_cpu.log_std.gradient, last_layer_gpu.log_std.gradient);
                rlt::copy(device, device_gpu, ppo_buffers.d_action_log_prob_d_action, gpu_d_action_train);
                auto gpu_d_action_tensor = rlt::to_tensor(device_gpu, gpu_d_action_train);
                auto gpu_d_action_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(batch_combined_reshaped, batch_state_reshaped);
                rlt::backward(device_gpu, ppo_gpu.actor, bwd_inputs, gpu_d_action_reshaped, actor_buffers);
                cudaDeviceSynchronize();
                rlt::step(device_gpu, actor_optimizer_gpu, ppo_gpu.actor);
                cudaDeviceSynchronize();

                // Critic forward + MSE loss + backward + step on GPU
                {
                    auto batch_obs_priv = rlt::view_range(device, dataset.all_observations_privileged, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                    auto batch_obs_priv_matrix = rlt::matrix_view(device, batch_obs_priv);
                    rlt::copy(device, device_gpu, batch_obs_priv_matrix, gpu_critic_obs);
                    auto gpu_critic_obs_tensor = rlt::to_tensor(device_gpu, gpu_critic_obs);
                    using OBS_PRIV_SHAPE = typename ON_POLICY_RUNNER_DATASET_TYPE::OBS_PRIV_SHAPE;
                    using CRITIC_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, BATCH_SIZE>, (TI)1>;
                    auto gpu_critic_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_critic_obs_tensor, CRITIC_INPUT_SHAPE{});
                    rlt::forward(device_gpu, ppo_gpu.critic, gpu_critic_obs_reshaped, critic_buffers, rng_gpu);
                    cudaDeviceSynchronize();
                    {
                        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, 1>> cpu_critic_output, cpu_d_critic;
                        rlt::malloc(device, cpu_critic_output);
                        rlt::malloc(device, cpu_d_critic);
                        auto critic_output_tensor = rlt::output(device_gpu, ppo_gpu.critic);
                        auto critic_output_matrix = rlt::matrix_view(device_gpu, critic_output_tensor);
                        rlt::copy(device_gpu, device, critic_output_matrix, cpu_critic_output);
                        T critic_loss = rlt::nn::loss_functions::mse::evaluate(device, cpu_critic_output, batch_target_values);
                        ppo_critic_loss_sum += critic_loss;
                        ppo_critic_batches++;
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
                ppo_update_batches++;
            }
        }

        // Sync trained actor weights → rollout actor for next data collection
        rlt::copy(device_gpu, device_gpu, ppo_gpu.actor, rollout_actor_gpu);
        rlt::copy(device_gpu, device, ppo_gpu.actor.head.log_std.parameters, ppo.actor.head.log_std.parameters);
        rlt::copy(device_gpu, device, actor_optimizer_gpu, actor_optimizer);
        rlt::copy(device_gpu, device, critic_optimizer_gpu, critic_optimizer);

        // Logging
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> training_elapsed = now - training_start;
        std::chrono::duration<T> step_elapsed = now - step_start;
        T sps_lifetime = on_policy_runner_gpu.step / training_elapsed.count();
        T sps_current = N_ENVIRONMENTS * STEPS_PER_ENV / step_elapsed.count();
        std::cout << "PPO step " << std::setw(5) << ppo_step_i
                  << "  env_step " << std::setw(10) << on_policy_runner_gpu.step
                  << "  elapsed " << std::setw(7) << std::setprecision(3) << training_elapsed.count() << "s"
                  << "  (sps lifetime " << std::setw(6) << std::setprecision(0) << std::fixed << sps_lifetime
                  << ", current " << std::setw(6) << std::setprecision(0) << sps_current << ")" << std::defaultfloat << std::endl;

        rlt::add_scalar(device, device.logger, "ppo/step", ppo_step_i);
        rlt::add_scalar(device, device.logger, "ppo/actor_learning_rate", rlt::get(device, actor_optimizer.parameters, 0).alpha);
        rlt::add_scalar(device, device.logger, "ppo/critic_learning_rate", rlt::get(device, critic_optimizer.parameters, 0).alpha);
        if(ppo_update_samples > 0){
            T inv_samples = static_cast<T>(1) / static_cast<T>(ppo_update_samples);
            T approx_kl = ppo_approx_kl_sum * inv_samples;
            rlt::add_scalar(device, device.logger, "ppo/actor_loss", ppo_actor_loss_sum * inv_samples);
            rlt::add_scalar(device, device.logger, "ppo/entropy", ppo_entropy_sum * inv_samples);
            rlt::add_scalar(device, device.logger, "ppo/approx_kl", approx_kl);
            rlt::add_scalar(device, device.logger, "ppo/policy_kl", approx_kl);
            rlt::add_scalar(device, device.logger, "ppo/clip_fraction", static_cast<T>(ppo_clipped_samples) * inv_samples);
            rlt::add_scalar(device, device.logger, "ppo/ratio_mean", ppo_ratio_sum * inv_samples);
        }
        if(ppo_update_batches > 0){
            T inv_batches = static_cast<T>(1) / static_cast<T>(ppo_update_batches);
            rlt::add_scalar(device, device.logger, "ppo/advantage/mean", ppo_advantage_mean_sum * inv_batches);
            rlt::add_scalar(device, device.logger, "ppo/advantage/std", ppo_advantage_std_sum * inv_batches);
        }
        if(ppo_critic_batches > 0){
            rlt::add_scalar(device, device.logger, "ppo/critic_loss", ppo_critic_loss_sum / static_cast<T>(ppo_critic_batches));
        }
        rlt::add_scalar(device, device.logger, "steps_per_second", sps_current);
        rlt::add_scalar(device, device.logger, "timing/steps_per_second", sps_current);
        rlt::add_scalar(device, device.logger, "timing/steps_per_second_lifetime", sps_lifetime);
        rlt::add_scalar(device, device.logger, "timing/step_time_s", step_elapsed.count());
        rlt::add_scalar(device, device.logger, "timing/total_time_s", training_elapsed.count());
        rlt::add_scalar(device, device.logger, "rendering/anti_aliasing_grid_size", RENDER_ANTI_ALIASING_ACTIVE ? static_cast<T>(RENDER_ANTI_ALIASING_GRID_SIZE) : static_cast<T>(1));
        rlt::add_scalar(device, device.logger, "rendering/motion_blur_samples", RENDER_MOTION_BLUR_ACTIVE ? static_cast<T>(RENDER_MOTION_BLUR_SAMPLES) : static_cast<T>(1));
        if constexpr(RENDER_MOTION_BLUR_ACTIVE){
            rlt::add_scalar(device, device.logger, "rendering/shutter_fraction_min", RENDER_SHUTTER_FRACTION_MIN);
            rlt::add_scalar(device, device.logger, "rendering/shutter_fraction_max", RENDER_SHUTTER_FRACTION_MAX);
        }
        rlt::add_scalar(device, device.logger, "rendering/target_frame_roll_pitch_randomization_range", TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE);
        rlt::add_scalar(device, device.logger, "rendering/target_frame_brightness_mismatch_range", TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE);

        if(save_extrack_step){
            {
                auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, on_policy_runner_gpu.step);
                auto latest_folder = rlt::get_latest_folder(device, extrack_paths);
                std::filesystem::create_directories(step_folder);

                CHECKPOINT_ACTOR_TYPE eval_actor;
                rlt::malloc(device, eval_actor);
                rlt::copy(device_gpu, device, ppo_gpu.actor, eval_actor);

                char fov_buf[32];
                std::snprintf(fov_buf, sizeof(fov_buf), "%.6g", (double)envs[0].parameters.fov);
                std::string state_obs_string = rlt::string(device, envs[0].dynamics, ACTOR_STATE_OBS{});
                std::string image_obs_string = std::string("CameraRGBStackedWithTarget(") + fov_buf + ", "
                    + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ", "
                    + std::to_string(FRAME_STACK_STRIDE) + ", " + std::to_string(FRAME_STACK_N) + ")";
                std::string obs_string = image_obs_string + ", " + state_obs_string;
                std::string rendering_string = std::string("{\"anti_aliasing\": ") + (RENDER_ANTI_ALIASING_ACTIVE ? "true" : "false")
                    + ", \"anti_aliasing_grid_size\": " + std::to_string(RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : (TI)1)
                    + ", \"motion_blur\": " + (RENDER_MOTION_BLUR_ACTIVE ? "true" : "false")
                    + ", \"motion_blur_samples\": " + std::to_string(RENDER_MOTION_BLUR_ACTIVE ? RENDER_MOTION_BLUR_SAMPLES : (TI)1)
                    + ", \"shutter_fraction_min\": " + std::to_string(RENDER_SHUTTER_FRACTION_MIN)
                    + ", \"shutter_fraction_max\": " + std::to_string(RENDER_SHUTTER_FRACTION_MAX)
                    + ", \"target_frame_roll_pitch_randomization_range\": " + std::to_string(TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE)
                    + ", \"target_frame_brightness_mismatch_range\": " + std::to_string(TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE) + "}";
                std::string meta = "{\"environment\": {\"name\": \"l2f_visual\", \"observation\": \"" + obs_string + "\", \"output\": \"ActionMean\", \"rendering\": " + rendering_string + "}}";

                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, N_EXAMPLES, IMG_H, IMG_W, COMBINED_IMG_C>, true>> example_input_0_image;
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, N_EXAMPLES, STATE_OBS_DIM>, true>> example_input_1_state;
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_EXAMPLES, ACTION_DIM>, true>> example_output;
                rlt::malloc(device, example_input_0_image);
                rlt::malloc(device, example_input_1_state);
                rlt::malloc(device, example_output);

                static constexpr TI EXAMPLE_ROW_OFFSET = STEPS_TOTAL - N_EXAMPLES;
                {
                    int total_elements = N_EXAMPLES * COMBINED_OBS_DIM;
                    build_frame_stacked_with_target_from_dataset_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                        rlt::data(gpu_frame_stack_history),
                        rlt::data(gpu_all_target_observations),
                        gpu_episode_start_step_per_row,
                        rlt::data(gpu_combined_batch),
                        OBSERVATION_DIM, IMG_C, FRAME_STACK_N, FRAME_STACK_STRIDE, COMBINED_IMG_C, COMBINED_OBS_DIM,
                        frame_step_start, (int)EXAMPLE_ROW_OFFSET, (int)N_EXAMPLES, (int)N_ENVIRONMENTS);
                    rlt::check_status(device_gpu);
                    cudaDeviceSynchronize();
                    auto src_combined = rlt::view_range(device_gpu, gpu_combined_batch, (TI)0, rlt::tensor::ViewSpec<0, N_EXAMPLES>{});
                    auto src_state = rlt::view_range(device_gpu, gpu_all_state_observations, EXAMPLE_ROW_OFFSET, rlt::tensor::ViewSpec<0, N_EXAMPLES>{});
                    auto dst_image_2d = rlt::reshape_row_major(device, example_input_0_image, rlt::tensor::Shape<TI, N_EXAMPLES, COMBINED_OBS_DIM>{});
                    auto dst_state_2d = rlt::reshape_row_major(device, example_input_1_state, rlt::tensor::Shape<TI, N_EXAMPLES, STATE_OBS_DIM>{});
                    rlt::copy(device_gpu, device, src_combined, dst_image_2d);
                    rlt::copy(device_gpu, device, src_state, dst_state_2d);
                }

                {
                    using BRANCH_0 = typename rlt::utils::tuple_element<0, typename CHECKPOINT_ACTOR_TYPE::SPEC::BRANCH_TUPLE>::type;
                    using BRANCH_1 = typename rlt::utils::tuple_element<1, typename CHECKPOINT_ACTOR_TYPE::SPEC::BRANCH_TUPLE>::type;
                    using BRANCH_0_OUTPUT_SHAPE = rlt::nn_models::parallel::detail::output_shape<typename CHECKPOINT_ACTOR_TYPE::SPEC::CAPABILITY, BRANCH_0>;
                    using BRANCH_1_OUTPUT_SHAPE = rlt::nn_models::parallel::detail::output_shape<typename CHECKPOINT_ACTOR_TYPE::SPEC::CAPABILITY, BRANCH_1>;
                    static constexpr TI BRANCH_0_DIM = rlt::get_last(BRANCH_0_OUTPUT_SHAPE{});
                    static constexpr TI BRANCH_1_DIM = rlt::get_last(BRANCH_1_OUTPUT_SHAPE{});
                    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_EXAMPLES, BRANCH_0_DIM>, true>> branch_0_out;
                    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_EXAMPLES, BRANCH_1_DIM>, true>> branch_1_out;
                    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_EXAMPLES, BRANCH_0_DIM + BRANCH_1_DIM>, true>> concat_out;
                    typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<0>(eval_actor.pipelines))>::template Buffer<true> buffer_0;
                    typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<1>(eval_actor.pipelines))>::template Buffer<true> buffer_1;
                    rlt::malloc(device, branch_0_out);
                    rlt::malloc(device, branch_1_out);
                    rlt::malloc(device, concat_out);
                    rlt::malloc(device, buffer_0);
                    rlt::malloc(device, buffer_1);
                    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
                    auto image_eval_view = rlt::view_memory<rlt::tensor::Shape<TI, N_EXAMPLES, IMG_H, IMG_W, COMBINED_IMG_C>>(device, example_input_0_image);
                    auto state_eval_view = rlt::view_memory<rlt::tensor::Shape<TI, N_EXAMPLES, STATE_OBS_DIM>>(device, example_input_1_state);
                    rlt::evaluate(device, rlt::get<0>(eval_actor.pipelines), image_eval_view, branch_0_out, buffer_0, rng, eval_mode);
                    rlt::evaluate(device, rlt::get<1>(eval_actor.pipelines), state_eval_view, branch_1_out, buffer_1, rng, eval_mode);
                    auto concat_0 = rlt::view_range(device, concat_out, (TI)0, rlt::tensor::ViewSpec<1, BRANCH_0_DIM>{});
                    auto concat_1 = rlt::view_range(device, concat_out, (TI)BRANCH_0_DIM, rlt::tensor::ViewSpec<1, BRANCH_1_DIM>{});
                    rlt::copy(device, device, branch_0_out, concat_0);
                    rlt::copy(device, device, branch_1_out, concat_1);
                    typename decltype(eval_actor.head)::template Buffer<true> head_buffer;
                    rlt::malloc(device, head_buffer);
                    rlt::evaluate(device, eval_actor.head, concat_out, example_output, head_buffer, rng, eval_mode);
                    rlt::free(device, head_buffer);
                    rlt::free(device, branch_0_out);
                    rlt::free(device, branch_1_out);
                    rlt::free(device, concat_out);
                    rlt::free(device, buffer_0);
                    rlt::free(device, buffer_1);
                }

                if constexpr(EXPORT_CHECKPOINT_TAR){
                    std::filesystem::path checkpoint_path = step_folder / "checkpoint.tar";
                    rlt::persist::backends::tar::Writer writer;
                    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> root_group{"", &writer};
                    auto actor_group = rlt::create_group(device, root_group, "actor");
                    rlt::set_attribute(device, actor_group, "checkpoint_name", step_folder.string().c_str());
                    rlt::set_attribute(device, actor_group, "meta", meta.c_str());
                    rlt::save(device, eval_actor, actor_group);
                    auto example_group = rlt::create_group(device, root_group, "example");
                    auto inputs_group = rlt::create_group(device, example_group, "inputs");
                    rlt::save(device, example_input_0_image, inputs_group, "0");
                    rlt::save(device, example_input_1_state, inputs_group, "1");
                    auto outputs_group = rlt::create_group(device, example_group, "outputs");
                    auto example_output_canonical = rlt::reshape_row_major(device, example_output, rlt::tensor::Shape<TI, 1, N_EXAMPLES, ACTION_DIM>{});
                    rlt::save(device, example_output_canonical, outputs_group, "0");
                    rlt::persist::backends::tar::finalize(device, writer);
                    std::ofstream f(checkpoint_path, std::ios::binary);
                    f.write(writer.buffer.data(), writer.buffer.size());
                    f.close();
                    rlt::link_latest_artifact(device, latest_folder, checkpoint_path, step_folder, on_policy_runner_gpu.step);
                }
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
                auto save_hdf5 = [&](auto batch_size_tag){
                    static constexpr TI EXAMPLE_BATCH_SIZE = decltype(batch_size_tag)::value;
                    using SIZED_EVAL_ACTOR_TYPE = typename CHECKPOINT_ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, EXAMPLE_BATCH_SIZE>;
                    SIZED_EVAL_ACTOR_TYPE sized_eval_actor;
                    rlt::malloc(device, sized_eval_actor);
                    rlt::copy(device_gpu, device, ppo_gpu.actor, sized_eval_actor);
                    std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
                    std::filesystem::path checkpoint_path = step_folder / (std::string("checkpoint_") + std::to_string(EXAMPLE_BATCH_SIZE) + "examples.h5");
                    rlt::persist::backends::hdf5::File root_file(checkpoint_path.string(), rlt::persist::backends::hdf5::Mode::WRITE);
                    auto actor_group = rlt::create_group(device, root_file, "actor");
                    rlt::set_attribute(device, actor_group, "checkpoint_name", step_folder.string().c_str());
                    rlt::set_attribute(device, actor_group, "meta", meta.c_str());
                    rlt::save(device, sized_eval_actor, actor_group);
                    auto example_group = rlt::create_group(device, root_file, "example");
                    auto inputs_group = rlt::create_group(device, example_group, "inputs");
                    auto example_input_0_image_view = rlt::view_range(device, example_input_0_image, (TI)0, rlt::tensor::ViewSpec<1, EXAMPLE_BATCH_SIZE>{});
                    auto example_input_1_state_view = rlt::view_range(device, example_input_1_state, (TI)0, rlt::tensor::ViewSpec<1, EXAMPLE_BATCH_SIZE>{});
                    rlt::save(device, example_input_0_image_view, inputs_group, "0");
                    rlt::save(device, example_input_1_state_view, inputs_group, "1");
                    auto outputs_group = rlt::create_group(device, example_group, "outputs");
                    auto example_output_canonical = rlt::reshape_row_major(device, example_output, rlt::tensor::Shape<TI, 1, N_EXAMPLES, ACTION_DIM>{});
                    auto example_output_view = rlt::view_range(device, example_output_canonical, (TI)0, rlt::tensor::ViewSpec<1, EXAMPLE_BATCH_SIZE>{});
                    rlt::save(device, example_output_view, outputs_group, "0");
                    rlt::free(device, sized_eval_actor);
                    return checkpoint_path;
                };
                auto reduced_checkpoint_path = save_hdf5(rlt::utils::typing::integral_constant<TI, REDUCED_BATCH_SIZE>{});
                rlt::link_latest_artifact(device, latest_folder, reduced_checkpoint_path, step_folder, on_policy_runner_gpu.step);
                auto full_checkpoint_path = save_hdf5(rlt::utils::typing::integral_constant<TI, N_EXAMPLES>{});
                rlt::link_latest_artifact(device, latest_folder, full_checkpoint_path, step_folder, on_policy_runner_gpu.step);
#endif
                if constexpr(EXPORT_CHECKPOINT_CODE){
                    auto actor_weights = rlt::save_code(device, eval_actor, std::string("rl_tools::checkpoint::actor"), true);
                    std::stringstream output_ss;
                    output_ss << actor_weights;
                    output_ss << "\n" << "namespace rl_tools::checkpoint::example::inputs{";
                    output_ss << "\n" << rlt::save_code(device, example_input_0_image, std::string("_0"), true);
                    output_ss << "\n" << rlt::save_code(device, example_input_1_state, std::string("_1"), true);
                    output_ss << "\n" << "}";
                    output_ss << "\n" << "namespace rl_tools::checkpoint::example::outputs{";
                    {
                        auto example_output_canonical = rlt::reshape_row_major(device, example_output, rlt::tensor::Shape<TI, 1, N_EXAMPLES, ACTION_DIM>{});
                        output_ss << "\n" << rlt::save_code(device, example_output_canonical, std::string("_0"), true);
                    }
                    output_ss << "\n" << "}";
                    output_ss << "\n" << "namespace rl_tools::checkpoint::meta{";
                    output_ss << "\n" << "   " << "char name[] = \"" << step_folder.string() << "\";";
                    output_ss << "\n" << "   " << "char commit_hash[] = \"" << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
                    output_ss << "\n" << "   " << "char observation[] = \"" << obs_string << "\";";
                    output_ss << "\n" << "}";
                    std::string output_string = output_ss.str();
#ifdef RL_TOOLS_ENABLE_ZLIB
                    {
                        std::filesystem::path checkpoint_code_path = step_folder / "checkpoint.h.gz";
                        std::vector<uint8_t> compressed;
                        rlt::compress_zlib(output_string, compressed);
                        std::ofstream f(checkpoint_code_path, std::ios::binary);
                        f.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
                        f.close();
                        rlt::link_latest_artifact(device, latest_folder, checkpoint_code_path, step_folder, on_policy_runner_gpu.step);
                    }
#endif
                    {
                        std::filesystem::path checkpoint_code_path = step_folder / "checkpoint.h";
                        std::ofstream f(checkpoint_code_path);
                        f << output_string;
                        f.close();
                        rlt::link_latest_artifact(device, latest_folder, checkpoint_code_path, step_folder, on_policy_runner_gpu.step);
                    }
                }
                rlt::free(device, example_input_0_image);
                rlt::free(device, example_input_1_state);
                rlt::free(device, example_output);
                rlt::free(device, eval_actor);
                std::cerr << "Checkpoint saved: " << std::filesystem::absolute(step_folder) << std::endl;
            }
        }

        {
            auto& actor_log_std = ppo.actor.head;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                T log_std_val = rlt::get(device, actor_log_std.log_std.parameters, action_i);
                rlt::add_scalar(device, device.logger, "actor/log_std", log_std_val);
                rlt::add_scalar(device, device.logger, std::string("actor/action_std/") + std::to_string(action_i), rlt::math::exp(device.math, log_std_val));
            }
        }
    }

    close_video_pipe();

    std::cout << "Training finished at env step " << on_policy_runner_gpu.step << std::endl;

    // ---------------------------------------------------------------------
    // Cleanup (minimal — process exit will reclaim)
    // ---------------------------------------------------------------------
    rlt::free(device, ppo);
    rlt::free(device, ppo_buffers);
    rlt::free(device, on_policy_runner);
    rlt::free(device, dataset);
    rlt::free(device, actor_optimizer);
    rlt::free(device, critic_optimizer);
    rlt::free(device, cpu_episode_lengths_log);
    rlt::free(device, cpu_episode_returns_log);
    rlt::free(device, cpu_episode_end_reasons_log);
    rlt::free(device, reward_log_action);
    rlt::free(device, reward_log_rng);

    rlt::free(device_gpu, ppo_gpu);
    rlt::free(device_gpu, actor_buffers);
    rlt::free(device_gpu, critic_buffers);
    rlt::free(device_gpu, critic_buffers_gae);
    rlt::free(device_gpu, on_policy_runner_gpu);
    rlt::free(device_gpu, dataset_gpu);
    rlt::free(device_gpu, rollout_actor_gpu);
    rlt::free(device_gpu, rollout_actor_buffers);
    rlt::free(device_gpu, actor_optimizer_gpu);
    rlt::free(device_gpu, critic_optimizer_gpu);
    rlt::free(device_gpu, gpu_all_target_observations);
    rlt::free(device_gpu, gpu_frame_stack_history);
    rlt::free(device_gpu, gpu_rollout_combined);
    rlt::free(device_gpu, gpu_combined_batch);
    rlt::free(device_gpu, gpu_all_state_observations);
    rlt::free(device_gpu, gpu_actions_eval);
    rlt::free(device_gpu, gpu_actions_train);
    rlt::free(device_gpu, gpu_d_action_train);
    rlt::free(device_gpu, gpu_critic_obs);
    rlt::free(device_gpu, gpu_d_critic_output);
    rlt::free(device_gpu, gpu_gae_obs);
    rlt::free(device_gpu, gpu_gae_values);
    rlt::free(device_gpu, gpu_episode_lengths_log);
    rlt::free(device_gpu, gpu_episode_returns_log);
    rlt::free(device_gpu, gpu_episode_end_reasons_log);
    cudaFree(gpu_cameras);
    if constexpr(RENDER_MOTION_BLUR_ACTIVE){
        cudaFree(gpu_cameras_open);
        cudaFree(gpu_prev_cameras);
    }
    cudaFree(gpu_target_cameras);
    cudaFree(gpu_render_reset_arr);
    if constexpr(RENDER_MOTION_BLUR_ACTIVE){
        cudaFree(gpu_shutter_fraction_arr);
    }

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
    rlt::free(device, device.logger);
#endif

    return 0;
}
