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
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/gru/helper_operations_cuda.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_cuda.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cuda.h>

#include <rl_tools/nn/loss_functions/mse/operations_generic.h>
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>

#include "../../../../src/nn_models/port_checkpoint/raptor/policy.h"

#include <rl_tools/utils/extrack/operations_cpu.h>
#include <rl_tools/utils/zlib/operations_cpu.h>

#include <rl_tools/persist/backends/tar/operations_cpu.h>
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
#include <rl_tools/persist/backends/hdf5/hdf5.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#endif
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/flatten/persist.h>
#include <rl_tools/nn/layers/unflatten/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>
#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/matrix/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/conv2d/persist_code.h>
#include <rl_tools/nn/layers/gru/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/flatten/persist_code.h>
#include <rl_tools/nn/layers/unflatten/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/parallel/persist_code.h>

#include <cuda_bf16.h>

#include <array>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <vector>
#include <numeric>
#include <cstring>
#include <mutex>
#include <string>
#include <filesystem>
#include <fstream>

namespace rlt = rl_tools;

#define USE_FRAME_STACKING
// #define USE_GRU_TEMPORAL
#define STACK_TARGET_CHANNEL
#if defined(USE_FRAME_STACKING) && defined(USE_GRU_TEMPORAL)
#error "USE_FRAME_STACKING and USE_GRU_TEMPORAL are mutually exclusive"
#endif

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
using TYPE_POLICY = rlt::numeric_types::Policy<float,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Parameter, __nv_bfloat16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Activation, __nv_bfloat16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Gradient, __nv_bfloat16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, float>>;
using T_ACTIVATION = TYPE_POLICY::GET<rlt::numeric_types::categories::Activation>;
using T_GRADIENT = TYPE_POLICY::GET<rlt::numeric_types::categories::Gradient>;
using TEACHER_TYPE_POLICY = rlt::numeric_types::Policy<float>;
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
    false, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
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
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = { {0, 0}, {0, 0} };
static constexpr PARAMETERS_TYPE nominal_parameters = { {dynamics, integration, mdp}, disturbances };


// =========================================================================
// Environment static parameters
// =========================================================================
#ifdef USE_FRAME_STACKING
static constexpr TI ACTION_HISTORY_LENGTH = 4;
#else
static constexpr TI ACTION_HISTORY_LENGTH = 1; // for GRU / Markovian
#endif

struct STATIC_PARAMETERS {
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI CLOSED_FORM = false;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>>>;
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

using ACTOR_STATE_OBS = obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI, obs::LinearAccelerationBodyFrame<obs::LinearAccelerationBodyFrameSpecification<T, TI>>>>;
// using ACTOR_STATE_OBS = STATIC_PARAMETERS::OBSERVATION_TYPE;
static constexpr TI STATE_OBS_DIM = ACTOR_STATE_OBS::DIM; // 12


// =========================================================================
// Visual environment specification
// =========================================================================
static constexpr TI N_TOTAL_SCENES = 25;
static constexpr TI N_ACTIVE_SCENES = 2;
static constexpr TI N_ENVIRONMENTS_PER_SCENE = 64;
static constexpr TI N_ENVIRONMENTS = N_ACTIVE_SCENES * N_ENVIRONMENTS_PER_SCENE;
static constexpr TI CAM_WIDTH = 64;
static constexpr TI CAM_HEIGHT = 64;
static constexpr TI NUM_PROBES = 64;

constexpr bool HIGH_FIDELITY_SHADING = true;
using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, N_ENVIRONMENTS_PER_SCENE, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES, HIGH_FIDELITY_SHADING>;
using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;

// =========================================================================
// RAPTOR teacher (CPU only)
// =========================================================================
static constexpr TI RAPTOR_HIDDEN_DIM = 16;
using RAPTOR_OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI,
        obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
        obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
        obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI,
        obs::ActionHistory<obs::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
static constexpr TI RAPTOR_OBS_DIM = RAPTOR_OBSERVATION_TYPE::DIM;

using RAPTOR_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<TEACHER_TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
using RAPTOR_DENSE1 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE1_CONFIG>;
using RAPTOR_GRU_CONFIG = rlt::nn::layers::gru::Configuration<TEACHER_TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM>;
using RAPTOR_GRU = rlt::nn::layers::gru::BindConfiguration<RAPTOR_GRU_CONFIG>;
using RAPTOR_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<TEACHER_TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
using RAPTOR_DENSE2 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE2_CONFIG>;

using RAPTOR_MODULE = rlt::nn_models::sequential::Module<RAPTOR_DENSE1, RAPTOR_GRU, RAPTOR_DENSE2>;
using RAPTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, RAPTOR_OBS_DIM>;
using RAPTOR_CAPABILITY = rlt::nn::capability::Forward<true, false>;
using RAPTOR_MODEL = rlt::nn_models::sequential::Build<RAPTOR_CAPABILITY, RAPTOR_MODULE, RAPTOR_INPUT_SHAPE>;

// =========================================================================
// Student CNN (GPU)
// =========================================================================
static constexpr TI ACTOR_HIDDEN_DIM = 64;
static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
static constexpr TI TARGET_DIM = ACTION_DIM;
static constexpr TI INDOOR_POSITION_DIM = 3;
static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
static constexpr TI BATCH_SIZE = 512;
static constexpr TI STEPS_PER_ENV = 500;
static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * N_ENVIRONMENTS;
static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
static constexpr TI NUM_EPOCHS = 1000000;
static constexpr TI TEACHER_FORCING_EPOCHS = 0;
static constexpr T TEACHER_FORCING_FRACTION = 0.0;
static constexpr TI N_TRAIN_PASSES = 1;
static constexpr TI VIDEO_CADENCE = 10;
static constexpr TI CHECKPOINT_CADENCE = 100;
static constexpr T OBSERVATION_NOISE_STD = 0.00;
static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.5;
static constexpr TI ENV_GRID_SIDE = 8; // sqrt(N_ENVIRONMENTS_PER_SCENE)
static constexpr TI SCENE_GRID_COLS = N_ACTIVE_SCENES;
static constexpr TI SCENE_GRID_ROWS = (N_ACTIVE_SCENES + SCENE_GRID_COLS - 1) / SCENE_GRID_COLS;
static_assert(ENV_GRID_SIDE * ENV_GRID_SIDE == N_ENVIRONMENTS_PER_SCENE, "N_ENVIRONMENTS_PER_SCENE must be a perfect square for per-scene video mosaic");

static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");

// =========================================================================
// Frame stacking configuration
// =========================================================================
#ifdef USE_FRAME_STACKING
static constexpr TI FRAME_STACK_N = 5;
static constexpr TI FRAME_STACK_STRIDE = 20; // 100Hz / 20 = 5Hz
static constexpr TI FRAME_STACK_HISTORY_LENGTH = FRAME_STACK_STRIDE * (FRAME_STACK_N - 1) + 1;
static constexpr TI STACKED_IMG_C = ENVIRONMENT::Observation::CHANNELS * FRAME_STACK_N;
static constexpr TI STACKED_OBS_DIM = ENVIRONMENT::Observation::HEIGHT * ENVIRONMENT::Observation::WIDTH * STACKED_IMG_C;
static constexpr TI COMBINED_IMG_C_LOGICAL = STACKED_IMG_C + ENVIRONMENT::Observation::CHANNELS;
// Pad up to next multiple of 8 so cuDNN uses the tensor-core fast path for Conv1 without inserting an NHWC layout-padding reformat kernel.
static constexpr TI COMBINED_IMG_C = (COMBINED_IMG_C_LOGICAL + 7) & ~((TI)7);
static constexpr TI COMBINED_OBS_DIM = ENVIRONMENT::Observation::HEIGHT * ENVIRONMENT::Observation::WIDTH * COMBINED_IMG_C;
#elif defined(USE_GRU_TEMPORAL)
static constexpr TI BPTT_STEPS = 100;
static constexpr TI GRU_HIDDEN_DIM = 64;
static constexpr TI STACKED_IMG_C = ENVIRONMENT::Observation::CHANNELS;
static constexpr TI STACKED_OBS_DIM = OBSERVATION_DIM;
static constexpr TI COMBINED_IMG_C = STACKED_IMG_C + ENVIRONMENT::Observation::CHANNELS;
static constexpr TI COMBINED_OBS_DIM = ENVIRONMENT::Observation::HEIGHT * ENVIRONMENT::Observation::WIDTH * COMBINED_IMG_C;
static constexpr TI EMBED_DIM = ACTOR_HIDDEN_DIM * 2;
static constexpr TI N_WINDOWS = STEPS_PER_ENV / BPTT_STEPS;
static constexpr TI WINDOW_SAMPLES = BPTT_STEPS * N_ENVIRONMENTS;
static_assert(STEPS_PER_ENV % BPTT_STEPS == 0);
#else
static constexpr TI STACKED_IMG_C = ENVIRONMENT::Observation::CHANNELS;
static constexpr TI STACKED_OBS_DIM = OBSERVATION_DIM;
static constexpr TI COMBINED_IMG_C = STACKED_IMG_C + ENVIRONMENT::Observation::CHANNELS;
static constexpr TI COMBINED_OBS_DIM = ENVIRONMENT::Observation::HEIGHT * ENVIRONMENT::Observation::WIDTH * COMBINED_IMG_C;
#endif

// =========================================================================
// Trajectory recording
// =========================================================================
static constexpr TI TRAJECTORY_SAVE_INTERVAL = 10; // save every N epochs
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
// GPU collection kernels
// =========================================================================
namespace imitation_kernels{
    using DYNAMICS_TYPE = ENVIRONMENT::DYNAMICS_ENV;

    template<typename DEVICE, typename RNG>
    __global__
    void prologue_kernel(
        DEVICE device,
        DYNAMICS_TYPE* envs, PARAMETERS_TYPE* env_params, typename ENVIRONMENT::State* states,
        bool* terminated_flags, TI* episode_step_arr, bool* teacher_forcing_arr,
        T* episode_return_arr, bool* needs_reset_flags,
        T* episode_lengths_log, T* episode_tf_log,
        T teacher_forcing_fraction, bool full_teacher_forcing,
        T* teacher_obs_ptr, T* state_obs_ptr,
        T* raptor_gru_state_ptr, T* raptor_gru_initial_hidden_ptr, TI* raptor_gru_step_ptr,
#ifdef USE_GRU_TEMPORAL
        T_ACTIVATION* student_gru_state_ptr, T_ACTIVATION* student_gru_initial_hidden_ptr, TI* student_gru_step_ptr,
#endif
#ifdef USE_FRAME_STACKING
        TI* episode_start_step,
#endif
        T* brightness_scale_arr,
        T* scene_translation_arr,
        T* scene_yaw_arr,
        T* scene_yaw_cos_arr,
        T* scene_yaw_sin_arr,
        T* indoor_positions_ptr, TI* num_indoor_positions_ptr, TI* env_scene_ptr, TI max_indoor_pos,
        RNG rng, TI step_i, TI episode_step_limit
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& rng_state = rl_tools::get(rng.states, 0, env_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        bool need_reset = terminated_flags[env_i] || episode_step_arr[env_i] >= episode_step_limit;
        needs_reset_flags[env_i] = need_reset;
        if(need_reset){
            if(episode_step_arr[env_i] > 0){
                episode_lengths_log[env_i] = (T)episode_step_arr[env_i];
                episode_tf_log[env_i] = teacher_forcing_arr[env_i] ? (T)1 : (T)0;
            } else {
                episode_lengths_log[env_i] = (T)-1;
                episode_tf_log[env_i] = (T)0;
            }
            rl_tools::sample_initial_parameters(device, env, params, rng_state);
            rl_tools::sample_initial_state(device, env, params, state, rng_state);
            TI scene_idx = env_scene_ptr[env_i];
            TI num_pos = num_indoor_positions_ptr[scene_idx];
            TI pos_idx = rl_tools::random::uniform_int_distribution(device.random, (TI)0, num_pos - 1, rng_state);
            T* pos = indoor_positions_ptr + (scene_idx * max_indoor_pos + pos_idx) * INDOOR_POSITION_DIM;
            scene_translation_arr[env_i * 3 + 0] = pos[0];
            scene_translation_arr[env_i * 3 + 1] = pos[1];
            scene_translation_arr[env_i * 3 + 2] = pos[2];
            T scene_yaw = rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)(2.0 * 3.14159265358979323846), rng_state);
            scene_yaw_arr[env_i] = scene_yaw;
            scene_yaw_cos_arr[env_i] = rl_tools::math::cos(device.math, scene_yaw);
            scene_yaw_sin_arr[env_i] = rl_tools::math::sin(device.math, scene_yaw);
            episode_step_arr[env_i] = 0;
            terminated_flags[env_i] = false;
            episode_return_arr[env_i] = (T)0;
            teacher_forcing_arr[env_i] = full_teacher_forcing || rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, rng_state) < teacher_forcing_fraction;
            brightness_scale_arr[env_i] = (T)1 + (rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, rng_state) * (T)2 - (T)1) * BRIGHTNESS_RANDOMIZATION_RANGE;
            for(TI h = 0; h < RAPTOR_HIDDEN_DIM; h++){
                raptor_gru_state_ptr[env_i * RAPTOR_HIDDEN_DIM + h] = raptor_gru_initial_hidden_ptr[h];
            }
            raptor_gru_step_ptr[env_i] = 0;
#ifdef USE_FRAME_STACKING
            episode_start_step[env_i] = step_i;
#endif
#ifdef USE_GRU_TEMPORAL
            for(TI h = 0; h < GRU_HIDDEN_DIM; h++){
                student_gru_state_ptr[env_i * GRU_HIDDEN_DIM + h] = student_gru_initial_hidden_ptr[h];
            }
            student_gru_step_ptr[env_i] = 0;
#endif
        } else {
            episode_lengths_log[env_i] = (T)-1;
            episode_tf_log[env_i] = (T)0;
        }
        {
            rlt::Matrix<rlt::matrix::Specification<T, TI, 1, RAPTOR_OBS_DIM, true, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> obs_mat;
            obs_mat._data = teacher_obs_ptr + env_i * RAPTOR_OBS_DIM;
            rl_tools::observe(device, env, params, state, RAPTOR_OBSERVATION_TYPE{}, obs_mat, rng_state);
        }
        {
            rlt::Matrix<rlt::matrix::Specification<T, TI, 1, STATE_OBS_DIM, true, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> obs_mat;
            obs_mat._data = state_obs_ptr + env_i * STATE_OBS_DIM;
            rl_tools::observe(device, env, params, state, ACTOR_STATE_OBS{}, obs_mat, rng_state);
        }
    }

    template<typename DEVICE, typename RNG>
    __global__
    void epilogue_kernel(
        DEVICE device,
        DYNAMICS_TYPE* envs, PARAMETERS_TYPE* env_params, typename ENVIRONMENT::State* states,
        bool* terminated_flags, TI* episode_step_arr, T* episode_return_arr,
        T* teacher_actions_ptr, T_ACTIVATION* student_actions_ptr, T_ACTIVATION* all_targets_ptr,
        RNG rng, TI step_i
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& rng_state = rl_tools::get(rng.states, 0, env_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        TI pos = step_i * N_ENVIRONMENTS + env_i;
        for(TI d = 0; d < TARGET_DIM; d++){
            all_targets_ptr[pos * TARGET_DIM + d] = (T_ACTIVATION)teacher_actions_ptr[env_i * ACTION_DIM + d];
        }
        T action_arr[ACTION_DIM];
        for(TI a = 0; a < ACTION_DIM; a++){
            action_arr[a] = (T)student_actions_ptr[env_i * ACTION_DIM + a];
        }
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_DIM, true, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> action_matrix;
        action_matrix._data = action_arr;
        typename ENVIRONMENT::State next_state;
        rl_tools::step(device, env, params, state, action_matrix, next_state, rng_state);
        terminated_flags[env_i] = rl_tools::terminated(device, env, params, next_state, rng_state);
        state = next_state;
        episode_step_arr[env_i]++;
    }

    template<typename DEVICE>
    __global__
    void make_cameras_kernel(
        DEVICE device,
        DYNAMICS_TYPE* envs, PARAMETERS_TYPE* env_params, typename ENVIRONMENT::State* states,
        rlt::rendering::raytracing::CameraData<T>* gpu_cameras,
        T fov, T aspect,
        T camera_offset_body_0, T camera_offset_body_1, T camera_offset_body_2,
        T camera_forward_body_0, T camera_forward_body_1, T camera_forward_body_2,
        T camera_up_body_0, T camera_up_body_1, T camera_up_body_2,
        T* scene_translation_arr, T* scene_yaw_cos_arr, T* scene_yaw_sin_arr
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& state = states[env_i];
        T offset_body[3] = {camera_offset_body_0, camera_offset_body_1, camera_offset_body_2};
        T forward_body[3] = {camera_forward_body_0, camera_forward_body_1, camera_forward_body_2};
        T up_body[3] = {camera_up_body_0, camera_up_body_1, camera_up_body_2};
        T cam_pos_local[3];
        rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, offset_body, cam_pos_local);
        T cam_forward_local[3];
        rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, forward_body, cam_forward_local);
        T cam_up_local[3];
        rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, up_body, cam_up_local);
        T c = scene_yaw_cos_arr[env_i];
        T s = scene_yaw_sin_arr[env_i];
        auto rotate_scene_yaw = [&](const T in[3], T out[3]){
            out[0] = c * in[0] - s * in[1];
            out[1] = s * in[0] + c * in[1];
            out[2] = in[2];
        };
        T state_position_world[3];
        rotate_scene_yaw(state.position, state_position_world);
        T cam_pos_world[3];
        rotate_scene_yaw(cam_pos_local, cam_pos_world);
        T cam_forward_world[3];
        rotate_scene_yaw(cam_forward_local, cam_forward_world);
        T cam_up_world[3];
        rotate_scene_yaw(cam_up_local, cam_up_world);
        T position[3] = {
            state_position_world[0] + cam_pos_world[0] + scene_translation_arr[env_i * 3 + 0],
            state_position_world[1] + cam_pos_world[1] + scene_translation_arr[env_i * 3 + 1],
            state_position_world[2] + cam_pos_world[2] + scene_translation_arr[env_i * 3 + 2]
        };
        T look_at[3] = {
            position[0] + cam_forward_world[0],
            position[1] + cam_forward_world[1],
            position[2] + cam_forward_world[2]
        };
        T up[3] = {cam_up_world[0], cam_up_world[1], cam_up_world[2]};
        gpu_cameras[env_i] = rlt::make_camera_data(position, look_at, up, fov, aspect);
    }

    template<typename DEVICE>
    __global__
    void make_target_cameras_kernel(
        DEVICE device,
        rlt::rendering::raytracing::CameraData<T>* target_cameras,
        T fov, T aspect,
        T camera_offset_body_0, T camera_offset_body_1, T camera_offset_body_2,
        T camera_forward_body_0, T camera_forward_body_1, T camera_forward_body_2,
        T camera_up_body_0, T camera_up_body_1, T camera_up_body_2,
        T* scene_translation_arr, T* scene_yaw_cos_arr, T* scene_yaw_sin_arr
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        T offset_body[3] = {camera_offset_body_0, camera_offset_body_1, camera_offset_body_2};
        T forward_body[3] = {camera_forward_body_0, camera_forward_body_1, camera_forward_body_2};
        T up_body[3] = {camera_up_body_0, camera_up_body_1, camera_up_body_2};
        T c = scene_yaw_cos_arr[env_i];
        T s = scene_yaw_sin_arr[env_i];
        auto rotate_scene_yaw = [&](const T in[3], T out[3]){
            out[0] = c * in[0] - s * in[1];
            out[1] = s * in[0] + c * in[1];
            out[2] = in[2];
        };
        T offset_world[3];
        rotate_scene_yaw(offset_body, offset_world);
        T forward_world[3];
        rotate_scene_yaw(forward_body, forward_world);
        T up_world[3];
        rotate_scene_yaw(up_body, up_world);
        T position[3] = {
            scene_translation_arr[env_i * 3 + 0] + offset_world[0],
            scene_translation_arr[env_i * 3 + 1] + offset_world[1],
            scene_translation_arr[env_i * 3 + 2] + offset_world[2]
        };
        T look_at[3] = {
            position[0] + forward_world[0],
            position[1] + forward_world[1],
            position[2] + forward_world[2]
        };
        T up[3] = {up_world[0], up_world[1], up_world[2]};
        target_cameras[env_i] = rlt::make_camera_data(position, look_at, up, fov, aspect);
    }

    __global__
    void mse_batch_loss_kernel(const T_ACTIVATION* student_output_ptr, const T_ACTIVATION* target_ptr, T* loss_out, TI n_elements){
        if(blockIdx.x == 0 && threadIdx.x == 0){
            T acc = 0;
            for(TI i = 0; i < n_elements; i++){
                T diff = static_cast<T>(student_output_ptr[i]) - static_cast<T>(target_ptr[i]);
                acc += diff * diff;
            }
            loss_out[0] = n_elements > 0 ? acc / static_cast<T>(n_elements) : static_cast<T>(0);
        }
    }

    __global__
    void reduce_episode_stats_kernel(
        const T* episode_lengths_log,
        const T* episode_tf_log,
        const TI* episode_step_arr,
        const bool* teacher_forcing_arr,
        T* stats_out
    ){
        if(blockIdx.x == 0 && threadIdx.x == 0){
            T tf_length_sum = 0;
            T tf_episode_count = 0;
            T student_length_sum = 0;
            T student_episode_count = 0;
            for(TI pos = 0; pos < STEPS_TOTAL; pos++){
                T episode_length = episode_lengths_log[pos];
                if(episode_length >= (T)0){
                    if(episode_tf_log[pos] > (T)0.5){
                        tf_length_sum += episode_length;
                        tf_episode_count += (T)1;
                    } else {
                        student_length_sum += episode_length;
                        student_episode_count += (T)1;
                    }
                }
            }
            for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                TI episode_step = episode_step_arr[env_i];
                if(episode_step > 0){
                    if(teacher_forcing_arr[env_i]){
                        tf_length_sum += (T)episode_step;
                        tf_episode_count += (T)1;
                    } else {
                        student_length_sum += (T)episode_step;
                        student_episode_count += (T)1;
                    }
                }
            }
            stats_out[0] = tf_length_sum;
            stats_out[1] = tf_episode_count;
            stats_out[2] = student_length_sum;
            stats_out[3] = student_episode_count;
        }
    }

#ifdef USE_FRAME_STACKING
    template<typename DEVICE>
    __global__
    void record_episode_start_kernel(DEVICE device, TI* episode_start_step, TI* episode_start_step_per_row, TI step_i){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        episode_start_step_per_row[step_i * N_ENVIRONMENTS + env_i] = episode_start_step[env_i];
    }

    __global__
    void compute_gather_indices_kernel(
        int* gather_indices,
        TI* episode_start_step_per_row,
        TI batch_offset, TI batch_size
    ){
        TI s = threadIdx.x + blockIdx.x * blockDim.x;
        if(s >= batch_size) return;
        TI row = batch_offset + s;
        TI env_i = row % N_ENVIRONMENTS;
        TI step_i_local = row / N_ENVIRONMENTS;
        TI ep_start = episode_start_step_per_row[row];
        for(TI f = 0; f < FRAME_STACK_N; f++){
            TI back = f * FRAME_STACK_STRIDE;
            TI desired_step = step_i_local >= back ? step_i_local - back : ep_start;
            if(desired_step < ep_start) desired_step = ep_start;
            gather_indices[s * FRAME_STACK_N + f] = (int)(desired_step * N_ENVIRONMENTS + env_i);
        }
    }

    __global__
    void compute_gather_indices_rollout_kernel(
        int* gather_indices,
        TI* episode_start_step,
        TI step_i, TI batch_size
    ){
        TI s = threadIdx.x + blockIdx.x * blockDim.x;
        if(s >= batch_size) return;
        TI env_i = s < N_ENVIRONMENTS ? s : 0;
        TI ep_start = s < N_ENVIRONMENTS ? episode_start_step[env_i] : episode_start_step[0];
        for(TI f = 0; f < FRAME_STACK_N; f++){
            TI back = f * FRAME_STACK_STRIDE;
            TI desired_step = step_i >= back ? step_i - back : ep_start;
            if(desired_step < ep_start) desired_step = ep_start;
            gather_indices[s * FRAME_STACK_N + f] = (int)(desired_step * N_ENVIRONMENTS + env_i);
        }
    }

#ifdef STACK_TARGET_CHANNEL
#endif
#endif

    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename STATE_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    __global__
    void raptor_evaluate_step_kernel(
        DEVICE device,
        const rlt::nn_models::sequential::ModuleForward<MODEL_SPEC> model,
        rlt::Tensor<INPUT_SPEC> input,
        rlt::nn_models::sequential::ModuleState<STATE_SPEC> state,
        rlt::Tensor<OUTPUT_SPEC> output,
        rlt::nn_models::sequential::ModuleBuffer<BUFFER_SPEC> buffers,
        RNG rng,
        rlt::Mode<MODE> mode
    ){
        rlt::_evaluate_step(device, model, input, state, state.content_state, output, buffers, buffers.content_buffer, rng, mode);
    }
}

struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
    static constexpr T ALPHA = 1e-3;
    static constexpr T EPSILON = 1e-5;
    static constexpr T EPSILON_SQRT = 1e-5;
};

template<typename CAPABILITY, typename T_TYPE_POLICY = TYPE_POLICY>
struct StudentActor{
#ifdef USE_GRU_TEMPORAL
    static constexpr TI STEPS = BPTT_STEPS;
    static constexpr TI FORWARD_BATCH_SIZE = N_ENVIRONMENTS;
#else
    static constexpr TI STEPS = 1;
    static constexpr TI FORWARD_BATCH_SIZE = BATCH_SIZE;
#endif

    static constexpr TI IMG_H = ENVIRONMENT::Observation::HEIGHT;
    static constexpr TI IMG_W = ENVIRONMENT::Observation::WIDTH;
    static constexpr TI IMG_C = ENVIRONMENT::Observation::CHANNELS;

#ifdef STACK_TARGET_CHANNEL
    using IMAGE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
#else
    using TARGET_IMAGE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, IMG_H, IMG_W, STACKED_IMG_C>;
    using IMAGE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, IMG_H, IMG_W, STACKED_IMG_C>;
#endif
    using STATE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, STATE_OBS_DIM>;

    using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 16, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
    using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 32, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
    using CONV3_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 64, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV3 = rlt::nn::layers::conv2d::BindConfiguration<CONV3_CONFIG>;
    using CONV4_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 128, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV4 = rlt::nn::layers::conv2d::BindConfiguration<CONV4_CONFIG>;
    using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, TI>;
    using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
    using IMAGE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using IMAGE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<IMAGE_DENSE_EMBED_CONFIG>;
    using IMAGE_BRANCH = rlt::nn_models::sequential::Module<CONV1, CONV2, CONV3, CONV4, OUTPUT_FLATTEN, IMAGE_DENSE_EMBED>;
#ifndef STACK_TARGET_CHANNEL
    using TARGET_IMAGE_BRANCH = rlt::nn_models::sequential::Module<INPUT_FLATTEN, IMAGE_STANDARDIZE, UNFLATTEN, CONV1, CONV2, CONV3, CONV4, OUTPUT_FLATTEN, IMAGE_DENSE_EMBED>;
#endif

    // State branch: Standardize→Dense(64)
    using STATE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, TI>;
    using STATE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STATE_STANDARDIZE_CONFIG>;
    using STATE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using STATE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<STATE_DENSE_EMBED_CONFIG>;
    using STATE_BRANCH = rlt::nn_models::sequential::Module<STATE_STANDARDIZE, STATE_DENSE_EMBED>;

    using HEAD_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using HEAD_DENSE1 = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE1_CONFIG>;
    using HEAD_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using HEAD_DENSE2 = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE2_CONFIG>;
    using HEAD_DENSE_OUT_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, TARGET_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD_DENSE_OUT = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE_OUT_CONFIG>;
#ifdef USE_GRU_TEMPORAL
    using GRU_HEAD_CONFIG = rlt::nn::layers::gru::Configuration<T_TYPE_POLICY, TI, GRU_HIDDEN_DIM>;
    using GRU_HEAD = rlt::nn::layers::gru::BindConfiguration<GRU_HEAD_CONFIG>;
    using SEQUENTIAL_HEAD = rlt::nn_models::sequential::Module<GRU_HEAD, HEAD_DENSE1, HEAD_DENSE2, HEAD_DENSE_OUT>;
#ifdef STACK_TARGET_CHANNEL
    using BRANCH_IMAGE = rlt::nn_models::parallel::Branch<IMAGE_BRANCH, IMAGE_INPUT_SHAPE>;
    using BRANCH_STATE = rlt::nn_models::parallel::Branch<STATE_BRANCH, STATE_INPUT_SHAPE>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, SEQUENTIAL_HEAD, BRANCH_IMAGE, BRANCH_STATE>;
#else
    using BRANCH_TARGET_IMAGE = rlt::nn_models::parallel::Branch<TARGET_IMAGE_BRANCH, TARGET_IMAGE_INPUT_SHAPE>;
    using BRANCH_IMAGE = rlt::nn_models::parallel::Branch<IMAGE_BRANCH, IMAGE_INPUT_SHAPE>;
    using BRANCH_STATE = rlt::nn_models::parallel::Branch<STATE_BRANCH, STATE_INPUT_SHAPE>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, SEQUENTIAL_HEAD, BRANCH_TARGET_IMAGE, BRANCH_IMAGE, BRANCH_STATE>;
#endif
#else
    using SEQUENTIAL_HEAD = rlt::nn_models::sequential::Module<HEAD_DENSE1, HEAD_DENSE2, HEAD_DENSE_OUT>;
#ifdef STACK_TARGET_CHANNEL
    using BRANCH_IMAGE = rlt::nn_models::parallel::Branch<IMAGE_BRANCH, IMAGE_INPUT_SHAPE>;
    using BRANCH_STATE = rlt::nn_models::parallel::Branch<STATE_BRANCH, STATE_INPUT_SHAPE>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, SEQUENTIAL_HEAD, BRANCH_IMAGE, BRANCH_STATE>;
#else
    using BRANCH_TARGET_IMAGE = rlt::nn_models::parallel::Branch<TARGET_IMAGE_BRANCH, TARGET_IMAGE_INPUT_SHAPE>;
    using BRANCH_IMAGE = rlt::nn_models::parallel::Branch<IMAGE_BRANCH, IMAGE_INPUT_SHAPE>;
    using BRANCH_STATE = rlt::nn_models::parallel::Branch<STATE_BRANCH, STATE_INPUT_SHAPE>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, SEQUENTIAL_HEAD, BRANCH_TARGET_IMAGE, BRANCH_IMAGE, BRANCH_STATE>;
#endif
#endif
};

using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, true>;
using STUDENT_TYPE = typename StudentActor<CAPABILITY_ADAM>::MODEL;
using STUDENT_BUFFERS = typename STUDENT_TYPE::Buffer<true>;
using CAPABILITY_FORWARD_CPU = rlt::nn::capability::Forward<true>;
using CPU_STUDENT_TYPE = typename StudentActor<CAPABILITY_FORWARD_CPU, TEACHER_TYPE_POLICY>::MODEL;
using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, ADAM_PARAMETERS, true>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;

static constexpr TI IMG_H = ENVIRONMENT::Observation::HEIGHT;
static constexpr TI IMG_W = ENVIRONMENT::Observation::WIDTH;
static constexpr TI IMG_C = ENVIRONMENT::Observation::CHANNELS;

// =========================================================================
// Main
// =========================================================================
// Parse hex hash string (40 chars) into 20 bytes
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

template<typename DEVICE, typename OBS_SPEC>
void render_target_observation(
    DEVICE& device,
    ENVIRONMENT& env,
    const typename ENVIRONMENT::Parameters& parameters,
    rlt::Matrix<OBS_SPEC>& observation
){
    static_assert(OBS_SPEC::ROWS == 1);
    static_assert(OBS_SPEC::COLS == OBSERVATION_DIM);

    typename ENVIRONMENT::State target_state = {};
    target_state.orientation[0] = (T)1;
    target_state.orientation[1] = (T)0;
    target_state.orientation[2] = (T)0;
    target_state.orientation[3] = (T)0;
    auto camera = rlt::rl::environments::l2f_visual::make_camera_for_state(device, env, parameters, target_state);

    for(TI camera_i = 0; camera_i < N_ENVIRONMENTS_PER_SCENE; camera_i++){
        rlt::set(device, env.renderer->cameras, camera, camera_i);
    }
    rlt::set_cameras(device, *env.renderer, env.renderer->cameras);
    rlt::render(device, *env.renderer);
    rlt::read_frame_buffer(device, *env.renderer, env.renderer->frame_buffer);

    constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    const uint32_t* fb_data = rlt::data(env.renderer->frame_buffer);
    for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
        const uint32_t rgba = fb_data[pixel_i];
        rlt::set(observation, 0, pixel_i * 3 + 0, static_cast<T>((rgba >>  0) & 0xFF) / static_cast<T>(255));
        rlt::set(observation, 0, pixel_i * 3 + 1, static_cast<T>((rgba >>  8) & 0xFF) / static_cast<T>(255));
        rlt::set(observation, 0, pixel_i * 3 + 2, static_cast<T>((rgba >> 16) & 0xFF) / static_cast<T>(255));
    }
}

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

__global__ void brightness_apply_kernel(float* __restrict__ obs, const float* __restrict__ brightness_scales, int num_envs, int obs_dim){
    int env_i = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(env_i >= num_envs || idx >= obs_dim) return;
    float scale = brightness_scales[env_i];
    int global_idx = env_i * obs_dim + idx;
    obs[global_idx] = fminf(fmaxf(obs[global_idx] * scale, 0.0f), 1.0f);
}

__global__ void observation_noise_kernel(float* __restrict__ obs, int n, float std, unsigned long long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    curandState rng_state;
    curand_init(seed, idx, 0, &rng_state);
    float val = obs[idx] + curand_normal(&rng_state) * std;
    obs[idx] = fminf(fmaxf(val, 0.0f), 1.0f);
}

#ifdef USE_FRAME_STACKING
template<typename T_OUT>
__global__ void gather_frames_kernel(
    const float* __restrict__ all_obs,
    const int* __restrict__ gather_idx,
    T_OUT* __restrict__ stacked_out,
    int obs_dim, int img_c, int n_frames, int stacked_img_c, int stacked_obs_dim, int batch_size
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx >= batch_size * stacked_obs_dim) return;
    int sample = global_idx / stacked_obs_dim;
    int offset = global_idx % stacked_obs_dim;
    int pixel = offset / stacked_img_c;
    int frame_channel = offset % stacked_img_c;
    int frame = frame_channel / img_c;
    int channel = frame_channel % img_c;
    int src_row = gather_idx[sample * n_frames + frame];
    stacked_out[global_idx] = (T_OUT)all_obs[src_row * obs_dim + pixel * img_c + channel];
}

#ifdef STACK_TARGET_CHANNEL
template<typename T_OUT>
__global__ void gather_frames_with_target_kernel(
    const float* __restrict__ student_obs,
    const float* __restrict__ target_obs,
    const int* __restrict__ student_gather_idx,
    int target_row_base,
    T_OUT* __restrict__ combined_out,
    int obs_dim, int img_c, int n_frames, int combined_img_c, int combined_obs_dim, int batch_size
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx >= batch_size * combined_obs_dim) return;
    int sample = global_idx / combined_obs_dim;
    int offset = global_idx % combined_obs_dim;
    int pixel = offset / combined_img_c;
    int frame_channel = offset % combined_img_c;
    if(frame_channel < n_frames * img_c){
        int frame = frame_channel / img_c;
        int channel = frame_channel % img_c;
        int src_row = student_gather_idx[sample * n_frames + frame];
        combined_out[global_idx] = (T_OUT)student_obs[src_row * obs_dim + pixel * img_c + channel];
    } else {
        int channel = frame_channel - n_frames * img_c;
        int src_row = target_row_base + sample;
        combined_out[global_idx] = (T_OUT)target_obs[src_row * obs_dim + pixel * img_c + channel];
    }
}

template<typename T_OUT>
__global__ void build_frame_stacked_with_target_from_history_kernel(
    const float* __restrict__ history_obs,
    const float* __restrict__ target_obs,
    const TI* __restrict__ episode_start_step,
    TI step_i,
    T_OUT* __restrict__ combined_out,
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
        combined_out[global_idx] = (T_OUT)history_obs[src_row * obs_dim + pixel * img_c + channel];
    } else if(frame_channel < logical_channels) {
        int channel = frame_channel - n_frames * img_c;
        combined_out[global_idx] = (T_OUT)target_obs[sample * obs_dim + pixel * img_c + channel];
    } else {
        combined_out[global_idx] = (T_OUT)0.0f;
    }
}
#endif
#endif

#ifdef STACK_TARGET_CHANNEL
template<typename T_OUT>
__global__ void concat_target_channels_kernel(
    const float* __restrict__ student_obs,
    const float* __restrict__ target_obs,
    T_OUT* __restrict__ combined_out,
    int obs_dim, int img_c, int combined_img_c, int combined_obs_dim, int batch_size
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx >= batch_size * combined_obs_dim) return;
    int sample = global_idx / combined_obs_dim;
    int offset = global_idx % combined_obs_dim;
    int pixel = offset / combined_img_c;
    int channel_in_combined = offset % combined_img_c;
    if(channel_in_combined < img_c){
        combined_out[global_idx] = (T_OUT)student_obs[sample * obs_dim + pixel * img_c + channel_in_combined];
    } else {
        int target_channel = channel_in_combined - img_c;
        combined_out[global_idx] = (T_OUT)target_obs[sample * obs_dim + pixel * img_c + target_channel];
    }
}
#endif

int main(int argc, char** argv){
#ifdef RL_TOOLS_DEBUG_CUDA_CHECK
#define CUDA_CHECK(msg) { cudaError_t e = cudaGetLastError(); if(e != cudaSuccess){ std::cerr << "CUDA ERROR [" << msg << "]: " << cudaGetErrorString(e) << std::endl; return 1; } e = cudaDeviceSynchronize(); if(e != cudaSuccess){ std::cerr << "CUDA SYNC ERROR [" << msg << "]: " << cudaGetErrorString(e) << std::endl; return 1; } }
#else
#define CUDA_CHECK(msg) ((void)0)
#endif
    TI seed = 0;
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <scene_directory or scene.glb> [seed]" << std::endl;
        return 1;
    }
    if(argc > 2){
        seed = std::atoi(argv[2]);
    }

    static constexpr TI EPOCHS_PER_SCENE = 50;
    static constexpr TI RENDER_TIMING_SAMPLE_PERIOD = 16;

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
        for(TI i = 0; i < N_TOTAL_SCENES; i++){
            scene_paths.push_back(all_glbs[i]);
        }
        std::cout << "Selected " << scene_paths.size() << " scenes from " << scene_arg << std::endl;
        for(TI i = 0; i < scene_paths.size(); i++){
            std::cout << "  [" << i << "] " << std::filesystem::path(scene_paths[i]).filename().string() << std::endl;
        }
    } else {
        scene_paths.resize(N_TOTAL_SCENES, scene_arg);
        std::cout << "Replicating single scene across " << N_TOTAL_SCENES << " renderers: " << scene_arg << std::endl;
    }
    if(scene_paths.size() != N_TOTAL_SCENES){
        std::cerr << "Expected exactly " << N_TOTAL_SCENES << " scenes, got " << scene_paths.size() << std::endl;
        return 1;
    }

    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device);

    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = "l2f_visual_imitation_cuda";
    rlt::init(device, extrack_config, extrack_paths, seed);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);

    // =========================================================================
    // RAPTOR teacher (CPU)
    // =========================================================================
    RAPTOR_MODEL raptor;
    typename RAPTOR_MODEL::Buffer<true> raptor_buffer;
    typename RAPTOR_MODEL::State<true> raptor_state;
    rlt::malloc(device, raptor);
    rlt::malloc(device, raptor_buffer);
    rlt::malloc(device, raptor_state);
    rlt::copy(device, device, rl_tools::checkpoint::actor::module, raptor);
    rlt::reset(device, raptor, raptor_state, rng);

    // RAPTOR input/output tensors (CPU)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, RAPTOR_OBS_DIM>>> teacher_obs;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTION_DIM>>> teacher_actions;
    rlt::malloc(device, teacher_obs);
    rlt::malloc(device, teacher_actions);

    // CPU state observations buffer (per step)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, STATE_OBS_DIM>>> cpu_state_obs_step;
    rlt::malloc(device, cpu_state_obs_step);

    // All teacher actions for entire epoch (CPU)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, ACTION_DIM>>> cpu_all_teacher_actions;
    rlt::malloc(device, cpu_all_teacher_actions);

    // =========================================================================
    // Student (CPU copy for warmup, float for CPU compatibility)
    // =========================================================================
    using CPU_STUDENT_INIT_TYPE = typename StudentActor<CAPABILITY_ADAM, TEACHER_TYPE_POLICY>::MODEL;
    CPU_STUDENT_INIT_TYPE student_cpu;
    rlt::malloc(device, student_cpu);
    rlt::init_weights(device, student_cpu, rng);

    // =========================================================================
    // Environment setup (N renderers for N scenes)
    // =========================================================================
    using RENDERER_TYPE = rlt::rendering::raytracing::Renderer<typename ENVIRONMENT::SPEC::RENDERER_SPEC>;
    using SCENE_TYPE = rlt::rendering::raytracing::scene::procthor::Scene<typename ENVIRONMENT::SPEC::SCENE_SPEC>;

    ENVIRONMENT envs[N_ENVIRONMENTS];
    typename ENVIRONMENT::Parameters env_parameters[N_ENVIRONMENTS];

    std::array<RENDERER_TYPE*, N_TOTAL_SCENES> renderers{};
    std::array<SCENE_TYPE*, N_TOTAL_SCENES> scenes{};

    // Load all scenes
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

    // GPU buffers for per-scene indoor positions
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
            std::cout << "  Active scene slot " << active_scene_i << " -> scene " << actual_scene_i << std::endl;
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

    // Set up envs (use first renderer for env0 reference)
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        TI scene_i = env_i / N_ENVIRONMENTS_PER_SCENE;
        rlt::malloc(device, envs[env_i].dynamics);
        envs[env_i].renderer = renderers[scene_i];
        envs[env_i].scene = scenes[scene_i];
        envs[env_i].owns_renderer = false;
        envs[env_i].renderer_initialized = true;
        envs[env_i].use_target_mode = true;
        rlt::initial_parameters(device, envs[env_i], env_parameters[env_i]);
        env_parameters[env_i].scene_translation[0] = 0;
        env_parameters[env_i].scene_translation[1] = 0;
        env_parameters[env_i].scene_translation[2] = 0;
    }
    auto& env0 = envs[0];

    {
        std::string ui = rlt::get_ui(device, envs[0].dynamics);
        if(!ui.empty()){
            std::filesystem::create_directories(extrack_paths.seed);
            std::ofstream ui_file(extrack_paths.seed / "ui.esm.js");
            ui_file << ui;
        }
    }

    EpisodeRecorder episode_recorders[TRAJECTORY_NUM_ENVS];
    std::vector<std::vector<TrajectoryStep>> completed_episodes;
    T simulation_dt = static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY);
    TI global_step = 0;

    // =========================================================================
    // GPU init
    // =========================================================================
    rlt::init(device_gpu);
    RNG_GPU rng_gpu;
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_gpu, rng_gpu, seed);

    STUDENT_TYPE student_gpu;
    STUDENT_BUFFERS student_buffers;
    OPTIMIZER optimizer_gpu;
    rlt::malloc(device_gpu, student_gpu);
    rlt::malloc(device_gpu, student_buffers);
    rlt::malloc(device_gpu, optimizer_gpu);
    rlt::copy(device, device_gpu, student_cpu, student_gpu);
    rlt::init(device_gpu, optimizer_gpu);
    rlt::reset_optimizer_state(device_gpu, optimizer_gpu, student_gpu);
    using ROLLOUT_STUDENT_TYPE = typename STUDENT_TYPE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>;
    ROLLOUT_STUDENT_TYPE rollout_student_gpu;
    typename ROLLOUT_STUDENT_TYPE::template Buffer<true> rollout_student_buffers;
    rlt::malloc(device_gpu, rollout_student_gpu);
    rlt::malloc(device_gpu, rollout_student_buffers);
    rlt::copy(device, device_gpu, student_cpu, rollout_student_gpu);
#ifdef USE_GRU_TEMPORAL
    typename ROLLOUT_STUDENT_TYPE::template State<true> rollout_student_state_gpu;
    rlt::malloc(device_gpu, rollout_student_state_gpu);
    rlt::reset(device_gpu, rollout_student_gpu, rollout_student_state_gpu, rng_gpu);
#endif

    // GPU RAPTOR teacher
    RAPTOR_MODEL raptor_gpu;
    typename RAPTOR_MODEL::Buffer<true> raptor_buffer_gpu;
    typename RAPTOR_MODEL::State<true> raptor_state_gpu;
    rlt::malloc(device_gpu, raptor_gpu);
    rlt::malloc(device_gpu, raptor_buffer_gpu);
    rlt::malloc(device_gpu, raptor_state_gpu);
    rlt::copy(device, device_gpu, raptor, raptor_gpu);
    rlt::reset(device_gpu, raptor_gpu, raptor_state_gpu, rng_gpu);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, RAPTOR_OBS_DIM>>> gpu_teacher_obs;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTION_DIM>>> gpu_teacher_actions_step;
    rlt::malloc(device_gpu, gpu_teacher_obs);
    rlt::malloc(device_gpu, gpu_teacher_actions_step);

    // GPU cameras
    rlt::rendering::raytracing::CameraData<float>* gpu_cameras = nullptr;
    cudaMalloc(&gpu_cameras, N_ENVIRONMENTS * sizeof(rlt::rendering::raytracing::CameraData<float>));
    rlt::rendering::raytracing::CameraData<float>* gpu_target_cameras = nullptr;
    cudaMalloc(&gpu_target_cameras, N_ENVIRONMENTS * sizeof(rlt::rendering::raytracing::CameraData<float>));

    // Event for cross-stream synchronization (make_cameras on device_gpu.stream → optix_stream)
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
    std::vector<cudaEvent_t> render_pass_start_events(STEPS_PER_ENV * N_ACTIVE_SCENES);
    std::vector<cudaEvent_t> render_pass_stop_events(STEPS_PER_ENV * N_ACTIVE_SCENES);
    std::vector<cudaEvent_t> target_render_pass_start_events(STEPS_PER_ENV * N_ACTIVE_SCENES);
    std::vector<cudaEvent_t> target_render_pass_stop_events(STEPS_PER_ENV * N_ACTIVE_SCENES);
    for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            TI event_i = step_i * N_ACTIVE_SCENES + active_scene_i;
            cudaEventCreate(&render_pass_start_events[event_i]);
            cudaEventCreate(&render_pass_stop_events[event_i]);
            cudaEventCreate(&target_render_pass_start_events[event_i]);
            cudaEventCreate(&target_render_pass_stop_events[event_i]);
        }
    }
#ifdef USE_GRU_TEMPORAL
    TI max_train_timing_calls = N_TRAIN_PASSES * N_WINDOWS;
    TI max_logged_loss_calls = N_WINDOWS;
#else
    TI max_train_timing_calls = N_TRAIN_PASSES * N_BATCHES;
    TI max_logged_loss_calls = N_BATCHES;
#endif
    std::vector<cudaEvent_t> train_forward_start_events(max_train_timing_calls);
    std::vector<cudaEvent_t> train_forward_stop_events(max_train_timing_calls);
    std::vector<cudaEvent_t> train_backward_start_events(max_train_timing_calls);
    std::vector<cudaEvent_t> train_backward_stop_events(max_train_timing_calls);
    std::vector<cudaEvent_t> train_update_start_events(max_train_timing_calls);
    std::vector<cudaEvent_t> train_update_stop_events(max_train_timing_calls);
    for(TI call_i = 0; call_i < max_train_timing_calls; call_i++){
        cudaEventCreate(&train_forward_start_events[call_i]);
        cudaEventCreate(&train_forward_stop_events[call_i]);
        cudaEventCreate(&train_backward_start_events[call_i]);
        cudaEventCreate(&train_backward_stop_events[call_i]);
        cudaEventCreate(&train_update_start_events[call_i]);
        cudaEventCreate(&train_update_stop_events[call_i]);
    }
    T* gpu_logged_batch_losses = nullptr;
    cudaMalloc(&gpu_logged_batch_losses, max_logged_loss_calls * sizeof(T));
    std::vector<T> cpu_logged_batch_losses(max_logged_loss_calls);
    T* gpu_epoch_episode_stats = nullptr;
    cudaMalloc(&gpu_epoch_episode_stats, 4 * sizeof(T));
    std::array<T, 4> cpu_epoch_episode_stats{};

    // GPU tensors
    static constexpr TI GPU_OBS_ROWS = STEPS_TOTAL + BATCH_SIZE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, OBSERVATION_DIM>>> gpu_all_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, OBSERVATION_DIM>>> gpu_all_target_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, STATE_OBS_DIM>>> gpu_all_state_observations;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, TARGET_DIM>>> gpu_all_targets;
#ifdef USE_GRU_TEMPORAL
    rlt::Matrix<rlt::matrix::Specification<T_GRADIENT, TI, WINDOW_SAMPLES, TARGET_DIM>> gpu_d_action_train;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, TARGET_DIM>>> gpu_student_output_train;
#else
    rlt::Matrix<rlt::matrix::Specification<T_GRADIENT, TI, BATCH_SIZE, TARGET_DIM>> gpu_d_action_train;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, TARGET_DIM>>> gpu_student_output_train;
#endif
    rlt::malloc(device_gpu, gpu_all_observations);
    rlt::malloc(device_gpu, gpu_all_target_observations);
    rlt::malloc(device_gpu, gpu_all_state_observations);
    rlt::malloc(device_gpu, gpu_all_targets);
    rlt::malloc(device_gpu, gpu_d_action_train);
    rlt::malloc(device_gpu, gpu_student_output_train);

    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTION_DIM>>> gpu_student_actions_step;
    rlt::malloc(device_gpu, gpu_student_actions_step);

#ifdef USE_GRU_TEMPORAL
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTOR_HIDDEN_DIM>>> gpu_rollout_branch_a;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTOR_HIDDEN_DIM>>> gpu_rollout_branch_b;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, EMBED_DIM>>> gpu_rollout_concat;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTION_DIM>>> gpu_rollout_actions;
    rlt::malloc(device_gpu, gpu_rollout_branch_a);
    rlt::malloc(device_gpu, gpu_rollout_branch_b);
    rlt::malloc(device_gpu, gpu_rollout_concat);
    rlt::malloc(device_gpu, gpu_rollout_actions);
#endif

#if defined(STACK_TARGET_CHANNEL) && !defined(USE_FRAME_STACKING)
#ifdef USE_GRU_TEMPORAL
    static constexpr TI COMBINED_BUFFER_ROWS = WINDOW_SAMPLES;
#else
    static constexpr TI COMBINED_BUFFER_ROWS = N_ENVIRONMENTS;
#endif
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, COMBINED_BUFFER_ROWS, COMBINED_OBS_DIM>>> gpu_rollout_combined;
    rlt::malloc(device_gpu, gpu_rollout_combined);
#endif
#ifdef USE_FRAME_STACKING
#ifdef STACK_TARGET_CHANNEL
    static constexpr TI FRAME_STACK_HISTORY_ROWS = FRAME_STACK_HISTORY_LENGTH * N_ENVIRONMENTS;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, FRAME_STACK_HISTORY_ROWS, OBSERVATION_DIM>>> gpu_frame_stack_history;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, COMBINED_OBS_DIM>>> gpu_all_combined_observations;
    rlt::malloc(device_gpu, gpu_frame_stack_history);
    {
        size_t combined_observations_bytes = static_cast<size_t>(STEPS_TOTAL) * static_cast<size_t>(COMBINED_OBS_DIM) * sizeof(T_ACTIVATION);
        cudaError_t allocation_status = cudaMalloc((void**)rlt::data_pointer(gpu_all_combined_observations), combined_observations_bytes);
        if(allocation_status != cudaSuccess){
            std::cerr << "Failed to allocate gpu_all_combined_observations: " << cudaGetErrorString(allocation_status) << std::endl;
            return 1;
        }
    }
#else
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, BATCH_SIZE, STACKED_OBS_DIM>>> gpu_stacked_batch;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, BATCH_SIZE, STACKED_OBS_DIM>>> gpu_stacked_target_batch;
    rlt::malloc(device_gpu, gpu_stacked_batch);
    rlt::malloc(device_gpu, gpu_stacked_target_batch);
#endif
    TI* gpu_episode_start_step = nullptr;
    cudaMalloc(&gpu_episode_start_step, N_ENVIRONMENTS * sizeof(TI));
    cudaMemset(gpu_episode_start_step, 0, N_ENVIRONMENTS * sizeof(TI));
#ifndef STACK_TARGET_CHANNEL
    int* gpu_gather_indices = nullptr;
    TI* gpu_episode_start_step_per_row = nullptr;
    cudaMalloc(&gpu_gather_indices, BATCH_SIZE * FRAME_STACK_N * sizeof(int));
    cudaMalloc(&gpu_episode_start_step_per_row, STEPS_TOTAL * sizeof(TI));
    int* gpu_target_gather_indices = nullptr;
    cudaMalloc(&gpu_target_gather_indices, BATCH_SIZE * FRAME_STACK_N * sizeof(int));
    int* gpu_rollout_gather_indices = nullptr;
    cudaMalloc(&gpu_rollout_gather_indices, N_ENVIRONMENTS * FRAME_STACK_N * sizeof(int));
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, STACKED_OBS_DIM>>> gpu_rollout_stacked;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, STACKED_OBS_DIM>>> gpu_rollout_stacked_target;
    rlt::malloc(device_gpu, gpu_rollout_stacked);
    rlt::malloc(device_gpu, gpu_rollout_stacked_target);
    int* gpu_rollout_target_gather_indices = nullptr;
    cudaMalloc(&gpu_rollout_target_gather_indices, N_ENVIRONMENTS * FRAME_STACK_N * sizeof(int));
#endif
#endif

    // =========================================================================
    // GPU-resident environment state
    // =========================================================================
    using DYNAMICS_TYPE = ENVIRONMENT::DYNAMICS_ENV;
    DYNAMICS_TYPE* gpu_dynamics_arr = nullptr;
    PARAMETERS_TYPE* gpu_params_arr = nullptr;
    typename ENVIRONMENT::State* gpu_states_arr = nullptr;
    bool* gpu_terminated_arr = nullptr;
    TI* gpu_episode_step_arr = nullptr;
    bool* gpu_teacher_forcing_arr = nullptr;
    T* gpu_episode_return_arr = nullptr;
    bool* gpu_needs_reset = nullptr;
    T* gpu_episode_lengths_log = nullptr;
    T* gpu_episode_tf_log = nullptr;
    T* gpu_brightness_scale_arr = nullptr;
    T* gpu_scene_translation_arr = nullptr;
    T* gpu_scene_yaw_arr = nullptr;
    T* gpu_scene_yaw_cos_arr = nullptr;
    T* gpu_scene_yaw_sin_arr = nullptr;
    cudaMalloc(&gpu_dynamics_arr, N_ENVIRONMENTS * sizeof(DYNAMICS_TYPE));
    cudaMalloc(&gpu_params_arr, N_ENVIRONMENTS * sizeof(PARAMETERS_TYPE));
    cudaMalloc(&gpu_states_arr, N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::State));
    cudaMalloc(&gpu_terminated_arr, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_step_arr, N_ENVIRONMENTS * sizeof(TI));
    cudaMalloc(&gpu_teacher_forcing_arr, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_return_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_needs_reset, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_lengths_log, STEPS_TOTAL * sizeof(T));
    cudaMalloc(&gpu_episode_tf_log, STEPS_TOTAL * sizeof(T));
    cudaMalloc(&gpu_brightness_scale_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_scene_translation_arr, N_ENVIRONMENTS * 3 * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_cos_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_sin_arr, N_ENVIRONMENTS * sizeof(T));
    {
        std::vector<T> ones(N_ENVIRONMENTS, (T)1);
        cudaMemcpy(gpu_brightness_scale_arr, ones.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        std::vector<T> scene_translations(N_ENVIRONMENTS * 3, (T)0);
        cudaMemcpy(gpu_scene_translation_arr, scene_translations.data(), N_ENVIRONMENTS * 3 * sizeof(T), cudaMemcpyHostToDevice);
        std::vector<T> scene_yaws(N_ENVIRONMENTS, (T)0);
        cudaMemcpy(gpu_scene_yaw_arr, scene_yaws.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_scene_yaw_cos_arr, ones.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        std::vector<T> zeros(N_ENVIRONMENTS, (T)0);
        cudaMemcpy(gpu_scene_yaw_sin_arr, zeros.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
    }
    typename ENVIRONMENT::State cpu_states_for_cameras[N_ENVIRONMENTS];
    {
        DYNAMICS_TYPE cpu_dynamics[N_ENVIRONMENTS];
        PARAMETERS_TYPE cpu_params[N_ENVIRONMENTS];
        for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
            cpu_dynamics[env_i] = envs[env_i].dynamics;
            cpu_params[env_i] = env_parameters[env_i].dynamics;
        }
        cudaMemcpy(gpu_dynamics_arr, cpu_dynamics, N_ENVIRONMENTS * sizeof(DYNAMICS_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_params_arr, cpu_params, N_ENVIRONMENTS * sizeof(PARAMETERS_TYPE), cudaMemcpyHostToDevice);
    }
    {
        bool init_terminated[N_ENVIRONMENTS];
        TI init_step[N_ENVIRONMENTS];
        T init_return[N_ENVIRONMENTS];
        bool init_teacher_forcing[N_ENVIRONMENTS];
        for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
            init_terminated[env_i] = true;
            init_step[env_i] = 0;
            init_return[env_i] = (T)0;
            init_teacher_forcing[env_i] = true;
        }
        cudaMemcpy(gpu_terminated_arr, init_terminated, N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_step_arr, init_step, N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_return_arr, init_return, N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_teacher_forcing_arr, init_teacher_forcing, N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
    }
    T episode_length_sum_tf = 0;
    TI episode_count_tf = 0;
    T episode_length_sum_student = 0;
    TI episode_count_student = 0;

    // =========================================================================
    // Training loop
    // =========================================================================
    std::cout << "Starting imitation learning (visual L2F hover, CUDA)" << std::endl;
    std::cout << "  N_ENVIRONMENTS: " << N_ENVIRONMENTS << std::endl;
    std::cout << "  STEPS_PER_ENV: " << STEPS_PER_ENV << std::endl;
    std::cout << "  BATCH_SIZE: " << BATCH_SIZE << std::endl;
    std::cout << "  N_BATCHES: " << N_BATCHES << std::endl;
    std::cout << "  N_TRAIN_PASSES: " << N_TRAIN_PASSES << std::endl;
    std::cout << "  OBSERVATION_DIM (image): " << OBSERVATION_DIM << std::endl;
    std::cout << "  STATE_OBS_DIM: " << STATE_OBS_DIM << std::endl;
    std::cout << "  RAPTOR_OBS_DIM: " << RAPTOR_OBS_DIM << std::endl;
    std::cout << "  TEACHER_FORCING_EPOCHS: " << TEACHER_FORCING_EPOCHS << std::endl;
    std::cout << "  TEACHER_FORCING_FRACTION: " << TEACHER_FORCING_FRACTION << std::endl;
#ifdef USE_FRAME_STACKING
    std::cout << "  FRAME_STACK_N: " << FRAME_STACK_N << std::endl;
    std::cout << "  FRAME_STACK_STRIDE: " << FRAME_STACK_STRIDE << " (" << (FRAME_STACK_STRIDE > 0 ? SIMULATION_FREQUENCY / FRAME_STACK_STRIDE : SIMULATION_FREQUENCY) << " Hz)" << std::endl;
    std::cout << "  STACKED_IMG_C: " << STACKED_IMG_C << std::endl;
#ifdef STACK_TARGET_CHANNEL
    std::cout << "  COMBINED_IMG_C: " << COMBINED_IMG_C << std::endl;
#endif
#elif defined(USE_GRU_TEMPORAL)
    std::cout << "  BPTT_STEPS: " << BPTT_STEPS << std::endl;
    std::cout << "  GRU_HIDDEN_DIM: " << GRU_HIDDEN_DIM << std::endl;
    std::cout << "  N_WINDOWS: " << N_WINDOWS << std::endl;
#endif

    using NO_AUTO_RESET_MODE = rlt::Mode<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Default<>>>;
    NO_AUTO_RESET_MODE no_auto_reset_mode;

    auto training_start = std::chrono::high_resolution_clock::now();

    // Video recording buffers
    static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    static constexpr TI MOSAIC_W = SCENE_GRID_COLS * ENV_GRID_SIDE * CAM_WIDTH * 2;
    static constexpr TI MOSAIC_H = SCENE_GRID_ROWS * ENV_GRID_SIDE * CAM_HEIGHT;
    std::vector<uint32_t> video_pixel_buffer(N_ENVIRONMENTS * CAM_PIXELS);
    std::vector<uint8_t> mosaic_frame(MOSAIC_W * MOSAIC_H * 3);

    auto curriculum_step_limit = [](TI epoch) -> TI {
        if(epoch < 100) return 50;
        if(epoch < 1000) return 100;
        if(epoch < 3000) return 200;
        return 500;
    };

    for(TI epoch_i = 0; epoch_i < NUM_EPOCHS; epoch_i++){
        TI current_episode_step_limit = curriculum_step_limit(epoch_i);
        auto epoch_start = std::chrono::high_resolution_clock::now();
        std::array<RENDERER_TYPE*, N_ACTIVE_SCENES> active_renderers{};
        std::array<cudaStream_t, N_ACTIVE_SCENES> active_scene_render_streams{};
        std::array<void*, N_ACTIVE_SCENES> active_scene_camera_buffers{};
        std::array<const uint32_t*, N_ACTIVE_SCENES> active_scene_framebuffer_ptrs{};
        std::shuffle(scene_permutation.begin(), scene_permutation.end(), scene_rng);
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            active_scene_indices[active_scene_i] = scene_permutation[active_scene_i];
        }
        upload_active_scenes();
        for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
            TI active_scene_i = env_i / N_ENVIRONMENTS_PER_SCENE;
            TI actual_scene_i = active_scene_indices[active_scene_i];
            envs[env_i].renderer = renderers[actual_scene_i];
            envs[env_i].scene = scenes[actual_scene_i];
        }
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            TI actual_scene_i = active_scene_indices[active_scene_i];
            auto* renderer = renderers[actual_scene_i];
            active_renderers[active_scene_i] = renderer;
            OWLParams rgb_lp = (OWLParams)renderer->backend.rgb_launch_params;
            active_scene_render_streams[active_scene_i] = (cudaStream_t)owlParamsGetCudaStream(rgb_lp, 0);
            active_scene_camera_buffers[active_scene_i] = (void*)owlBufferGetPointer((OWLBuffer)renderer->backend.owl_cameras_buffer, 0);
            active_scene_framebuffer_ptrs[active_scene_i] = rlt::get_framebuffer_device_ptr(device, *renderer);
        }
        {
            std::vector<unsigned char> reset_terminated(N_ENVIRONMENTS, 1);
            std::vector<TI> reset_step(N_ENVIRONMENTS, 0);
            std::vector<T> reset_return(N_ENVIRONMENTS, (T)0);
            std::vector<unsigned char> reset_teacher_forcing(N_ENVIRONMENTS, 1);
            cudaMemcpy(gpu_terminated_arr, reset_terminated.data(), N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_episode_step_arr, reset_step.data(), N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_episode_return_arr, reset_return.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_teacher_forcing_arr, reset_teacher_forcing.data(), N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
        }
#ifdef USE_GRU_TEMPORAL
        rlt::reset(device_gpu, rollout_student_gpu, rollout_student_state_gpu, rng_gpu);
#endif
        bool full_teacher_forcing = false;
        bool record_video = (epoch_i % CHECKPOINT_CADENCE == 0);
        TI epoch_end_step = global_step + STEPS_PER_ENV * N_ENVIRONMENTS;
        FILE* ffmpeg_pipe = nullptr;
        if(record_video){
            auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, epoch_end_step);
            auto video_path = step_folder / "video.mp4";
            char ffmpeg_cmd[1024];
            snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
                "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %lux%lu -framerate %lu -i - "
                "-c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast -loglevel warning %s",
                (unsigned long)MOSAIC_W, (unsigned long)MOSAIC_H, (unsigned long)SIMULATION_FREQUENCY, video_path.c_str());
            ffmpeg_pipe = popen(ffmpeg_cmd, "w");
            if(!ffmpeg_pipe){
                std::cerr << "Failed to open ffmpeg pipe for " << video_path << std::endl;
                record_video = false;
            }
        }

        // =================================================================
        // Data collection (GPU-resident)
        // =================================================================
        float epoch_render_time_ms = 0;
        float epoch_render_gpu_time_ms = 0;
        float epoch_train_forward_time_ms = 0;
        float epoch_train_backward_time_ms = 0;
        float epoch_train_update_time_ms = 0;
        TI epoch_train_forward_calls = 0;
        TI epoch_train_backward_calls = 0;
        TI epoch_train_update_calls = 0;
        {
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = (N_ENVIRONMENTS + BLOCKSIZE - 1) / BLOCKSIZE;
            dim3 grid(N_BLOCKS);
            dim3 block(BLOCKSIZE);
            rlt::devices::cuda::TAG<DEVICE_GPU, true> tag_device{};
            auto& raptor_gru_layer = rlt::nn_models::sequential::layer<1>(raptor_gpu);
            auto& raptor_gru_state_content = rlt::nn_models::sequential::content_state<1>(raptor_state_gpu.content_state);
#ifdef USE_GRU_TEMPORAL
            auto& student_gru_layer = rlt::nn_models::sequential::layer<0>(rollout_student_gpu.head);
            auto& student_gru_state = rlt::nn_models::sequential::content_state<0>(rollout_student_state_gpu.head_state.content_state);
#endif
            T cam_aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
            for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
                imitation_kernels::prologue_kernel<<<grid, block, 0, device_gpu.stream>>>(
                    tag_device, gpu_dynamics_arr, gpu_params_arr, gpu_states_arr,
                    gpu_terminated_arr, gpu_episode_step_arr, gpu_teacher_forcing_arr,
                    gpu_episode_return_arr, gpu_needs_reset,
                    gpu_episode_lengths_log + step_i * N_ENVIRONMENTS,
                    gpu_episode_tf_log + step_i * N_ENVIRONMENTS,
                    TEACHER_FORCING_FRACTION, full_teacher_forcing,
                    rlt::data(gpu_teacher_obs),
                    rlt::data(gpu_all_state_observations) + (TI)(step_i * N_ENVIRONMENTS) * STATE_OBS_DIM,
                    rlt::data(raptor_gru_state_content.state),
                    rlt::data(raptor_gru_layer.initial_hidden_state.parameters),
                    rlt::data(raptor_gru_state_content.step),
#ifdef USE_GRU_TEMPORAL
                    rlt::data(student_gru_state.state),
                    rlt::data(student_gru_layer.initial_hidden_state.parameters),
                    rlt::data(student_gru_state.step),
#endif
#ifdef USE_FRAME_STACKING
                    gpu_episode_start_step,
#endif
                    gpu_brightness_scale_arr,
                    gpu_scene_translation_arr,
                    gpu_scene_yaw_arr,
                    gpu_scene_yaw_cos_arr,
                    gpu_scene_yaw_sin_arr,
                    gpu_indoor_positions, gpu_num_indoor_positions, gpu_env_scene, MAX_INDOOR_POS,
                    rng_gpu, step_i, current_episode_step_limit);
                CUDA_CHECK("prologue_kernel");
#if defined(USE_FRAME_STACKING) && !defined(STACK_TARGET_CHANNEL)
                imitation_kernels::record_episode_start_kernel<<<grid, block, 0, device_gpu.stream>>>(tag_device, gpu_episode_start_step, gpu_episode_start_step_per_row, step_i);
#endif
                auto render_start = std::chrono::high_resolution_clock::now();
                imitation_kernels::make_cameras_kernel<<<grid, block, 0, device_gpu.stream>>>(
                    tag_device, gpu_dynamics_arr, gpu_params_arr, gpu_states_arr,
                    gpu_cameras,
                    env_parameters[0].fov, cam_aspect,
                    env_parameters[0].camera_mount.offset_body[0], env_parameters[0].camera_mount.offset_body[1], env_parameters[0].camera_mount.offset_body[2],
                    env_parameters[0].camera_mount.forward_body[0], env_parameters[0].camera_mount.forward_body[1], env_parameters[0].camera_mount.forward_body[2],
                    env_parameters[0].camera_mount.up_body[0], env_parameters[0].camera_mount.up_body[1], env_parameters[0].camera_mount.up_body[2],
                    gpu_scene_translation_arr, gpu_scene_yaw_cos_arr, gpu_scene_yaw_sin_arr);
                CUDA_CHECK("make_cameras_kernel");
                T* obs_ptr = rlt::data(gpu_all_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
                cudaEventRecord(cameras_ready_event, device_gpu.stream);
                for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                    TI event_i = step_i * N_ACTIVE_SCENES + active_scene_i;
                    auto& renderer = *active_renderers[active_scene_i];
                    constexpr TI n_envs_s = N_ENVIRONMENTS_PER_SCENE;
                    TI base_env = active_scene_i * N_ENVIRONMENTS_PER_SCENE;
                    cudaStream_t optix_stream = active_scene_render_streams[active_scene_i];
                    cudaStreamWaitEvent(optix_stream, cameras_ready_event, 0);
                    cudaMemcpyAsync(
                        active_scene_camera_buffers[active_scene_i],
                        gpu_cameras + base_env,
                        n_envs_s * sizeof(rlt::rendering::raytracing::CameraData<float>),
                        cudaMemcpyDeviceToDevice, optix_stream);
                    int total_scatter = n_envs_s * CAM_PIXELS;
                    int pf_block = 256;
                    int pf_grid = (total_scatter + pf_block - 1) / pf_block;
                    if(step_i % RENDER_TIMING_SAMPLE_PERIOD == 0){
                        cudaEventRecord(render_pass_start_events[event_i], optix_stream);
                    }
                    rlt::render_rgb_only_launch(device, renderer);
                    if(step_i % RENDER_TIMING_SAMPLE_PERIOD == 0){
                        cudaEventRecord(render_pass_stop_events[event_i], optix_stream);
                    }
                    const uint32_t* fb_ptr = active_scene_framebuffer_ptrs[active_scene_i];
                    if constexpr(OBSERVATION_NOISE_STD == 0 && BRIGHTNESS_RANDOMIZATION_RANGE > 0){
                        scatter_pixel_to_float_kernel<true><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, obs_ptr, gpu_brightness_scale_arr, base_env, n_envs_s, CAM_PIXELS, OBSERVATION_DIM);
                    } else {
                        scatter_pixel_to_float_kernel<false><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, obs_ptr, nullptr, base_env, n_envs_s, CAM_PIXELS, OBSERVATION_DIM);
                    }
                    cudaEventRecord(render_scatter_done_events[active_scene_i], optix_stream);
                }
                rlt::evaluate_step(device_gpu, raptor_gpu, gpu_teacher_obs, raptor_state_gpu, gpu_teacher_actions_step, raptor_buffer_gpu, rng_gpu, no_auto_reset_mode);
                CUDA_CHECK("raptor evaluate_step");
                for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                    cudaStreamWaitEvent(device_gpu.stream, render_scatter_done_events[active_scene_i], 0);
                }
                CUDA_CHECK("multi-scene render");
                if constexpr(OBSERVATION_NOISE_STD > 0){
                    constexpr TI NOISE_N = N_ENVIRONMENTS * OBSERVATION_DIM;
                    int noise_block = 256;
                    int noise_grid = (NOISE_N + noise_block - 1) / noise_block;
                    unsigned long long noise_seed = seed + (unsigned long long)epoch_i * STEPS_PER_ENV + step_i;
                    observation_noise_kernel<<<noise_grid, noise_block, 0, device_gpu.stream>>>(obs_ptr, NOISE_N, OBSERVATION_NOISE_STD, noise_seed);
                    CUDA_CHECK("observation_noise_kernel");
                }
                if constexpr(OBSERVATION_NOISE_STD > 0 && BRIGHTNESS_RANDOMIZATION_RANGE > 0){
                    int br_block = 256;
                    dim3 br_grid((OBSERVATION_DIM + br_block - 1) / br_block, N_ENVIRONMENTS);
                    brightness_apply_kernel<<<br_grid, br_block, 0, device_gpu.stream>>>(obs_ptr, gpu_brightness_scale_arr, N_ENVIRONMENTS, OBSERVATION_DIM);
                    CUDA_CHECK("brightness_apply_kernel");
                }
                // Target frame rendering (second pass)
                imitation_kernels::make_target_cameras_kernel<<<grid, block, 0, device_gpu.stream>>>(
                    tag_device, gpu_target_cameras,
                    env_parameters[0].fov, cam_aspect,
                    env_parameters[0].camera_mount.offset_body[0], env_parameters[0].camera_mount.offset_body[1], env_parameters[0].camera_mount.offset_body[2],
                    env_parameters[0].camera_mount.forward_body[0], env_parameters[0].camera_mount.forward_body[1], env_parameters[0].camera_mount.forward_body[2],
                    env_parameters[0].camera_mount.up_body[0], env_parameters[0].camera_mount.up_body[1], env_parameters[0].camera_mount.up_body[2],
                    gpu_scene_translation_arr, gpu_scene_yaw_cos_arr, gpu_scene_yaw_sin_arr);
                CUDA_CHECK("make_target_cameras_kernel");
                T* target_obs_ptr = rlt::data(gpu_all_target_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
                cudaEventRecord(target_cameras_ready_event, device_gpu.stream);
                for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                    TI event_i = step_i * N_ACTIVE_SCENES + active_scene_i;
                    auto& renderer = *active_renderers[active_scene_i];
                    constexpr TI n_envs_s = N_ENVIRONMENTS_PER_SCENE;
                    TI base_env = active_scene_i * N_ENVIRONMENTS_PER_SCENE;
                    cudaStream_t optix_stream = active_scene_render_streams[active_scene_i];
                    cudaStreamWaitEvent(optix_stream, target_cameras_ready_event, 0);
                    cudaMemcpyAsync(
                        active_scene_camera_buffers[active_scene_i],
                        gpu_target_cameras + base_env,
                        n_envs_s * sizeof(rlt::rendering::raytracing::CameraData<float>),
                        cudaMemcpyDeviceToDevice, optix_stream);
                    int total_scatter = n_envs_s * CAM_PIXELS;
                    int pf_block = 256;
                    int pf_grid = (total_scatter + pf_block - 1) / pf_block;
                    if(step_i % RENDER_TIMING_SAMPLE_PERIOD == 0){
                        cudaEventRecord(target_render_pass_start_events[event_i], optix_stream);
                    }
                    rlt::render_rgb_only_launch(device, renderer);
                    if(step_i % RENDER_TIMING_SAMPLE_PERIOD == 0){
                        cudaEventRecord(target_render_pass_stop_events[event_i], optix_stream);
                    }
                    const uint32_t* fb_ptr = active_scene_framebuffer_ptrs[active_scene_i];
                    if constexpr(BRIGHTNESS_RANDOMIZATION_RANGE > 0){
                        scatter_pixel_to_float_kernel<true><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, target_obs_ptr, gpu_brightness_scale_arr, base_env, n_envs_s, CAM_PIXELS, OBSERVATION_DIM);
                    } else {
                        scatter_pixel_to_float_kernel<false><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, target_obs_ptr, nullptr, base_env, n_envs_s, CAM_PIXELS, OBSERVATION_DIM);
                    }
                    cudaEventRecord(target_render_scatter_done_events[active_scene_i], optix_stream);
                }
#if defined(USE_FRAME_STACKING) && defined(STACK_TARGET_CHANNEL)
                {
                    TI history_slot = step_i % FRAME_STACK_HISTORY_LENGTH;
                    T* history_slot_ptr = rlt::data(gpu_frame_stack_history) + (TI)(history_slot * N_ENVIRONMENTS) * OBSERVATION_DIM;
                    cudaMemcpyAsync(history_slot_ptr, obs_ptr, N_ENVIRONMENTS * OBSERVATION_DIM * sizeof(T), cudaMemcpyDeviceToDevice, device_gpu.stream);
                }
#endif
                for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                    cudaStreamWaitEvent(device_gpu.stream, target_render_scatter_done_events[active_scene_i], 0);
                }
#if defined(USE_FRAME_STACKING) && defined(STACK_TARGET_CHANNEL)
                {
                    int total_elements = N_ENVIRONMENTS * COMBINED_OBS_DIM;
                    build_frame_stacked_with_target_from_history_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                        rlt::data(gpu_frame_stack_history),
                        target_obs_ptr,
                        gpu_episode_start_step,
                        step_i,
                        rlt::data(gpu_all_combined_observations) + (TI)(step_i * N_ENVIRONMENTS) * COMBINED_OBS_DIM,
                        OBSERVATION_DIM, IMG_C, FRAME_STACK_N, FRAME_STACK_STRIDE, COMBINED_IMG_C, COMBINED_OBS_DIM, N_ENVIRONMENTS
                    );
                }
#endif
                CUDA_CHECK("target render");
                auto render_end = std::chrono::high_resolution_clock::now();
                epoch_render_time_ms += std::chrono::duration<float, std::milli>(render_end - render_start).count();
                if(record_video && ffmpeg_pipe){
                    static constexpr TI VIDEO_OBS_SIZE = N_ENVIRONMENTS * OBSERVATION_DIM;
                    std::vector<float> cpu_obs(VIDEO_OBS_SIZE);
                    std::vector<float> cpu_target_obs(VIDEO_OBS_SIZE);
                    cudaStreamSynchronize(device_gpu.stream);
                    cudaMemcpy(cpu_obs.data(), obs_ptr, VIDEO_OBS_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(cpu_target_obs.data(), target_obs_ptr, VIDEO_OBS_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
                    for(TI scene_row = 0; scene_row < SCENE_GRID_ROWS; scene_row++){
                        for(TI scene_col = 0; scene_col < SCENE_GRID_COLS; scene_col++){
                            TI active_scene_i = scene_row * SCENE_GRID_COLS + scene_col;
                            if(active_scene_i >= N_ACTIVE_SCENES){
                                continue;
                            }
                            for(TI local_row = 0; local_row < ENV_GRID_SIDE; local_row++){
                                for(TI local_col = 0; local_col < ENV_GRID_SIDE; local_col++){
                                    TI local_env = local_row * ENV_GRID_SIDE + local_col;
                                    TI env_i = active_scene_i * N_ENVIRONMENTS_PER_SCENE + local_env;
                                    const float* env_obs = cpu_obs.data() + env_i * OBSERVATION_DIM;
                                    const float* env_target_obs = cpu_target_obs.data() + env_i * OBSERVATION_DIM;
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
                    fwrite(mosaic_frame.data(), 1, mosaic_frame.size(), ffmpeg_pipe);
                }
                {
                    auto step_state_obs = rlt::view_range(device_gpu, gpu_all_state_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                    auto step_state_obs_reshaped = rlt::reshape_row_major(device_gpu, step_state_obs, rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, STATE_OBS_DIM>{});
#ifdef USE_FRAME_STACKING
#ifndef STACK_TARGET_CHANNEL
                    {
                        constexpr TI GI_BLOCK = 256;
                        constexpr TI GI_GRID = (N_ENVIRONMENTS + GI_BLOCK - 1) / GI_BLOCK;
                        imitation_kernels::compute_gather_indices_rollout_kernel<<<GI_GRID, GI_BLOCK, 0, device_gpu.stream>>>(
                            gpu_rollout_gather_indices, gpu_episode_start_step, step_i, N_ENVIRONMENTS);
                        imitation_kernels::compute_gather_indices_rollout_kernel<<<GI_GRID, GI_BLOCK, 0, device_gpu.stream>>>(
                            gpu_rollout_target_gather_indices, gpu_episode_start_step, step_i, N_ENVIRONMENTS);
                    }
                    {
                        int total_elements = N_ENVIRONMENTS * STACKED_OBS_DIM;
                        gather_frames_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                            rlt::data(gpu_all_observations), gpu_rollout_gather_indices, rlt::data(gpu_rollout_stacked),
                            OBSERVATION_DIM, IMG_C, FRAME_STACK_N, STACKED_IMG_C, STACKED_OBS_DIM, N_ENVIRONMENTS);
                        gather_frames_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                            rlt::data(gpu_all_target_observations), gpu_rollout_target_gather_indices, rlt::data(gpu_rollout_stacked_target),
                            OBSERVATION_DIM, IMG_C, FRAME_STACK_N, STACKED_IMG_C, STACKED_OBS_DIM, N_ENVIRONMENTS);
                    }
                    using ROLLOUT_IMG_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, IMG_H, IMG_W, STACKED_IMG_C>;
                    auto step_target_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_rollout_stacked_target, ROLLOUT_IMG_SHAPE{});
                    auto step_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_rollout_stacked, ROLLOUT_IMG_SHAPE{});
                    auto inputs = rlt::nn_models::parallel::pack_inputs(step_target_obs_reshaped, step_obs_reshaped, step_state_obs_reshaped);
#endif
#ifdef STACK_TARGET_CHANNEL
                    auto step_combined = rlt::view_range(device_gpu, gpu_all_combined_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                    using ROLLOUT_IMG_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, IMG_H, IMG_W, COMBINED_IMG_C>;
                    auto step_combined_reshaped = rlt::reshape_row_major(device_gpu, step_combined, ROLLOUT_IMG_SHAPE{});
                    auto inputs = rlt::nn_models::parallel::pack_inputs(step_combined_reshaped, step_state_obs_reshaped);
#endif
#else
#ifdef STACK_TARGET_CHANNEL
                    {
                        int total_elements = N_ENVIRONMENTS * COMBINED_OBS_DIM;
                        concat_target_channels_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                            rlt::data(gpu_all_observations) + step_i * N_ENVIRONMENTS * OBSERVATION_DIM,
                            rlt::data(gpu_all_target_observations) + step_i * N_ENVIRONMENTS * OBSERVATION_DIM,
                            rlt::data(gpu_rollout_combined),
                            OBSERVATION_DIM, IMG_C, COMBINED_IMG_C, COMBINED_OBS_DIM, N_ENVIRONMENTS);
                    }
                    auto rollout_combined_view = rlt::view_range(device_gpu, gpu_rollout_combined, (TI)0, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                    using ROLLOUT_IMG_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, IMG_H, IMG_W, COMBINED_IMG_C>;
                    auto step_combined_reshaped = rlt::reshape_row_major(device_gpu, rollout_combined_view, ROLLOUT_IMG_SHAPE{});
                    auto inputs = rlt::nn_models::parallel::pack_inputs(step_combined_reshaped, step_state_obs_reshaped);
#else
                    auto step_target_obs = rlt::view_range(device_gpu, gpu_all_target_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                    auto step_obs = rlt::view_range(device_gpu, gpu_all_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                    using ROLLOUT_IMG_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, IMG_H, IMG_W, STACKED_IMG_C>;
                    auto step_target_obs_reshaped = rlt::reshape_row_major(device_gpu, step_target_obs, ROLLOUT_IMG_SHAPE{});
                    auto step_obs_reshaped = rlt::reshape_row_major(device_gpu, step_obs, ROLLOUT_IMG_SHAPE{});
                    auto inputs = rlt::nn_models::parallel::pack_inputs(step_target_obs_reshaped, step_obs_reshaped, step_state_obs_reshaped);
#endif
#endif
#ifdef USE_GRU_TEMPORAL
                    rlt::evaluate_step(device_gpu, rollout_student_gpu, inputs, rollout_student_state_gpu, gpu_student_actions_step, rollout_student_buffers, rng_gpu, no_auto_reset_mode);
#else
                    rlt::evaluate(device_gpu, rollout_student_gpu, inputs, gpu_student_actions_step, rollout_student_buffers, rng_gpu);
#endif
                }
                {
                    imitation_kernels::epilogue_kernel<<<grid, block, 0, device_gpu.stream>>>(
                        tag_device, gpu_dynamics_arr, gpu_params_arr, gpu_states_arr,
                        gpu_terminated_arr, gpu_episode_step_arr,
                        gpu_episode_return_arr,
                        rlt::data(gpu_teacher_actions_step),
                        rlt::data(gpu_student_actions_step),
                        rlt::data(gpu_all_targets),
                        rng_gpu, step_i);
                }
                CUDA_CHECK("epilogue_kernel");
                global_step += N_ENVIRONMENTS;
            }
        }
        if(ffmpeg_pipe){ pclose(ffmpeg_pipe); ffmpeg_pipe = nullptr; }

        // =================================================================
        // Training (GPU)
        // =================================================================
        T epoch_loss = 0;
        T epoch_loss_sum = 0;
        TI epoch_loss_count = 0;

#ifdef USE_GRU_TEMPORAL
        for(TI pass = 0; pass < N_TRAIN_PASSES; pass++){
            TI window_order[N_WINDOWS];
            for(TI i = 0; i < N_WINDOWS; i++) window_order[i] = i;
            for(TI i = N_WINDOWS - 1; i > 0; i--){
                TI j = rlt::random::uniform_int_distribution(device.random, (TI)0, i, rng);
                std::swap(window_order[i], window_order[j]);
            }
            for(TI wi = 0; wi < N_WINDOWS; wi++){
                TI window_i = window_order[wi];
                TI window_offset = window_i * WINDOW_SAMPLES;

                rlt::zero_gradient(device_gpu, student_gpu);

#ifdef STACK_TARGET_CHANNEL
                {
                    int total_elements = WINDOW_SAMPLES * COMBINED_OBS_DIM;
                    concat_target_channels_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                        rlt::data(gpu_all_observations) + window_offset * OBSERVATION_DIM,
                        rlt::data(gpu_all_target_observations) + window_offset * OBSERVATION_DIM,
                        rlt::data(gpu_rollout_combined),
                        OBSERVATION_DIM, IMG_C, COMBINED_IMG_C, COMBINED_OBS_DIM, WINDOW_SAMPLES);
                }
                using GRU_IMG_SHAPE = rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, IMG_H, IMG_W, COMBINED_IMG_C>;
                auto win_combined_reshaped = rlt::reshape_row_major(device_gpu, gpu_rollout_combined, GRU_IMG_SHAPE{});
                auto win_state = rlt::view_range(device_gpu, gpu_all_state_observations, window_offset, rlt::tensor::ViewSpec<0, WINDOW_SAMPLES>{});
                using GRU_STATE_SHAPE = rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, STATE_OBS_DIM>;
                auto win_state_reshaped = rlt::reshape_row_major(device_gpu, win_state, GRU_STATE_SHAPE{});
                TI train_forward_call_i = epoch_train_forward_calls;
                cudaEventRecord(train_forward_start_events[train_forward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(win_combined_reshaped, win_state_reshaped); rlt::forward(device_gpu, student_gpu, inputs, gpu_student_output_train, student_buffers, rng_gpu); }
                cudaEventRecord(train_forward_stop_events[train_forward_call_i], device_gpu.stream);
                epoch_train_forward_calls++;
#else
                auto win_target_obs = rlt::view_range(device_gpu, gpu_all_target_observations, window_offset, rlt::tensor::ViewSpec<0, WINDOW_SAMPLES>{});
                using GRU_IMG_SHAPE = rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, IMG_H, IMG_W, IMG_C>;
                auto win_target_obs_reshaped = rlt::reshape_row_major(device_gpu, win_target_obs, GRU_IMG_SHAPE{});
                auto win_obs = rlt::view_range(device_gpu, gpu_all_observations, window_offset, rlt::tensor::ViewSpec<0, WINDOW_SAMPLES>{});
                auto win_obs_reshaped = rlt::reshape_row_major(device_gpu, win_obs, GRU_IMG_SHAPE{});
                auto win_state = rlt::view_range(device_gpu, gpu_all_state_observations, window_offset, rlt::tensor::ViewSpec<0, WINDOW_SAMPLES>{});
                using GRU_STATE_SHAPE = rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, STATE_OBS_DIM>;
                auto win_state_reshaped = rlt::reshape_row_major(device_gpu, win_state, GRU_STATE_SHAPE{});
                TI train_forward_call_i = epoch_train_forward_calls;
                cudaEventRecord(train_forward_start_events[train_forward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(win_target_obs_reshaped, win_obs_reshaped, win_state_reshaped); rlt::forward(device_gpu, student_gpu, inputs, gpu_student_output_train, student_buffers, rng_gpu); }
                cudaEventRecord(train_forward_stop_events[train_forward_call_i], device_gpu.stream);
                epoch_train_forward_calls++;
#endif

                auto student_output_matrix = rlt::matrix_view(device_gpu, gpu_student_output_train);
                auto target_tensor = rlt::view_range(device_gpu, gpu_all_targets, window_offset, rlt::tensor::ViewSpec<0, WINDOW_SAMPLES>{});
                auto target_matrix = rlt::matrix_view(device_gpu, target_tensor);
                rlt::nn::loss_functions::mse::gradient(device_gpu, student_output_matrix, target_matrix, gpu_d_action_train, (T)0.5);

                if(pass == 0){
                    imitation_kernels::mse_batch_loss_kernel<<<1, 1, 0, device_gpu.stream>>>(
                        rlt::data(gpu_student_output_train),
                        rlt::data(gpu_all_targets) + window_offset * TARGET_DIM,
                        gpu_logged_batch_losses + epoch_loss_count,
                        WINDOW_SAMPLES * TARGET_DIM);
                    epoch_loss_count++;
                }

                auto gpu_d_action_tensor = rlt::to_tensor(device_gpu, gpu_d_action_train);
                using GRU_ACTION_SHAPE = rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, TARGET_DIM>;
                auto gpu_d_action_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_action_tensor, GRU_ACTION_SHAPE{});
#ifdef STACK_TARGET_CHANNEL
                TI train_backward_call_i = epoch_train_backward_calls;
                cudaEventRecord(train_backward_start_events[train_backward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(win_combined_reshaped, win_state_reshaped); rlt::backward(device_gpu, student_gpu, inputs, gpu_d_action_reshaped, student_buffers); }
                cudaEventRecord(train_backward_stop_events[train_backward_call_i], device_gpu.stream);
                epoch_train_backward_calls++;
#else
                TI train_backward_call_i = epoch_train_backward_calls;
                cudaEventRecord(train_backward_start_events[train_backward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(win_target_obs_reshaped, win_obs_reshaped, win_state_reshaped); rlt::backward(device_gpu, student_gpu, inputs, gpu_d_action_reshaped, student_buffers); }
                cudaEventRecord(train_backward_stop_events[train_backward_call_i], device_gpu.stream);
                epoch_train_backward_calls++;
#endif
                TI train_update_call_i = epoch_train_update_calls;
                cudaEventRecord(train_update_start_events[train_update_call_i], device_gpu.stream);
                rlt::step(device_gpu, optimizer_gpu, student_gpu);
                cudaEventRecord(train_update_stop_events[train_update_call_i], device_gpu.stream);
                epoch_train_update_calls++;
            }
        }
        if(epoch_train_update_calls > 0){
            cudaEventSynchronize(train_update_stop_events[epoch_train_update_calls - 1]);
        } else if(epoch_train_backward_calls > 0){
            cudaEventSynchronize(train_backward_stop_events[epoch_train_backward_calls - 1]);
        } else if(epoch_train_forward_calls > 0){
            cudaEventSynchronize(train_forward_stop_events[epoch_train_forward_calls - 1]);
        }
        for(TI call_i = 0; call_i < epoch_train_forward_calls; call_i++){
            float train_forward_time_ms = 0;
            cudaEventElapsedTime(&train_forward_time_ms, train_forward_start_events[call_i], train_forward_stop_events[call_i]);
            epoch_train_forward_time_ms += train_forward_time_ms;
        }
        for(TI call_i = 0; call_i < epoch_train_backward_calls; call_i++){
            float train_backward_time_ms = 0;
            cudaEventElapsedTime(&train_backward_time_ms, train_backward_start_events[call_i], train_backward_stop_events[call_i]);
            epoch_train_backward_time_ms += train_backward_time_ms;
        }
        for(TI call_i = 0; call_i < epoch_train_update_calls; call_i++){
            float train_update_time_ms = 0;
            cudaEventElapsedTime(&train_update_time_ms, train_update_start_events[call_i], train_update_stop_events[call_i]);
            epoch_train_update_time_ms += train_update_time_ms;
        }
        if(epoch_loss_count > 0){
            cudaMemcpy(cpu_logged_batch_losses.data(), gpu_logged_batch_losses, epoch_loss_count * sizeof(T), cudaMemcpyDeviceToHost);
            for(TI loss_i = 0; loss_i < epoch_loss_count; loss_i++){
                epoch_loss_sum += cpu_logged_batch_losses[loss_i];
            }
        }
        epoch_loss = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : (T)0;
        rlt::copy(device_gpu, device_gpu, student_gpu, rollout_student_gpu);
        rlt::reset(device_gpu, rollout_student_gpu, rollout_student_state_gpu, rng_gpu);
#else
        for(TI pass = 0; pass < N_TRAIN_PASSES; pass++){
            // Shuffle batch order
            TI batch_order[N_BATCHES];
            for(TI i = 0; i < N_BATCHES; i++) batch_order[i] = i;
            for(TI i = N_BATCHES - 1; i > 0; i--){
                TI j = rlt::random::uniform_int_distribution(device.random, (TI)0, i, rng);
                std::swap(batch_order[i], batch_order[j]);
            }

            for(TI batch_idx = 0; batch_idx < N_BATCHES; batch_idx++){
                TI batch_i = batch_order[batch_idx];
                TI batch_offset = batch_i * BATCH_SIZE;

                rlt::zero_gradient(device_gpu, student_gpu);

                // Student forward on GPU
#ifdef USE_FRAME_STACKING
#ifndef STACK_TARGET_CHANNEL
                {
                    constexpr TI GI_BLOCK = 256;
                    constexpr TI GI_GRID = (BATCH_SIZE + GI_BLOCK - 1) / GI_BLOCK;
                    imitation_kernels::compute_gather_indices_kernel<<<GI_GRID, GI_BLOCK, 0, device_gpu.stream>>>(
                        gpu_gather_indices, gpu_episode_start_step_per_row, batch_offset, BATCH_SIZE);
                    imitation_kernels::compute_gather_indices_kernel<<<GI_GRID, GI_BLOCK, 0, device_gpu.stream>>>(
                        gpu_target_gather_indices, gpu_episode_start_step_per_row, batch_offset, BATCH_SIZE);
                }
                {
                    int total_elements = BATCH_SIZE * STACKED_OBS_DIM;
                    gather_frames_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                        rlt::data(gpu_all_observations), gpu_gather_indices, rlt::data(gpu_stacked_batch),
                        OBSERVATION_DIM, IMG_C, FRAME_STACK_N, STACKED_IMG_C, STACKED_OBS_DIM, BATCH_SIZE);
                    gather_frames_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                        rlt::data(gpu_all_target_observations), gpu_target_gather_indices, rlt::data(gpu_stacked_target_batch),
                        OBSERVATION_DIM, IMG_C, FRAME_STACK_N, STACKED_IMG_C, STACKED_OBS_DIM, BATCH_SIZE);
                }
                using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, STACKED_IMG_C>;
                auto gpu_target_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_stacked_target_batch, ACTOR_INPUT_SHAPE{});
                auto gpu_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_stacked_batch, ACTOR_INPUT_SHAPE{});
                auto gpu_state_obs_batch = rlt::view_range(device_gpu, gpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_state_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_state_obs_batch, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});
                TI train_forward_call_i = epoch_train_forward_calls;
                cudaEventRecord(train_forward_start_events[train_forward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_target_obs_batch_reshaped, gpu_obs_batch_reshaped, gpu_state_obs_batch_reshaped); rlt::forward(device_gpu, student_gpu, inputs, gpu_student_output_train, student_buffers, rng_gpu); }
                cudaEventRecord(train_forward_stop_events[train_forward_call_i], device_gpu.stream);
                epoch_train_forward_calls++;
#endif
#ifdef STACK_TARGET_CHANNEL
                using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
                auto gpu_combined_batch = rlt::view_range(device_gpu, gpu_all_combined_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_combined_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_combined_batch, ACTOR_INPUT_SHAPE{});
                auto gpu_state_obs_batch = rlt::view_range(device_gpu, gpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_state_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_state_obs_batch, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});
                TI train_forward_call_i = epoch_train_forward_calls;
                cudaEventRecord(train_forward_start_events[train_forward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_combined_batch_reshaped, gpu_state_obs_batch_reshaped); rlt::forward(device_gpu, student_gpu, inputs, gpu_student_output_train, student_buffers, rng_gpu); }
                cudaEventRecord(train_forward_stop_events[train_forward_call_i], device_gpu.stream);
                epoch_train_forward_calls++;
#endif
#else
#ifdef STACK_TARGET_CHANNEL
                rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, BATCH_SIZE, COMBINED_OBS_DIM>>> gpu_combined_batch_train;
                rlt::malloc(device_gpu, gpu_combined_batch_train);
                {
                    int total_elements = BATCH_SIZE * COMBINED_OBS_DIM;
                    concat_target_channels_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                        rlt::data(gpu_all_observations) + batch_offset * OBSERVATION_DIM,
                        rlt::data(gpu_all_target_observations) + batch_offset * OBSERVATION_DIM,
                        rlt::data(gpu_combined_batch_train),
                        OBSERVATION_DIM, IMG_C, COMBINED_IMG_C, COMBINED_OBS_DIM, BATCH_SIZE);
                }
                using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
                auto gpu_combined_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_combined_batch_train, ACTOR_INPUT_SHAPE{});
                auto gpu_state_obs_batch = rlt::view_range(device_gpu, gpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_state_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_state_obs_batch, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});
                TI train_forward_call_i = epoch_train_forward_calls;
                cudaEventRecord(train_forward_start_events[train_forward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_combined_batch_reshaped, gpu_state_obs_batch_reshaped); rlt::forward(device_gpu, student_gpu, inputs, gpu_student_output_train, student_buffers, rng_gpu); }
                cudaEventRecord(train_forward_stop_events[train_forward_call_i], device_gpu.stream);
                epoch_train_forward_calls++;
#else
                auto gpu_target_obs_batch = rlt::view_range(device_gpu, gpu_all_target_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, STACKED_IMG_C>;
                auto gpu_target_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_target_obs_batch, ACTOR_INPUT_SHAPE{});
                auto gpu_obs_batch = rlt::view_range(device_gpu, gpu_all_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_obs_batch, ACTOR_INPUT_SHAPE{});
                auto gpu_state_obs_batch = rlt::view_range(device_gpu, gpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_state_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_state_obs_batch, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});
                TI train_forward_call_i = epoch_train_forward_calls;
                cudaEventRecord(train_forward_start_events[train_forward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_target_obs_batch_reshaped, gpu_obs_batch_reshaped, gpu_state_obs_batch_reshaped); rlt::forward(device_gpu, student_gpu, inputs, gpu_student_output_train, student_buffers, rng_gpu); }
                cudaEventRecord(train_forward_stop_events[train_forward_call_i], device_gpu.stream);
                epoch_train_forward_calls++;
#endif
#endif

                // MSE loss gradient
                auto student_output_matrix = rlt::matrix_view(device_gpu, gpu_student_output_train);
                auto target_batch_tensor = rlt::view_range(device_gpu, gpu_all_targets, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto target_batch = rlt::matrix_view(device_gpu, target_batch_tensor);
                rlt::nn::loss_functions::mse::gradient(device_gpu, student_output_matrix, target_batch, gpu_d_action_train, (T)0.5);

                // Compute loss for logging
                if(pass == 0){
                    imitation_kernels::mse_batch_loss_kernel<<<1, 1, 0, device_gpu.stream>>>(
                        rlt::data(gpu_student_output_train),
                        rlt::data(gpu_all_targets) + batch_offset * TARGET_DIM,
                        gpu_logged_batch_losses + epoch_loss_count,
                        BATCH_SIZE * TARGET_DIM);
                    epoch_loss_count++;
                }

                // Student backward + Adam step
                auto gpu_d_action_tensor = rlt::to_tensor(device_gpu, gpu_d_action_train);
                auto gpu_d_action_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, TARGET_DIM>{});
#ifdef USE_FRAME_STACKING
#ifdef STACK_TARGET_CHANNEL
                TI train_backward_call_i = epoch_train_backward_calls;
                cudaEventRecord(train_backward_start_events[train_backward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_combined_batch_reshaped, gpu_state_obs_batch_reshaped); rlt::backward(device_gpu, student_gpu, inputs, gpu_d_action_reshaped, student_buffers); }
                cudaEventRecord(train_backward_stop_events[train_backward_call_i], device_gpu.stream);
                epoch_train_backward_calls++;
#else
                TI train_backward_call_i = epoch_train_backward_calls;
                cudaEventRecord(train_backward_start_events[train_backward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_target_obs_batch_reshaped, gpu_obs_batch_reshaped, gpu_state_obs_batch_reshaped); rlt::backward(device_gpu, student_gpu, inputs, gpu_d_action_reshaped, student_buffers); }
                cudaEventRecord(train_backward_stop_events[train_backward_call_i], device_gpu.stream);
                epoch_train_backward_calls++;
#endif
#else
#ifdef STACK_TARGET_CHANNEL
                TI train_backward_call_i = epoch_train_backward_calls;
                cudaEventRecord(train_backward_start_events[train_backward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_combined_batch_reshaped, gpu_state_obs_batch_reshaped); rlt::backward(device_gpu, student_gpu, inputs, gpu_d_action_reshaped, student_buffers); }
                cudaEventRecord(train_backward_stop_events[train_backward_call_i], device_gpu.stream);
                epoch_train_backward_calls++;
#else
                TI train_backward_call_i = epoch_train_backward_calls;
                cudaEventRecord(train_backward_start_events[train_backward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_target_obs_batch_reshaped, gpu_obs_batch_reshaped, gpu_state_obs_batch_reshaped); rlt::backward(device_gpu, student_gpu, inputs, gpu_d_action_reshaped, student_buffers); }
                cudaEventRecord(train_backward_stop_events[train_backward_call_i], device_gpu.stream);
                epoch_train_backward_calls++;
#endif
#endif
                TI train_update_call_i = epoch_train_update_calls;
                cudaEventRecord(train_update_start_events[train_update_call_i], device_gpu.stream);
                rlt::step(device_gpu, optimizer_gpu, student_gpu);
                cudaEventRecord(train_update_stop_events[train_update_call_i], device_gpu.stream);
                epoch_train_update_calls++;
#if !defined(USE_FRAME_STACKING) && defined(STACK_TARGET_CHANNEL)
                rlt::free(device_gpu, gpu_combined_batch_train);
#endif
            }
        }
        if(epoch_train_update_calls > 0){
            cudaEventSynchronize(train_update_stop_events[epoch_train_update_calls - 1]);
        } else if(epoch_train_backward_calls > 0){
            cudaEventSynchronize(train_backward_stop_events[epoch_train_backward_calls - 1]);
        } else if(epoch_train_forward_calls > 0){
            cudaEventSynchronize(train_forward_stop_events[epoch_train_forward_calls - 1]);
        }
        for(TI call_i = 0; call_i < epoch_train_forward_calls; call_i++){
            float train_forward_time_ms = 0;
            cudaEventElapsedTime(&train_forward_time_ms, train_forward_start_events[call_i], train_forward_stop_events[call_i]);
            epoch_train_forward_time_ms += train_forward_time_ms;
        }
        for(TI call_i = 0; call_i < epoch_train_backward_calls; call_i++){
            float train_backward_time_ms = 0;
            cudaEventElapsedTime(&train_backward_time_ms, train_backward_start_events[call_i], train_backward_stop_events[call_i]);
            epoch_train_backward_time_ms += train_backward_time_ms;
        }
        for(TI call_i = 0; call_i < epoch_train_update_calls; call_i++){
            float train_update_time_ms = 0;
            cudaEventElapsedTime(&train_update_time_ms, train_update_start_events[call_i], train_update_stop_events[call_i]);
            epoch_train_update_time_ms += train_update_time_ms;
        }
        if(epoch_loss_count > 0){
            cudaMemcpy(cpu_logged_batch_losses.data(), gpu_logged_batch_losses, epoch_loss_count * sizeof(T), cudaMemcpyDeviceToHost);
            for(TI loss_i = 0; loss_i < epoch_loss_count; loss_i++){
                epoch_loss_sum += cpu_logged_batch_losses[loss_i];
            }
        }
        epoch_loss = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : (T)0;
        rlt::copy(device_gpu, device_gpu, student_gpu, rollout_student_gpu);
#ifdef USE_GRU_TEMPORAL
        rlt::reset(device_gpu, rollout_student_gpu, rollout_student_state_gpu, rng_gpu);
#endif
#endif
        if(epoch_train_forward_calls == 0 && epoch_train_backward_calls == 0 && epoch_train_update_calls == 0){
            cudaStreamSynchronize(device_gpu.stream);
        }
        TI render_timing_samples = 0;
        for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i += RENDER_TIMING_SAMPLE_PERIOD){
            float render_pass_time_ms = 0;
            float target_render_pass_time_ms = 0;
            for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                TI event_i = step_i * N_ACTIVE_SCENES + active_scene_i;
                float scene_render_time_ms = 0;
                cudaEventElapsedTime(&scene_render_time_ms, render_pass_start_events[event_i], render_pass_stop_events[event_i]);
                render_pass_time_ms = std::max(render_pass_time_ms, scene_render_time_ms);
                float target_scene_render_time_ms = 0;
                cudaEventElapsedTime(&target_scene_render_time_ms, target_render_pass_start_events[event_i], target_render_pass_stop_events[event_i]);
                target_render_pass_time_ms = std::max(target_render_pass_time_ms, target_scene_render_time_ms);
            }
            epoch_render_gpu_time_ms += render_pass_time_ms + target_render_pass_time_ms;
            render_timing_samples++;
        }
        if(render_timing_samples > 0){
            epoch_render_gpu_time_ms *= static_cast<float>(STEPS_PER_ENV) / static_cast<float>(render_timing_samples);
        }
        imitation_kernels::reduce_episode_stats_kernel<<<1, 1, 0, device_gpu.stream>>>(
            gpu_episode_lengths_log,
            gpu_episode_tf_log,
            gpu_episode_step_arr,
            gpu_teacher_forcing_arr,
            gpu_epoch_episode_stats
        );
        cudaMemcpy(cpu_epoch_episode_stats.data(), gpu_epoch_episode_stats, cpu_epoch_episode_stats.size() * sizeof(T), cudaMemcpyDeviceToHost);
        episode_length_sum_tf = cpu_epoch_episode_stats[0];
        episode_count_tf = static_cast<TI>(cpu_epoch_episode_stats[1]);
        episode_length_sum_student = cpu_epoch_episode_stats[2];
        episode_count_student = static_cast<TI>(cpu_epoch_episode_stats[3]);

        // Logging
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> training_elapsed = now - training_start;
        std::chrono::duration<T> epoch_elapsed = now - epoch_start;
        T mean_episode_length_tf = episode_count_tf > 0 ? episode_length_sum_tf / episode_count_tf : 0;
        T mean_episode_length_student = episode_count_student > 0 ? episode_length_sum_student / episode_count_student : 0;
        TI episode_count = episode_count_tf + episode_count_student;
        T mean_episode_length = episode_count > 0 ? (episode_length_sum_tf + episode_length_sum_student) / episode_count : 0;
        T fps = epoch_elapsed.count() > 0 ? static_cast<T>(STEPS_TOTAL) / epoch_elapsed.count() : 0;
        T render_time_s = static_cast<T>(epoch_render_time_ms) / static_cast<T>(1000);
        T render_gpu_time_s = static_cast<T>(epoch_render_gpu_time_ms) / static_cast<T>(1000);
        T render_fps = render_time_s > 0 ? static_cast<T>(STEPS_PER_ENV * N_ENVIRONMENTS) / render_time_s : 0;
        T render_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * render_time_s / epoch_elapsed.count() : 0;
        T render_gpu_fps = render_gpu_time_s > 0 ? static_cast<T>(STEPS_PER_ENV * N_ENVIRONMENTS) / render_gpu_time_s : 0;
        T render_gpu_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * render_gpu_time_s / epoch_elapsed.count() : 0;
        T train_forward_time_s = static_cast<T>(epoch_train_forward_time_ms) / static_cast<T>(1000);
        T train_backward_time_s = static_cast<T>(epoch_train_backward_time_ms) / static_cast<T>(1000);
        T train_update_time_s = static_cast<T>(epoch_train_update_time_ms) / static_cast<T>(1000);
        T train_forward_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * train_forward_time_s / epoch_elapsed.count() : 0;
        T train_backward_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * train_backward_time_s / epoch_elapsed.count() : 0;
        T train_update_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * train_update_time_s / epoch_elapsed.count() : 0;
        T train_forward_avg_ms = epoch_train_forward_calls > 0 ? static_cast<T>(epoch_train_forward_time_ms) / static_cast<T>(epoch_train_forward_calls) : 0;
        T train_backward_avg_ms = epoch_train_backward_calls > 0 ? static_cast<T>(epoch_train_backward_time_ms) / static_cast<T>(epoch_train_backward_calls) : 0;
        T train_update_avg_ms = epoch_train_update_calls > 0 ? static_cast<T>(epoch_train_update_time_ms) / static_cast<T>(epoch_train_update_calls) : 0;

        std::cout << (full_teacher_forcing ? "[TF] " : "[TF=" + std::to_string((int)(TEACHER_FORCING_FRACTION * 100)) + "%] ")
                  << "Epoch: " << std::setw(5) << epoch_i
                  << " MSE: " << std::setw(10) << std::setprecision(6) << std::fixed << epoch_loss
                  << " mean_ep_len: " << std::setw(6) << std::setprecision(1) << mean_episode_length
                  << " ep_limit: " << std::setw(3) << current_episode_step_limit
                  << " episodes: " << std::setw(5) << episode_count
                  << " fps: " << std::setw(7) << std::setprecision(0) << fps
                  << " render: " << std::setw(5) << std::setprecision(1) << render_time_s << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << render_pct << "%"
                  << " " << std::setw(7) << std::setprecision(0) << render_fps << " fps)"
                  << " render_gpu: " << std::setw(5) << std::setprecision(1) << render_gpu_time_s << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << render_gpu_pct << "%"
                  << " " << std::setw(7) << std::setprecision(0) << render_gpu_fps << " fps)"
                  << " fwd: " << std::setw(5) << std::setprecision(1) << train_forward_time_s << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << train_forward_pct << "% "
                  << std::setw(6) << std::setprecision(3) << train_forward_avg_ms << "ms)"
                  << " bwd: " << std::setw(5) << std::setprecision(1) << train_backward_time_s << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << train_backward_pct << "% "
                  << std::setw(6) << std::setprecision(3) << train_backward_avg_ms << "ms)"
                  << " upd: " << std::setw(5) << std::setprecision(1) << train_update_time_s << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << train_update_pct << "% "
                  << std::setw(6) << std::setprecision(3) << train_update_avg_ms << "ms)"
                  << " epoch_time: " << std::setw(6) << std::setprecision(1) << epoch_elapsed.count() << "s"
                  << " total: " << std::setw(8) << std::setprecision(1) << training_elapsed.count() << "s"
                  << std::endl;

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
        rlt::set_step(device, device.logger, epoch_i);
        rlt::add_scalar(device, device.logger, "training/mse_loss", epoch_loss);
        rlt::add_scalar(device, device.logger, "training/episode_length", mean_episode_length);
        rlt::add_scalar(device, device.logger, "training/episodes", static_cast<T>(episode_count));
        if(episode_count_tf > 0){
            rlt::add_scalar(device, device.logger, "training/teacher/episode_length", mean_episode_length_tf);
            rlt::add_scalar(device, device.logger, "training/teacher/episodes", static_cast<T>(episode_count_tf));
        }
        if(episode_count_student > 0){
            rlt::add_scalar(device, device.logger, "training/student/episode_length", mean_episode_length_student);
            rlt::add_scalar(device, device.logger, "training/student/episodes", static_cast<T>(episode_count_student));
        }
        rlt::add_scalar(device, device.logger, "training/fps", fps);
        rlt::add_scalar(device, device.logger, "training/throughput_fps", fps);
        rlt::add_scalar(device, device.logger, "training/render_time_s", render_time_s);
        rlt::add_scalar(device, device.logger, "training/render_fps", render_fps);
        rlt::add_scalar(device, device.logger, "training/render_pct", render_pct);
        rlt::add_scalar(device, device.logger, "training/render_gpu_time_s", render_gpu_time_s);
        rlt::add_scalar(device, device.logger, "training/render_gpu_fps", render_gpu_fps);
        rlt::add_scalar(device, device.logger, "training/render_gpu_pct", render_gpu_pct);
        rlt::add_scalar(device, device.logger, "training/model_forward_time_s", train_forward_time_s);
        rlt::add_scalar(device, device.logger, "training/model_backward_time_s", train_backward_time_s);
        rlt::add_scalar(device, device.logger, "training/model_update_time_s", train_update_time_s);
        rlt::add_scalar(device, device.logger, "training/model_forward_pct", train_forward_pct);
        rlt::add_scalar(device, device.logger, "training/model_backward_pct", train_backward_pct);
        rlt::add_scalar(device, device.logger, "training/model_update_pct", train_update_pct);
        rlt::add_scalar(device, device.logger, "training/model_forward_avg_ms", train_forward_avg_ms);
        rlt::add_scalar(device, device.logger, "training/model_backward_avg_ms", train_backward_avg_ms);
        rlt::add_scalar(device, device.logger, "training/model_update_avg_ms", train_update_avg_ms);
        rlt::add_scalar(device, device.logger, "training/epoch_time_s", epoch_elapsed.count());
        rlt::add_scalar(device, device.logger, "training/total_time_s", training_elapsed.count());
        rlt::add_scalar(device, device.logger, "training/teacher_forcing", full_teacher_forcing ? (T)1 : TEACHER_FORCING_FRACTION);
        rlt::add_scalar(device, device.logger, "curriculum/episode_step_limit", static_cast<T>(current_episode_step_limit));
#endif

        if(epoch_i % CHECKPOINT_CADENCE == 0){
            auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, epoch_end_step);
            static constexpr TI CHECKPOINT_BATCH_SIZE = 1;
            using EVAL_TYPE = typename CPU_STUDENT_TYPE::template CHANGE_BATCH_SIZE<TI, CHECKPOINT_BATCH_SIZE>;
            EVAL_TYPE eval_student;
            rlt::malloc(device, eval_student);
            rlt::copy(device_gpu, device, student_gpu, eval_student);
            char fov_buf[32];
            std::snprintf(fov_buf, sizeof(fov_buf), "%.6g", (double)env_parameters[0].fov);
            std::string state_obs_string = rlt::string(device, envs[0].dynamics, ACTOR_STATE_OBS{});
            std::string image_obs_string;
#ifdef USE_FRAME_STACKING
#ifdef STACK_TARGET_CHANNEL
            image_obs_string = std::string("CameraRGBStackedWithTarget(") + fov_buf + ", "
                + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ", "
                + std::to_string(FRAME_STACK_STRIDE) + ", " + std::to_string(FRAME_STACK_N) + ")";
#else
            image_obs_string = std::string("CameraRGBStacked(") + fov_buf + ", "
                + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ", "
                + std::to_string(FRAME_STACK_STRIDE) + ", " + std::to_string(FRAME_STACK_N) + ")";
#endif
#else
#ifdef STACK_TARGET_CHANNEL
            image_obs_string = std::string("CameraRGBWithTarget(") + fov_buf + ", "
                + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ")";
#else
            image_obs_string = std::string("CameraRGB(") + fov_buf + ", "
                + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ")";
#endif
#endif
#ifdef STACK_TARGET_CHANNEL
            std::string obs_string = image_obs_string + ", " + state_obs_string;
#else
            std::string obs_string = "TargetImage(" + image_obs_string + "), " + image_obs_string + ", " + state_obs_string;
#endif
            std::string meta = "{\"environment\": {\"name\": \"l2f_visual\", \"observation\": \"" + obs_string + "\", \"output\": \"Action\"}}";
#ifdef STACK_TARGET_CHANNEL
            static constexpr TI TOTAL_INPUT_DIM = COMBINED_OBS_DIM + STATE_OBS_DIM;
#else
            static constexpr TI TOTAL_INPUT_DIM = STACKED_OBS_DIM + STACKED_OBS_DIM + STATE_OBS_DIM;
#endif
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, TOTAL_INPUT_DIM>, true>> example_input;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, TARGET_DIM>, true>> example_output;
            rlt::malloc(device, example_input);
            rlt::malloc(device, example_output);
            {
                rlt::randn(device, example_input, rng);
#ifdef STACK_TARGET_CHANNEL
                auto example_input_img = rlt::view_range(device, example_input, (TI)0, rlt::tensor::ViewSpec<1, COMBINED_OBS_DIM>{});
                auto example_input_img_reshaped = rlt::reshape_row_major(device, example_input_img, rlt::tensor::Shape<TI, 1, IMG_H, IMG_W, COMBINED_IMG_C>{});
                auto example_input_state = rlt::view_range(device, example_input, (TI)COMBINED_OBS_DIM, rlt::tensor::ViewSpec<1, STATE_OBS_DIM>{});
                using BRANCH_0 = typename rlt::utils::tuple_element<0, typename EVAL_TYPE::SPEC::BRANCH_TUPLE>::type;
                using BRANCH_1 = typename rlt::utils::tuple_element<1, typename EVAL_TYPE::SPEC::BRANCH_TUPLE>::type;
                using BRANCH_0_OUTPUT_SHAPE = rlt::nn_models::parallel::detail::output_shape<typename EVAL_TYPE::SPEC::CAPABILITY, BRANCH_0>;
                using BRANCH_1_OUTPUT_SHAPE = rlt::nn_models::parallel::detail::output_shape<typename EVAL_TYPE::SPEC::CAPABILITY, BRANCH_1>;
                static constexpr TI BRANCH_0_DIM = rlt::get_last(BRANCH_0_OUTPUT_SHAPE{});
                static constexpr TI BRANCH_1_DIM = rlt::get_last(BRANCH_1_OUTPUT_SHAPE{});
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BRANCH_0_DIM>, true>> branch_0_out;
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BRANCH_1_DIM>, true>> branch_1_out;
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BRANCH_0_DIM + BRANCH_1_DIM>, true>> concat_out;
                typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<0>(eval_student.pipelines))>::template Buffer<true> buffer_0;
                typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<1>(eval_student.pipelines))>::template Buffer<true> buffer_1;
                rlt::malloc(device, branch_0_out);
                rlt::malloc(device, branch_1_out);
                rlt::malloc(device, concat_out);
                rlt::malloc(device, buffer_0);
                rlt::malloc(device, buffer_1);
                rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
                rlt::evaluate(device, rlt::get<0>(eval_student.pipelines), example_input_img_reshaped, branch_0_out, buffer_0, rng, eval_mode);
                rlt::evaluate(device, rlt::get<1>(eval_student.pipelines), example_input_state, branch_1_out, buffer_1, rng, eval_mode);
                auto concat_0 = rlt::view_range(device, concat_out, (TI)0, rlt::tensor::ViewSpec<1, BRANCH_0_DIM>{});
                auto concat_1 = rlt::view_range(device, concat_out, (TI)BRANCH_0_DIM, rlt::tensor::ViewSpec<1, BRANCH_1_DIM>{});
                rlt::copy(device, device, branch_0_out, concat_0);
                rlt::copy(device, device, branch_1_out, concat_1);
#else
                auto example_input_target_img = rlt::view_range(device, example_input, (TI)0, rlt::tensor::ViewSpec<1, STACKED_OBS_DIM>{});
                auto example_input_target_img_reshaped = rlt::reshape_row_major(device, example_input_target_img, rlt::tensor::Shape<TI, 1, IMG_H, IMG_W, STACKED_IMG_C>{});
                auto example_input_img = rlt::view_range(device, example_input, (TI)STACKED_OBS_DIM, rlt::tensor::ViewSpec<1, STACKED_OBS_DIM>{});
                auto example_input_img_reshaped = rlt::reshape_row_major(device, example_input_img, rlt::tensor::Shape<TI, 1, IMG_H, IMG_W, STACKED_IMG_C>{});
                auto example_input_state = rlt::view_range(device, example_input, (TI)(2 * STACKED_OBS_DIM), rlt::tensor::ViewSpec<1, STATE_OBS_DIM>{});
                using BRANCH_0 = typename rlt::utils::tuple_element<0, typename EVAL_TYPE::SPEC::BRANCH_TUPLE>::type;
                using BRANCH_1 = typename rlt::utils::tuple_element<1, typename EVAL_TYPE::SPEC::BRANCH_TUPLE>::type;
                using BRANCH_2 = typename rlt::utils::tuple_element<2, typename EVAL_TYPE::SPEC::BRANCH_TUPLE>::type;
                using BRANCH_0_OUTPUT_SHAPE = rlt::nn_models::parallel::detail::output_shape<typename EVAL_TYPE::SPEC::CAPABILITY, BRANCH_0>;
                using BRANCH_1_OUTPUT_SHAPE = rlt::nn_models::parallel::detail::output_shape<typename EVAL_TYPE::SPEC::CAPABILITY, BRANCH_1>;
                using BRANCH_2_OUTPUT_SHAPE = rlt::nn_models::parallel::detail::output_shape<typename EVAL_TYPE::SPEC::CAPABILITY, BRANCH_2>;
                static constexpr TI BRANCH_0_DIM = rlt::get_last(BRANCH_0_OUTPUT_SHAPE{});
                static constexpr TI BRANCH_1_DIM = rlt::get_last(BRANCH_1_OUTPUT_SHAPE{});
                static constexpr TI BRANCH_2_DIM = rlt::get_last(BRANCH_2_OUTPUT_SHAPE{});
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BRANCH_0_DIM>, true>> branch_0_out;
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BRANCH_1_DIM>, true>> branch_1_out;
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BRANCH_2_DIM>, true>> branch_2_out;
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BRANCH_0_DIM + BRANCH_1_DIM + BRANCH_2_DIM>, true>> concat_out;
                typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<0>(eval_student.pipelines))>::template Buffer<true> buffer_0;
                typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<1>(eval_student.pipelines))>::template Buffer<true> buffer_1;
                typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<2>(eval_student.pipelines))>::template Buffer<true> buffer_2;
                rlt::malloc(device, branch_0_out);
                rlt::malloc(device, branch_1_out);
                rlt::malloc(device, branch_2_out);
                rlt::malloc(device, concat_out);
                rlt::malloc(device, buffer_0);
                rlt::malloc(device, buffer_1);
                rlt::malloc(device, buffer_2);
                rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
                rlt::evaluate(device, rlt::get<0>(eval_student.pipelines), example_input_target_img_reshaped, branch_0_out, buffer_0, rng, eval_mode);
                rlt::evaluate(device, rlt::get<1>(eval_student.pipelines), example_input_img_reshaped, branch_1_out, buffer_1, rng, eval_mode);
                rlt::evaluate(device, rlt::get<2>(eval_student.pipelines), example_input_state, branch_2_out, buffer_2, rng, eval_mode);
                auto concat_0 = rlt::view_range(device, concat_out, (TI)0, rlt::tensor::ViewSpec<1, BRANCH_0_DIM>{});
                auto concat_1 = rlt::view_range(device, concat_out, (TI)BRANCH_0_DIM, rlt::tensor::ViewSpec<1, BRANCH_1_DIM>{});
                auto concat_2 = rlt::view_range(device, concat_out, (TI)(BRANCH_0_DIM + BRANCH_1_DIM), rlt::tensor::ViewSpec<1, BRANCH_2_DIM>{});
                rlt::copy(device, device, branch_0_out, concat_0);
                rlt::copy(device, device, branch_1_out, concat_1);
                rlt::copy(device, device, branch_2_out, concat_2);
#endif
#ifdef USE_GRU_TEMPORAL
                typename decltype(eval_student.head)::State<true> head_state;
                typename decltype(eval_student.head)::template Buffer<true> head_buffer;
                rlt::malloc(device, head_state);
                rlt::malloc(device, head_buffer);
                rlt::reset(device, eval_student.head, head_state, rng);
                rlt::evaluate_step(device, eval_student.head, concat_out, head_state, example_output, head_buffer, rng, eval_mode);
                rlt::free(device, head_state);
                rlt::free(device, head_buffer);
#else
                typename decltype(eval_student.head)::template Buffer<true> head_buffer;
                rlt::malloc(device, head_buffer);
                rlt::evaluate(device, eval_student.head, concat_out, example_output, head_buffer, rng, eval_mode);
                rlt::free(device, head_buffer);
#endif
                rlt::free(device, branch_0_out);
                rlt::free(device, branch_1_out);
#ifndef STACK_TARGET_CHANNEL
                rlt::free(device, branch_2_out);
#endif
                rlt::free(device, concat_out);
                rlt::free(device, buffer_0);
                rlt::free(device, buffer_1);
#ifndef STACK_TARGET_CHANNEL
                rlt::free(device, buffer_2);
#endif
            }
            { // binary (tar)
                std::filesystem::path checkpoint_path = step_folder / "checkpoint.tar";
                rlt::persist::backends::tar::Writer writer;
                rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> root_group{"", &writer};
                auto actor_group = rlt::create_group(device, root_group, "actor");
                rlt::set_attribute(device, actor_group, "checkpoint_name", step_folder.string().c_str());
                rlt::set_attribute(device, actor_group, "meta", meta.c_str());
                rlt::save(device, eval_student, actor_group);
                auto example_group = rlt::create_group(device, root_group, "example");
                rlt::save(device, example_input, example_group, "input");
                rlt::save(device, example_output, example_group, "output");
                rlt::persist::backends::tar::finalize(device, writer);
                std::ofstream f(checkpoint_path, std::ios::binary);
                f.write(writer.buffer.data(), writer.buffer.size());
            }
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
            { // binary (hdf5)
                std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
                std::filesystem::path checkpoint_path = step_folder / "checkpoint.h5";
                rlt::persist::backends::hdf5::File root_file(checkpoint_path.string(), rlt::persist::backends::hdf5::Mode::WRITE);
                auto actor_group = rlt::create_group(device, root_file, "actor");
                rlt::set_attribute(device, actor_group, "checkpoint_name", step_folder.string().c_str());
                rlt::set_attribute(device, actor_group, "meta", meta.c_str());
                rlt::save(device, eval_student, actor_group);
                auto example_group = rlt::create_group(device, root_file, "example");
                rlt::save(device, example_input, example_group, "input");
                rlt::save(device, example_output, example_group, "output");
            }
#endif
            { // code (checkpoint.h)
                auto actor_weights = rlt::save_code(device, eval_student, std::string("rl_tools::checkpoint::actor"), true);
                std::stringstream output_ss;
                output_ss << actor_weights;
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
                }
#endif
                {
                    std::filesystem::path checkpoint_code_path = step_folder / "checkpoint.h";
                    std::ofstream f(checkpoint_code_path);
                    f << output_string;
                }
            }
            rlt::free(device, example_input);
            rlt::free(device, example_output);
            rlt::free(device, eval_student);
            std::cerr << "Checkpoint saved: " << step_folder << std::endl;
        }

        episode_length_sum_tf = 0;
        episode_count_tf = 0;
        episode_length_sum_student = 0;
        episode_count_student = 0;
    }

    std::cout << "Training finished." << std::endl;

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
    rlt::free(device, device.logger);
#endif

    // =========================================================================
    // Cleanup
    // =========================================================================
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        envs[env_i].renderer = nullptr;
        envs[env_i].scene = nullptr;
    }
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::free(device, envs[env_i]);
    }
    for(TI scene_i = 0; scene_i < N_TOTAL_SCENES; scene_i++){
        if(renderers[scene_i] != nullptr){
            rlt::free(device, *renderers[scene_i]);
            delete renderers[scene_i];
            renderers[scene_i] = nullptr;
        }
        if(scenes[scene_i] != nullptr){
            delete scenes[scene_i];
            scenes[scene_i] = nullptr;
        }
    }

    rlt::free(device, raptor);
    rlt::free(device, raptor_buffer);
    rlt::free(device, raptor_state);
    rlt::free(device, teacher_obs);
    rlt::free(device, teacher_actions);
    rlt::free(device, cpu_state_obs_step);
    rlt::free(device, cpu_all_teacher_actions);
    rlt::free(device, student_cpu);
    rlt::free(device, rng);

    rlt::free(device_gpu, rng_gpu);
    rlt::free(device_gpu, gpu_teacher_actions_step);
    cudaFree(gpu_cameras);
    cudaFree(gpu_target_cameras);
    cudaEventDestroy(cameras_ready_event);
    cudaEventDestroy(target_cameras_ready_event);
    for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
        cudaEventDestroy(render_scatter_done_events[active_scene_i]);
        cudaEventDestroy(target_render_scatter_done_events[active_scene_i]);
    }
    for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            TI event_i = step_i * N_ACTIVE_SCENES + active_scene_i;
            cudaEventDestroy(render_pass_start_events[event_i]);
            cudaEventDestroy(render_pass_stop_events[event_i]);
            cudaEventDestroy(target_render_pass_start_events[event_i]);
            cudaEventDestroy(target_render_pass_stop_events[event_i]);
        }
    }
    for(TI call_i = 0; call_i < max_train_timing_calls; call_i++){
        cudaEventDestroy(train_forward_start_events[call_i]);
        cudaEventDestroy(train_forward_stop_events[call_i]);
        cudaEventDestroy(train_backward_start_events[call_i]);
        cudaEventDestroy(train_backward_stop_events[call_i]);
        cudaEventDestroy(train_update_start_events[call_i]);
        cudaEventDestroy(train_update_stop_events[call_i]);
    }
    cudaFree(gpu_logged_batch_losses);
    rlt::free(device_gpu, student_gpu);
    rlt::free(device_gpu, rollout_student_gpu);
    rlt::free(device_gpu, rollout_student_buffers);
#ifdef USE_GRU_TEMPORAL
    rlt::free(device_gpu, rollout_student_state_gpu);
#endif
    rlt::free(device_gpu, raptor_gpu);
    rlt::free(device_gpu, raptor_buffer_gpu);
    rlt::free(device_gpu, raptor_state_gpu);
    rlt::free(device_gpu, gpu_teacher_obs);
    rlt::free(device_gpu, student_buffers);
    rlt::free(device_gpu, optimizer_gpu);
    rlt::free(device_gpu, gpu_all_observations);
    rlt::free(device_gpu, gpu_all_target_observations);
    rlt::free(device_gpu, gpu_all_state_observations);
    rlt::free(device_gpu, gpu_all_targets);
    rlt::free(device_gpu, gpu_d_action_train);
    rlt::free(device_gpu, gpu_student_output_train);
    rlt::free(device_gpu, gpu_student_actions_step);
    cudaFree(gpu_dynamics_arr);
    cudaFree(gpu_params_arr);
    cudaFree(gpu_states_arr);
    cudaFree(gpu_terminated_arr);
    cudaFree(gpu_episode_step_arr);
    cudaFree(gpu_teacher_forcing_arr);
    cudaFree(gpu_episode_return_arr);
    cudaFree(gpu_needs_reset);
    cudaFree(gpu_episode_lengths_log);
    cudaFree(gpu_episode_tf_log);
    cudaFree(gpu_epoch_episode_stats);
    cudaFree(gpu_brightness_scale_arr);
    cudaFree(gpu_scene_translation_arr);
    cudaFree(gpu_scene_yaw_arr);
    cudaFree(gpu_scene_yaw_cos_arr);
    cudaFree(gpu_scene_yaw_sin_arr);
    cudaFree(gpu_indoor_positions);
    cudaFree(gpu_num_indoor_positions);
    cudaFree(gpu_env_scene);
#if defined(STACK_TARGET_CHANNEL) && !defined(USE_FRAME_STACKING)
    rlt::free(device_gpu, gpu_rollout_combined);
#endif
#ifdef USE_GRU_TEMPORAL
    rlt::free(device_gpu, gpu_rollout_branch_a);
    rlt::free(device_gpu, gpu_rollout_branch_b);
    rlt::free(device_gpu, gpu_rollout_concat);
    rlt::free(device_gpu, gpu_rollout_actions);
#elif defined(USE_FRAME_STACKING)
#ifdef STACK_TARGET_CHANNEL
    rlt::free(device_gpu, gpu_frame_stack_history);
    rlt::free(device_gpu, gpu_all_combined_observations);
#else
    rlt::free(device_gpu, gpu_stacked_batch);
    rlt::free(device_gpu, gpu_stacked_target_batch);
    cudaFree(gpu_gather_indices);
    cudaFree(gpu_episode_start_step_per_row);
    cudaFree(gpu_rollout_gather_indices);
    cudaFree(gpu_target_gather_indices);
    rlt::free(device_gpu, gpu_rollout_stacked);
    rlt::free(device_gpu, gpu_rollout_stacked_target);
    cudaFree(gpu_rollout_target_gather_indices);
#endif
    cudaFree(gpu_episode_start_step);
#endif

    return 0;
}
