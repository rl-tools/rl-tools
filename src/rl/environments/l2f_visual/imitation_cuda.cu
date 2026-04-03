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

#include <array>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cstring>
#include <string>
#include <filesystem>
#include <fstream>

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
    false, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
};
static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
    0.2, 0.0, 0.3, 0.0, 1.0, true, -1, +1,
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    true, 0.5, 10, 35, 10000, 50000,
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

using ACTOR_STATE_OBS = obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI, obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>;
// using ACTOR_STATE_OBS = STATIC_PARAMETERS::OBSERVATION_TYPE;
static constexpr TI STATE_OBS_DIM = ACTOR_STATE_OBS::DIM; // 12


// =========================================================================
// Visual environment specification
// =========================================================================
static constexpr TI N_ENVIRONMENTS = 64;
static constexpr TI CAM_WIDTH = 64;
static constexpr TI CAM_HEIGHT = 64;
static constexpr TI NUM_PROBES = 64;

using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, N_ENVIRONMENTS, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES>;
using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;

// =========================================================================
// RAPTOR teacher (CPU only)
// =========================================================================
static constexpr TI RAPTOR_HIDDEN_DIM = 16;
static constexpr TI RAPTOR_OBS_DIM = 22; // Position(3) + RotMat(9) + LinVel(3) + AngVel(3) + ActionHistory(4)

using RAPTOR_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
using RAPTOR_DENSE1 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE1_CONFIG>;
using RAPTOR_GRU_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM>;
using RAPTOR_GRU = rlt::nn::layers::gru::BindConfiguration<RAPTOR_GRU_CONFIG>;
using RAPTOR_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
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
static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
static constexpr TI BATCH_SIZE = 512;
static constexpr TI STEPS_PER_ENV = 500;
static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * N_ENVIRONMENTS;
static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
static constexpr TI NUM_EPOCHS = 1000000;
static constexpr TI TEACHER_FORCING_EPOCHS = 0;
static constexpr T TEACHER_FORCING_FRACTION = 0.0;
static constexpr TI N_TRAIN_PASSES = 4;
static constexpr TI VIDEO_CADENCE = 10;
static constexpr TI CHECKPOINT_CADENCE = 100;
static constexpr TI GRID_SIDE = 8; // sqrt(N_ENVIRONMENTS)
static_assert(GRID_SIDE * GRID_SIDE == N_ENVIRONMENTS, "N_ENVIRONMENTS must be a perfect square for video mosaic");

static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");

// =========================================================================
// Frame stacking configuration
// =========================================================================
#define USE_FRAME_STACKING
// #define USE_GRU_TEMPORAL
#if defined(USE_FRAME_STACKING) && defined(USE_GRU_TEMPORAL)
#error "USE_FRAME_STACKING and USE_GRU_TEMPORAL are mutually exclusive"
#endif
#ifdef USE_FRAME_STACKING
static constexpr TI FRAME_STACK_N = 5;
static constexpr TI FRAME_STACK_STRIDE = 20; // 100Hz / 20 = 5Hz
static constexpr TI STACKED_IMG_C = ENVIRONMENT::Observation::CHANNELS * FRAME_STACK_N;
static constexpr TI STACKED_OBS_DIM = ENVIRONMENT::Observation::HEIGHT * ENVIRONMENT::Observation::WIDTH * STACKED_IMG_C;
#elif defined(USE_GRU_TEMPORAL)
static constexpr TI BPTT_STEPS = 100;
static constexpr TI GRU_HIDDEN_DIM = 64;
static constexpr TI STACKED_IMG_C = ENVIRONMENT::Observation::CHANNELS;
static constexpr TI STACKED_OBS_DIM = OBSERVATION_DIM;
static constexpr TI EMBED_DIM = ACTOR_HIDDEN_DIM * 2;
static constexpr TI N_WINDOWS = STEPS_PER_ENV / BPTT_STEPS;
static constexpr TI WINDOW_SAMPLES = BPTT_STEPS * N_ENVIRONMENTS;
static_assert(STEPS_PER_ENV % BPTT_STEPS == 0);
#else
static constexpr TI STACKED_IMG_C = ENVIRONMENT::Observation::CHANNELS;
static constexpr TI STACKED_OBS_DIM = OBSERVATION_DIM;
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
        T* episode_lengths_log, T* episode_returns_log, T* episode_tf_log,
        T teacher_forcing_fraction, bool full_teacher_forcing,
        T* teacher_obs_ptr, T* state_obs_ptr,
        T* raptor_gru_state_ptr, T* raptor_gru_initial_hidden_ptr, TI* raptor_gru_step_ptr,
#ifdef USE_GRU_TEMPORAL
        T* student_gru_state_ptr, T* student_gru_initial_hidden_ptr, TI* student_gru_step_ptr,
#endif
#ifdef USE_FRAME_STACKING
        TI* episode_start_step,
#endif
        RNG rng, TI step_i
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& rng_state = rl_tools::get(rng.states, 0, env_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        bool need_reset = terminated_flags[env_i] || episode_step_arr[env_i] >= EPISODE_STEP_LIMIT;
        needs_reset_flags[env_i] = need_reset;
        if(need_reset){
            if(episode_step_arr[env_i] > 0){
                episode_lengths_log[env_i] = (T)episode_step_arr[env_i];
                episode_returns_log[env_i] = episode_return_arr[env_i];
                episode_tf_log[env_i] = teacher_forcing_arr[env_i] ? (T)1 : (T)0;
            } else {
                episode_lengths_log[env_i] = (T)-1;
                episode_returns_log[env_i] = (T)0;
                episode_tf_log[env_i] = (T)0;
            }
            rl_tools::sample_initial_parameters(device, env, params, rng_state);
            rl_tools::sample_initial_state(device, env, params, state, rng_state);
            episode_step_arr[env_i] = 0;
            terminated_flags[env_i] = false;
            episode_return_arr[env_i] = (T)0;
            teacher_forcing_arr[env_i] = full_teacher_forcing || rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, rng_state) < teacher_forcing_fraction;
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
            episode_returns_log[env_i] = (T)0;
            episode_tf_log[env_i] = (T)0;
        }
        {
            rlt::Matrix<rlt::matrix::Specification<T, TI, 1, RAPTOR_OBS_DIM, true, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> obs_mat;
            obs_mat._data = teacher_obs_ptr + env_i * RAPTOR_OBS_DIM;
            rl_tools::observe(device, env, params, state, typename STATIC_PARAMETERS::OBSERVATION_TYPE{}, obs_mat, rng_state);
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
        bool* terminated_flags, TI* episode_step_arr, bool* teacher_forcing_arr, T* episode_return_arr,
        T* teacher_actions_ptr, T* student_actions_ptr, T* all_teacher_actions_ptr,
        RNG rng, TI step_i
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& rng_state = rl_tools::get(rng.states, 0, env_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        TI pos = step_i * N_ENVIRONMENTS + env_i;
        T action_arr[ACTION_DIM];
        T* src = teacher_forcing_arr[env_i] ? teacher_actions_ptr : student_actions_ptr;
        for(TI a = 0; a < ACTION_DIM; a++){
            action_arr[a] = src[env_i * ACTION_DIM + a];
            all_teacher_actions_ptr[pos * ACTION_DIM + a] = teacher_actions_ptr[env_i * ACTION_DIM + a];
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
        rlt::CameraData* gpu_cameras,
        T cos_fov, T aspect,
        T camera_offset_body_0, T camera_offset_body_1, T camera_offset_body_2,
        T camera_forward_body_0, T camera_forward_body_1, T camera_forward_body_2,
        T camera_up_body_0, T camera_up_body_1, T camera_up_body_2,
        T scene_translation_0, T scene_translation_1, T scene_translation_2
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& state = states[env_i];
        T offset_body[3] = {camera_offset_body_0, camera_offset_body_1, camera_offset_body_2};
        T forward_body[3] = {camera_forward_body_0, camera_forward_body_1, camera_forward_body_2};
        T up_body[3] = {camera_up_body_0, camera_up_body_1, camera_up_body_2};
        T cam_pos_world[3];
        rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, offset_body, cam_pos_world);
        T cam_forward_world[3];
        rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, forward_body, cam_forward_world);
        T cam_up_world[3];
        rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, up_body, cam_up_world);
        T px = state.position[0] + cam_pos_world[0] + scene_translation_0;
        T py = state.position[2] + cam_pos_world[2] + scene_translation_1;
        T pz = state.position[1] + cam_pos_world[1] + scene_translation_2;
        owl::vec3f position(px, py, pz);
        owl::vec3f look_at(px + cam_forward_world[0], py + cam_forward_world[2], pz + cam_forward_world[1]);
        owl::vec3f up(cam_up_world[0], cam_up_world[2], cam_up_world[1]);
        gpu_cameras[env_i] = rl_tools::make_camera_data(position, look_at, up, cos_fov, aspect);
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
    static constexpr T ALPHA = 3e-4;
    static constexpr T EPSILON = 1e-5;
    static constexpr T EPSILON_SQRT = 1e-5;
};

template<typename CAPABILITY>
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

    using IMAGE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, IMG_H, IMG_W, STACKED_IMG_C>;
    using STATE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, STATE_OBS_DIM>;

    // Image branch: Flatten→Standardize→Unflatten→Conv(s4)→Conv(s2)→Flatten→Dense(64)
    using INPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using INPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<INPUT_FLATTEN_CONFIG>;
    using IMAGE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
    using IMAGE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<IMAGE_STANDARDIZE_CONFIG>;
    using UNFLATTEN_CONFIG = rlt::nn::layers::unflatten::Configuration<TYPE_POLICY, TI, IMG_H, IMG_W, STACKED_IMG_C>;
    using UNFLATTEN = rlt::nn::layers::unflatten::BindConfiguration<UNFLATTEN_CONFIG>;
    using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 32, 4, 4, 4, 4, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
    using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 64, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
    using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
    using IMAGE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using IMAGE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<IMAGE_DENSE_EMBED_CONFIG>;
    using IMAGE_BRANCH = rlt::nn_models::sequential::Module<INPUT_FLATTEN, IMAGE_STANDARDIZE, UNFLATTEN, CONV1, CONV2, OUTPUT_FLATTEN, IMAGE_DENSE_EMBED>;

    // State branch: Standardize→Dense(64)
    using STATE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
    using STATE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STATE_STANDARDIZE_CONFIG>;
    using STATE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using STATE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<STATE_DENSE_EMBED_CONFIG>;
    using STATE_BRANCH = rlt::nn_models::sequential::Module<STATE_STANDARDIZE, STATE_DENSE_EMBED>;

    // Head MLP: 128D → 64 → 64 → ACTION_DIM (plain MLP, no log_std)
    using MLP_HEAD_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, ACTION_DIM, 2, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
    using MLP_HEAD = rlt::nn_models::mlp::BindConfiguration<MLP_HEAD_CONFIG>;

#ifdef USE_GRU_TEMPORAL
    using GRU_HEAD_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, GRU_HIDDEN_DIM>;
    using GRU_HEAD = rlt::nn::layers::gru::BindConfiguration<GRU_HEAD_CONFIG>;
    using HEAD_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using HEAD_DENSE1 = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE1_CONFIG>;
    using HEAD_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using HEAD_DENSE2 = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE2_CONFIG>;
    using HEAD_DENSE_OUT_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD_DENSE_OUT = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE_OUT_CONFIG>;
    using SEQUENTIAL_HEAD = rlt::nn_models::sequential::Module<GRU_HEAD, HEAD_DENSE1, HEAD_DENSE2, HEAD_DENSE_OUT>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, IMAGE_BRANCH, STATE_BRANCH, IMAGE_INPUT_SHAPE, STATE_INPUT_SHAPE, SEQUENTIAL_HEAD>;
#else
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, IMAGE_BRANCH, STATE_BRANCH, IMAGE_INPUT_SHAPE, STATE_INPUT_SHAPE, MLP_HEAD>;
#endif
};

using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, true>;
using STUDENT_TYPE = typename StudentActor<CAPABILITY_ADAM>::MODEL;
using STUDENT_BUFFERS = typename STUDENT_TYPE::Buffer<true>;
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

#ifdef USE_FRAME_STACKING
__global__ void gather_frames_kernel(
    const float* __restrict__ all_obs,
    const int* __restrict__ gather_idx,
    float* __restrict__ stacked_out,
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
    stacked_out[global_idx] = all_obs[src_row * obs_dim + pixel * img_c + channel];
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
        std::cerr << "Usage: " << argv[0] << " <conta:HASH or scene.glb> [seed]" << std::endl;
        return 1;
    }
    if(argc > 2){
        seed = std::atoi(argv[2]);
    }

    // Resolve scene path and hash
    std::string resolved_scene_path;
    rlt::rl::environments::l2f_visual::SceneHash scene_hash;
    const char* scene_arg = argv[1];
    if(std::strncmp(scene_arg, "conta:", 6) == 0){
        const char* hash_str = scene_arg + 6;
        if(std::strlen(hash_str) != 40){
            std::cerr << "Invalid conta hash: expected 40 hex characters, got " << std::strlen(hash_str) << std::endl;
            return 1;
        }
        if(!parse_hex_hash(hash_str, scene_hash.hash, rlt::rl::environments::l2f_visual::SceneHash::HASH_SIZE)){
            std::cerr << "Invalid conta hash: contains non-hex characters" << std::endl;
            return 1;
        }
        const char* conta_root = std::getenv("CONTA_ROOT");
        if(!conta_root){
            std::cerr << "CONTA_ROOT environment variable is not set" << std::endl;
            return 1;
        }
        resolved_scene_path = std::string(conta_root) + "/data/" + hash_str;
    } else {
        resolved_scene_path = scene_arg;
        std::memset(scene_hash.hash, 0, rlt::rl::environments::l2f_visual::SceneHash::HASH_SIZE);
    }
    const char* scene_path = resolved_scene_path.c_str();

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
    // Student (CPU copy for warmup)
    // =========================================================================
    STUDENT_TYPE student_cpu;
    rlt::malloc(device, student_cpu);
    rlt::init_weights(device, student_cpu, rng);

    // Warmup observations (CPU)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, OBSERVATION_DIM>>> warmup_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, STATE_OBS_DIM>>> warmup_state_observations;
    rlt::malloc(device, warmup_observations);
    rlt::malloc(device, warmup_state_observations);

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

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        env_parameters[env_i].scene_translation[0] = -3.92;
        env_parameters[env_i].scene_translation[1] =  1.0;
        env_parameters[env_i].scene_translation[2] =  5.67;
        env_parameters[env_i].scene_hash = scene_hash;
    }

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
    // Observation normalization warmup (CPU)
    // =========================================================================
    {
        STUDENT_BUFFERS student_buffers_cpu;
        rlt::malloc(device, student_buffers_cpu);

        std::cout << "Running observation normalization warmup..." << std::endl;
        for(TI obs_row = 0; obs_row < STEPS_TOTAL; obs_row++){
            TI env_i = obs_row % N_ENVIRONMENTS;
            typename ENVIRONMENT::State warmup_state;
            rlt::sample_initial_parameters(device, envs[env_i], env_parameters[env_i], rng);
            rlt::sample_initial_state(device, envs[env_i], env_parameters[env_i], warmup_state, rng);

            auto obs_slice = rlt::view(device, warmup_observations, obs_row);
            auto obs_matrix = rlt::matrix_view(device, obs_slice);
            rlt::observe(device, envs[env_i], env_parameters[env_i], warmup_state, typename ENVIRONMENT::Observation{}, obs_matrix, rng);

            auto state_obs_slice = rlt::view(device, warmup_state_observations, obs_row);
            auto state_obs_matrix = rlt::matrix_view(device, state_obs_slice);
            rlt::observe(device, envs[env_i].dynamics, env_parameters[env_i].dynamics, warmup_state, ACTOR_STATE_OBS{}, state_obs_matrix, rng);
        }

        rlt::Mode<rlt::nn::layers::standardize::AccumulateMode<>> accumulate_mode;
#ifdef USE_GRU_TEMPORAL
        using IMAGE_INPUT_SHAPE_WARMUP = rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, IMG_H, IMG_W, IMG_C>;
        using STATE_INPUT_SHAPE_WARMUP = rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, STATE_OBS_DIM>;
        static constexpr TI N_BATCHES_WARMUP = STEPS_TOTAL / WINDOW_SAMPLES;
        for(TI batch_i = 0; batch_i < N_BATCHES_WARMUP; batch_i++){
            auto batch_observations = rlt::view_range(device, warmup_observations, batch_i * WINDOW_SAMPLES, rlt::tensor::ViewSpec<0, WINDOW_SAMPLES>{});
            auto batch_observations_reshaped = rlt::reshape_row_major(device, batch_observations, IMAGE_INPUT_SHAPE_WARMUP{});
            auto batch_state_observations = rlt::view_range(device, warmup_state_observations, batch_i * WINDOW_SAMPLES, rlt::tensor::ViewSpec<0, WINDOW_SAMPLES>{});
            auto batch_state_observations_reshaped = rlt::reshape_row_major(device, batch_state_observations, STATE_INPUT_SHAPE_WARMUP{});
            rlt::forward(device, student_cpu, batch_observations_reshaped, batch_state_observations_reshaped, student_buffers_cpu, rng, accumulate_mode);
        }
#else
        using IMAGE_INPUT_SHAPE_WARMUP = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, STACKED_IMG_C>;
        using STATE_INPUT_SHAPE_WARMUP = rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>;
#ifdef USE_FRAME_STACKING
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, STACKED_OBS_DIM>>> warmup_stacked_batch;
        rlt::malloc(device, warmup_stacked_batch);
#endif
        for(TI batch_i = 0; batch_i < N_BATCHES; batch_i++){
#ifdef USE_FRAME_STACKING
            for(TI s = 0; s < BATCH_SIZE; s++){
                T* src = rlt::data(warmup_observations) + (batch_i * BATCH_SIZE + s) * OBSERVATION_DIM;
                T* dst = rlt::data(warmup_stacked_batch) + s * STACKED_OBS_DIM;
                for(TI p = 0; p < IMG_H * IMG_W; p++){
                    for(TI f = 0; f < FRAME_STACK_N; f++){
                        for(TI c = 0; c < IMG_C; c++){
                            dst[p * STACKED_IMG_C + f * IMG_C + c] = src[p * IMG_C + c];
                        }
                    }
                }
            }
            auto batch_observations_reshaped = rlt::reshape_row_major(device, warmup_stacked_batch, IMAGE_INPUT_SHAPE_WARMUP{});
#else
            auto batch_observations = rlt::view_range(device, warmup_observations, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            auto batch_observations_reshaped = rlt::reshape_row_major(device, batch_observations, IMAGE_INPUT_SHAPE_WARMUP{});
#endif
            auto batch_state_observations = rlt::view_range(device, warmup_state_observations, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            auto batch_state_observations_reshaped = rlt::reshape_row_major(device, batch_state_observations, STATE_INPUT_SHAPE_WARMUP{});
            rlt::forward(device, student_cpu, batch_observations_reshaped, batch_state_observations_reshaped, student_buffers_cpu, rng, accumulate_mode);
        }
#ifdef USE_FRAME_STACKING
        rlt::free(device, warmup_stacked_batch);
#endif
#endif
        std::cout << "Observation normalization warmup complete." << std::endl;
        rlt::free(device, student_buffers_cpu);
    }
    rlt::free(device, warmup_observations);
    rlt::free(device, warmup_state_observations);

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
#ifdef USE_GRU_TEMPORAL
    typename STUDENT_TYPE::State<true> student_state_gpu;
    rlt::malloc(device_gpu, student_state_gpu);
    rlt::reset(device_gpu, student_gpu, student_state_gpu, rng_gpu);
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
    rlt::CameraData* gpu_cameras = nullptr;
    cudaMalloc(&gpu_cameras, N_ENVIRONMENTS * sizeof(rlt::CameraData));

    // GPU tensors
    static constexpr TI GPU_OBS_ROWS = STEPS_TOTAL + BATCH_SIZE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, OBSERVATION_DIM>>> gpu_all_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, STATE_OBS_DIM>>> gpu_all_state_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, ACTION_DIM>>> gpu_all_teacher_actions;
#ifdef USE_GRU_TEMPORAL
    rlt::Matrix<rlt::matrix::Specification<T, TI, WINDOW_SAMPLES, ACTION_DIM>> gpu_d_action_train;
#else
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_d_action_train;
#endif
#ifdef USE_GRU_TEMPORAL
    rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ACTION_DIM>> gpu_actions_eval;
#else
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_actions_eval;
#endif
    rlt::malloc(device_gpu, gpu_all_observations);
    rlt::malloc(device_gpu, gpu_all_state_observations);
    rlt::malloc(device_gpu, gpu_all_teacher_actions);
    rlt::malloc(device_gpu, gpu_d_action_train);
    rlt::malloc(device_gpu, gpu_actions_eval);

#ifdef USE_GRU_TEMPORAL
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTOR_HIDDEN_DIM>>> gpu_rollout_branch_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTOR_HIDDEN_DIM>>> gpu_rollout_branch_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, EMBED_DIM>>> gpu_rollout_concat;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTION_DIM>>> gpu_rollout_actions;
    rlt::malloc(device_gpu, gpu_rollout_branch_a);
    rlt::malloc(device_gpu, gpu_rollout_branch_b);
    rlt::malloc(device_gpu, gpu_rollout_concat);
    rlt::malloc(device_gpu, gpu_rollout_actions);
#endif

#ifdef USE_FRAME_STACKING
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, STACKED_OBS_DIM>>> gpu_stacked_batch;
    rlt::malloc(device_gpu, gpu_stacked_batch);
    int* gpu_gather_indices = nullptr;
    TI* gpu_episode_start_step = nullptr;
    TI* gpu_episode_start_step_per_row = nullptr;
    cudaMalloc(&gpu_gather_indices, BATCH_SIZE * FRAME_STACK_N * sizeof(int));
    cudaMalloc(&gpu_episode_start_step, N_ENVIRONMENTS * sizeof(TI));
    cudaMalloc(&gpu_episode_start_step_per_row, STEPS_TOTAL * sizeof(TI));
    cudaMemset(gpu_episode_start_step, 0, N_ENVIRONMENTS * sizeof(TI));
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
    T* gpu_episode_returns_log = nullptr;
    T* gpu_episode_tf_log = nullptr;
    cudaMalloc(&gpu_dynamics_arr, N_ENVIRONMENTS * sizeof(DYNAMICS_TYPE));
    cudaMalloc(&gpu_params_arr, N_ENVIRONMENTS * sizeof(PARAMETERS_TYPE));
    cudaMalloc(&gpu_states_arr, N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::State));
    cudaMalloc(&gpu_terminated_arr, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_step_arr, N_ENVIRONMENTS * sizeof(TI));
    cudaMalloc(&gpu_teacher_forcing_arr, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_return_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_needs_reset, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_lengths_log, STEPS_TOTAL * sizeof(T));
    cudaMalloc(&gpu_episode_returns_log, STEPS_TOTAL * sizeof(T));
    cudaMalloc(&gpu_episode_tf_log, STEPS_TOTAL * sizeof(T));
    std::vector<T> cpu_episode_lengths_log(STEPS_TOTAL);
    std::vector<T> cpu_episode_tf_log(STEPS_TOTAL);
    std::vector<T> cpu_episode_returns_log(STEPS_TOTAL);
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
        for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
            init_terminated[env_i] = true;
            init_step[env_i] = 0;
            init_return[env_i] = (T)0;
        }
        cudaMemcpy(gpu_terminated_arr, init_terminated, N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_step_arr, init_step, N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_return_arr, init_return, N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
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
    static constexpr TI MOSAIC_W = GRID_SIDE * CAM_WIDTH;
    static constexpr TI MOSAIC_H = GRID_SIDE * CAM_HEIGHT;
    std::vector<uint32_t> video_pixel_buffer(N_ENVIRONMENTS * CAM_PIXELS);
    std::vector<uint8_t> mosaic_frame(MOSAIC_W * MOSAIC_H * 3);

    for(TI epoch_i = 0; epoch_i < NUM_EPOCHS; epoch_i++){
        auto epoch_start = std::chrono::high_resolution_clock::now();
        bool full_teacher_forcing = epoch_i < TEACHER_FORCING_EPOCHS;
        bool record_video = (epoch_i % VIDEO_CADENCE == 0);
        TI epoch_end_step = global_step + STEPS_PER_ENV * N_ENVIRONMENTS;
        FILE* ffmpeg_pipe = nullptr;
        if(record_video){
            auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, epoch_end_step);
            auto video_path = step_folder / "video.mp4";
            char ffmpeg_cmd[1024];
            snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
                "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %lux%lu -framerate 25 -i - "
                "-c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast -loglevel warning %s",
                (unsigned long)MOSAIC_W, (unsigned long)MOSAIC_H, video_path.c_str());
            ffmpeg_pipe = popen(ffmpeg_cmd, "w");
            if(!ffmpeg_pipe){
                std::cerr << "Failed to open ffmpeg pipe for " << video_path << std::endl;
                record_video = false;
            }
        }

        // =================================================================
        // Data collection (GPU-resident)
        // =================================================================
        {
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = (N_ENVIRONMENTS + BLOCKSIZE - 1) / BLOCKSIZE;
            dim3 grid(N_BLOCKS);
            dim3 block(BLOCKSIZE);
            rlt::devices::cuda::TAG<DEVICE_GPU, true> tag_device{};
            auto& raptor_gru_layer = rlt::nn_models::sequential::layer<1>(raptor_gpu);
            auto& raptor_gru_state_content = rlt::nn_models::sequential::content_state<1>(raptor_state_gpu.content_state);
#ifdef USE_GRU_TEMPORAL
            auto& student_gru_layer = rlt::nn_models::sequential::layer<0>(student_gpu.head);
            auto& student_gru_state = rlt::nn_models::sequential::content_state<0>(student_state_gpu.head_state.content_state);
#endif
            T cam_aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
            for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
                imitation_kernels::prologue_kernel<<<grid, block, 0, device_gpu.stream>>>(
                    tag_device, gpu_dynamics_arr, gpu_params_arr, gpu_states_arr,
                    gpu_terminated_arr, gpu_episode_step_arr, gpu_teacher_forcing_arr,
                    gpu_episode_return_arr, gpu_needs_reset,
                    gpu_episode_lengths_log + step_i * N_ENVIRONMENTS,
                    gpu_episode_returns_log + step_i * N_ENVIRONMENTS,
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
                    rng_gpu, step_i);
                CUDA_CHECK("prologue_kernel");
#ifdef USE_FRAME_STACKING
                imitation_kernels::record_episode_start_kernel<<<grid, block, 0, device_gpu.stream>>>(tag_device, gpu_episode_start_step, gpu_episode_start_step_per_row, step_i);
#endif
                imitation_kernels::make_cameras_kernel<<<grid, block, 0, device_gpu.stream>>>(
                    tag_device, gpu_dynamics_arr, gpu_params_arr, gpu_states_arr,
                    gpu_cameras,
                    env0.cos_fov, cam_aspect,
                    env0.camera_mount.offset_body[0], env0.camera_mount.offset_body[1], env0.camera_mount.offset_body[2],
                    env0.camera_mount.forward_body[0], env0.camera_mount.forward_body[1], env0.camera_mount.forward_body[2],
                    env0.camera_mount.up_body[0], env0.camera_mount.up_body[1], env0.camera_mount.up_body[2],
                    env_parameters[0].scene_translation[0], env_parameters[0].scene_translation[1], env_parameters[0].scene_translation[2]);
                CUDA_CHECK("make_cameras_kernel");
                T* obs_ptr = rlt::data(gpu_all_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
                if(env0.renderer->cameras_buffer == nullptr){
                    std::array<rlt::CameraData, N_ENVIRONMENTS> cpu_cameras_init;
                    cudaMemcpy(cpu_cameras_init.data(), gpu_cameras, N_ENVIRONMENTS * sizeof(rlt::CameraData), cudaMemcpyDeviceToHost);
                    CUDA_CHECK("cameras D2H init");
                    rlt::observe_batch_render_gpu(device, env0, cpu_cameras_init.data(), N_ENVIRONMENTS, obs_ptr);
                    CUDA_CHECK("observe_batch_render_gpu init");
                } else {
                    void* owl_cam_ptr = (void*)owlBufferGetPointer((OWLBuffer)env0.renderer->cameras_buffer, 0);
                    OWLParams rgb_lp = (OWLParams)env0.renderer->rgb_launch_params;
                    cudaStream_t optix_stream = (cudaStream_t)owlParamsGetCudaStream(rgb_lp, 0);
                    cudaMemcpyAsync(owl_cam_ptr, gpu_cameras, N_ENVIRONMENTS * sizeof(rlt::CameraData), cudaMemcpyDeviceToDevice, optix_stream);
                    CUDA_CHECK("cameras D2D copy");
                    rlt::render_rgb_only_launch(device, *env0.renderer);
                    CUDA_CHECK("render_rgb_only_launch");
                }
                if(record_video && ffmpeg_pipe){
                    rlt::read_frame_buffer(device, *env0.renderer, video_pixel_buffer.data(), video_pixel_buffer.size());
                    for(TI grid_row = 0; grid_row < GRID_SIDE; grid_row++){
                        for(TI grid_col = 0; grid_col < GRID_SIDE; grid_col++){
                            TI env_i = grid_row * GRID_SIDE + grid_col;
                            for(TI py = 0; py < CAM_HEIGHT; py++){
                                for(TI px = 0; px < CAM_WIDTH; px++){
                                    uint32_t rgba = video_pixel_buffer[env_i * CAM_PIXELS + py * CAM_WIDTH + px];
                                    TI mosaic_x = grid_col * CAM_WIDTH + px;
                                    TI mosaic_y = grid_row * CAM_HEIGHT + py;
                                    TI out_idx = (mosaic_y * MOSAIC_W + mosaic_x) * 3;
                                    mosaic_frame[out_idx + 0] = (rgba >>  0) & 0xFF;
                                    mosaic_frame[out_idx + 1] = (rgba >>  8) & 0xFF;
                                    mosaic_frame[out_idx + 2] = (rgba >> 16) & 0xFF;
                                }
                            }
                        }
                    }
                    fwrite(mosaic_frame.data(), 1, mosaic_frame.size(), ffmpeg_pipe);
                }
                rlt::evaluate_step(device_gpu, raptor_gpu, gpu_teacher_obs, raptor_state_gpu, gpu_teacher_actions_step, raptor_buffer_gpu, rng_gpu, no_auto_reset_mode);
                CUDA_CHECK("raptor evaluate_step");
                if(env0.renderer->cameras_buffer != nullptr){
                    rlt::render_rgb_only_sync(device, *env0.renderer);
                    CUDA_CHECK("render_rgb_only_sync");
                    const uint32_t* fb_ptr = (const uint32_t*)owlBufferGetPointer((OWLBuffer)env0.renderer->frame_buffer, 0);
                    constexpr TI TOTAL_PIXELS = N_ENVIRONMENTS * CAM_PIXELS;
                    int pf_block = 256;
                    int pf_grid = (TOTAL_PIXELS + pf_block - 1) / pf_block;
                    rlt::rl::environments::l2f_visual::cuda::pixel_to_float_kernel<<<pf_grid, pf_block>>>(fb_ptr, obs_ptr, N_ENVIRONMENTS, CAM_PIXELS);
                    CUDA_CHECK("pixel_to_float_kernel");
                }
                if(!full_teacher_forcing){
#ifdef USE_GRU_TEMPORAL
                    auto current_img = rlt::view_range(device_gpu, gpu_all_observations, (TI)(step_i * N_ENVIRONMENTS), rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                    auto current_img_reshaped = rlt::reshape_row_major(device_gpu, current_img, rlt::tensor::Shape<TI, N_ENVIRONMENTS, IMG_H, IMG_W, IMG_C>{});
                    auto current_state = rlt::view_range(device_gpu, gpu_all_state_observations, (TI)(step_i * N_ENVIRONMENTS), rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                    rlt::evaluate(device_gpu, student_gpu.pipeline_a, current_img_reshaped, gpu_rollout_branch_a, student_buffers.buffer_a, rng_gpu);
                    rlt::evaluate(device_gpu, student_gpu.pipeline_b, current_state, gpu_rollout_branch_b, student_buffers.buffer_b, rng_gpu);
                    rlt::_concatenate_cuda(device_gpu, gpu_rollout_branch_a, gpu_rollout_branch_b, gpu_rollout_concat);
                    rlt::evaluate_step(device_gpu, student_gpu.head, gpu_rollout_concat, student_state_gpu.head_state, gpu_rollout_actions, student_buffers.head_buffer, rng_gpu, no_auto_reset_mode);
                    CUDA_CHECK("student eval GRU");
#else
#ifdef USE_FRAME_STACKING
                    {
                        constexpr TI GI_BLOCK = 256;
                        constexpr TI GI_GRID = (BATCH_SIZE + GI_BLOCK - 1) / GI_BLOCK;
                        imitation_kernels::compute_gather_indices_rollout_kernel<<<GI_GRID, GI_BLOCK, 0, device_gpu.stream>>>(
                            gpu_gather_indices, gpu_episode_start_step, step_i, BATCH_SIZE);
                    }
                    {
                        int total_elements = BATCH_SIZE * STACKED_OBS_DIM;
                        gather_frames_kernel<<<(total_elements + 255) / 256, 256>>>(
                            rlt::data(gpu_all_observations), gpu_gather_indices, rlt::data(gpu_stacked_batch),
                            OBSERVATION_DIM, IMG_C, FRAME_STACK_N, STACKED_IMG_C, STACKED_OBS_DIM, BATCH_SIZE);
                    }
                    using EVAL_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, STACKED_IMG_C>;
                    auto gpu_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_stacked_batch, EVAL_INPUT_SHAPE{});
#else
                    auto gpu_obs_slice = rlt::view_range(device_gpu, gpu_all_observations, (TI)(step_i * N_ENVIRONMENTS), rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                    using EVAL_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, STACKED_IMG_C>;
                    auto gpu_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_obs_slice, EVAL_INPUT_SHAPE{});
#endif
                    auto gpu_state_obs_slice = rlt::view_range(device_gpu, gpu_all_state_observations, (TI)(step_i * N_ENVIRONMENTS), rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                    auto gpu_state_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_state_obs_slice, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});
                    auto gpu_actions_eval_tensor = rlt::to_tensor(device_gpu, gpu_actions_eval);
                    auto gpu_actions_eval_reshaped = rlt::reshape_row_major(device_gpu, gpu_actions_eval_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                    rlt::evaluate(device_gpu, student_gpu, gpu_obs_reshaped, gpu_state_obs_reshaped, gpu_actions_eval_reshaped, student_buffers, rng_gpu);
                    CUDA_CHECK("student eval");
#endif
                }
                {
#ifdef USE_GRU_TEMPORAL
                    T* student_ptr = rlt::data(gpu_rollout_actions);
#else
                    T* student_ptr = gpu_actions_eval._data;
#endif
                    imitation_kernels::epilogue_kernel<<<grid, block, 0, device_gpu.stream>>>(
                        tag_device, gpu_dynamics_arr, gpu_params_arr, gpu_states_arr,
                        gpu_terminated_arr, gpu_episode_step_arr, gpu_teacher_forcing_arr,
                        gpu_episode_return_arr,
                        rlt::data(gpu_teacher_actions_step), student_ptr,
                        rlt::data(gpu_all_teacher_actions),
                        rng_gpu, step_i);
                }
                CUDA_CHECK("epilogue_kernel");
                global_step += N_ENVIRONMENTS;
            }
        }
        cudaDeviceSynchronize();
        CUDA_CHECK("end of collection sync");
        cudaMemcpy(cpu_episode_lengths_log.data(), gpu_episode_lengths_log, STEPS_TOTAL * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_episode_returns_log.data(), gpu_episode_returns_log, STEPS_TOTAL * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_episode_tf_log.data(), gpu_episode_tf_log, STEPS_TOTAL * sizeof(T), cudaMemcpyDeviceToHost);
        for(TI pos = 0; pos < STEPS_TOTAL; pos++){
            if(cpu_episode_lengths_log[pos] >= (T)0){
                if(cpu_episode_tf_log[pos] > (T)0.5){
                    episode_length_sum_tf += cpu_episode_lengths_log[pos];
                    episode_count_tf++;
                } else {
                    episode_length_sum_student += cpu_episode_lengths_log[pos];
                    episode_count_student++;
                }
            }
        }
        if(ffmpeg_pipe){ pclose(ffmpeg_pipe); ffmpeg_pipe = nullptr; }

        // =================================================================
        // Training (GPU)
        // =================================================================
        T epoch_loss = 0;
        TI loss_count = 0;

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

                auto win_obs = rlt::view_range(device_gpu, gpu_all_observations, window_offset, rlt::tensor::ViewSpec<0, WINDOW_SAMPLES>{});
                using GRU_IMG_SHAPE = rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, IMG_H, IMG_W, IMG_C>;
                auto win_obs_reshaped = rlt::reshape_row_major(device_gpu, win_obs, GRU_IMG_SHAPE{});

                auto win_state = rlt::view_range(device_gpu, gpu_all_state_observations, window_offset, rlt::tensor::ViewSpec<0, WINDOW_SAMPLES>{});
                using GRU_STATE_SHAPE = rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, STATE_OBS_DIM>;
                auto win_state_reshaped = rlt::reshape_row_major(device_gpu, win_state, GRU_STATE_SHAPE{});

                rlt::forward(device_gpu, student_gpu, win_obs_reshaped, win_state_reshaped, student_buffers, rng_gpu);
                cudaDeviceSynchronize();

                auto student_output_tensor = rlt::output(device_gpu, student_gpu);
                auto student_output_matrix = rlt::matrix_view(device_gpu, student_output_tensor);
                auto target_tensor = rlt::view_range(device_gpu, gpu_all_teacher_actions, window_offset, rlt::tensor::ViewSpec<0, WINDOW_SAMPLES>{});
                auto target_matrix = rlt::matrix_view(device_gpu, target_tensor);
                rlt::nn::loss_functions::mse::gradient(device_gpu, student_output_matrix, target_matrix, gpu_d_action_train, (T)0.5);
                cudaDeviceSynchronize();

                if(wi == 0 && pass == 0){
                    rlt::Matrix<rlt::matrix::Specification<T, TI, WINDOW_SAMPLES, ACTION_DIM>> cpu_student_output, cpu_target;
                    rlt::malloc(device, cpu_student_output);
                    rlt::malloc(device, cpu_target);
                    rlt::copy(device_gpu, device, student_output_matrix, cpu_student_output);
                    rlt::copy(device_gpu, device, target_matrix, cpu_target);
                    T batch_loss = 0;
                    for(TI i = 0; i < WINDOW_SAMPLES; i++){
                        for(TI j = 0; j < ACTION_DIM; j++){
                            T diff = rlt::get(cpu_student_output, i, j) - rlt::get(cpu_target, i, j);
                            batch_loss += diff * diff;
                        }
                    }
                    epoch_loss = batch_loss / (WINDOW_SAMPLES * ACTION_DIM);
                    rlt::free(device, cpu_student_output);
                    rlt::free(device, cpu_target);
                }

                auto gpu_d_action_tensor = rlt::to_tensor(device_gpu, gpu_d_action_train);
                using GRU_ACTION_SHAPE = rlt::tensor::Shape<TI, BPTT_STEPS, N_ENVIRONMENTS, ACTION_DIM>;
                auto gpu_d_action_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_action_tensor, GRU_ACTION_SHAPE{});
                rlt::backward(device_gpu, student_gpu, win_obs_reshaped, win_state_reshaped, gpu_d_action_reshaped, student_buffers);
                cudaDeviceSynchronize();
                rlt::step(device_gpu, optimizer_gpu, student_gpu);
                cudaDeviceSynchronize();

                loss_count++;
            }
        }
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
                {
                    constexpr TI GI_BLOCK = 256;
                    constexpr TI GI_GRID = (BATCH_SIZE + GI_BLOCK - 1) / GI_BLOCK;
                    imitation_kernels::compute_gather_indices_kernel<<<GI_GRID, GI_BLOCK, 0, device_gpu.stream>>>(
                        gpu_gather_indices, gpu_episode_start_step_per_row, batch_offset, BATCH_SIZE);
                }
                {
                    int total_elements = BATCH_SIZE * STACKED_OBS_DIM;
                    gather_frames_kernel<<<(total_elements + 255) / 256, 256>>>(
                        rlt::data(gpu_all_observations), gpu_gather_indices, rlt::data(gpu_stacked_batch),
                        OBSERVATION_DIM, IMG_C, FRAME_STACK_N, STACKED_IMG_C, STACKED_OBS_DIM, BATCH_SIZE);
                }
                using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, STACKED_IMG_C>;
                auto gpu_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_stacked_batch, ACTOR_INPUT_SHAPE{});
#else
                auto gpu_obs_batch = rlt::view_range(device_gpu, gpu_all_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, STACKED_IMG_C>;
                auto gpu_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_obs_batch, ACTOR_INPUT_SHAPE{});
#endif
                auto gpu_state_obs_batch = rlt::view_range(device_gpu, gpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_state_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_state_obs_batch, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});

                rlt::forward(device_gpu, student_gpu, gpu_obs_batch_reshaped, gpu_state_obs_batch_reshaped, student_buffers, rng_gpu);
                cudaDeviceSynchronize();

                // MSE loss gradient
                auto student_output_tensor = rlt::output(device_gpu, student_gpu);
                auto student_output_matrix = rlt::matrix_view(device_gpu, student_output_tensor);
                auto target_batch_tensor = rlt::view_range(device_gpu, gpu_all_teacher_actions, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto target_batch = rlt::matrix_view(device_gpu, target_batch_tensor);
                rlt::nn::loss_functions::mse::gradient(device_gpu, student_output_matrix, target_batch, gpu_d_action_train, (T)0.5);
                cudaDeviceSynchronize();

                // Compute loss for logging (sample every N_BATCHES batches)
                if(batch_idx == 0 && pass == 0){
                    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> cpu_student_output, cpu_target;
                    rlt::malloc(device, cpu_student_output);
                    rlt::malloc(device, cpu_target);
                    rlt::copy(device_gpu, device, student_output_matrix, cpu_student_output);
                    rlt::copy(device_gpu, device, target_batch, cpu_target);
                    T batch_loss = 0;
                    for(TI i = 0; i < BATCH_SIZE; i++){
                        for(TI j = 0; j < ACTION_DIM; j++){
                            T diff = rlt::get(cpu_student_output, i, j) - rlt::get(cpu_target, i, j);
                            batch_loss += diff * diff;
                        }
                    }
                    epoch_loss = batch_loss / (BATCH_SIZE * ACTION_DIM);
                    rlt::free(device, cpu_student_output);
                    rlt::free(device, cpu_target);
                }

                // Student backward + Adam step
                auto gpu_d_action_tensor = rlt::to_tensor(device_gpu, gpu_d_action_train);
                auto gpu_d_action_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                rlt::backward(device_gpu, student_gpu, gpu_obs_batch_reshaped, gpu_state_obs_batch_reshaped, gpu_d_action_reshaped, student_buffers);
                cudaDeviceSynchronize();
                rlt::step(device_gpu, optimizer_gpu, student_gpu);
                cudaDeviceSynchronize();

                loss_count++;
            }
        }
#endif

        // Logging
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> training_elapsed = now - training_start;
        std::chrono::duration<T> epoch_elapsed = now - epoch_start;
        T mean_episode_length_tf = episode_count_tf > 0 ? episode_length_sum_tf / episode_count_tf : 0;
        T mean_episode_length_student = episode_count_student > 0 ? episode_length_sum_student / episode_count_student : 0;
        TI episode_count = episode_count_tf + episode_count_student;
        T mean_episode_length = episode_count_student > 0 ? mean_episode_length_student : mean_episode_length_tf;
        T fps = epoch_elapsed.count() > 0 ? static_cast<T>(STEPS_TOTAL) / epoch_elapsed.count() : 0;

        std::cout << (full_teacher_forcing ? "[TF] " : "[TF=" + std::to_string((int)(TEACHER_FORCING_FRACTION * 100)) + "%] ")
                  << "Epoch: " << std::setw(5) << epoch_i
                  << " MSE: " << std::setw(10) << std::setprecision(6) << std::fixed << epoch_loss
                  << " mean_ep_len: " << std::setw(6) << std::setprecision(1) << mean_episode_length
                  << " episodes: " << std::setw(5) << episode_count
                  << " fps: " << std::setw(7) << std::setprecision(0) << fps
                  << " epoch_time: " << std::setw(6) << std::setprecision(1) << epoch_elapsed.count() << "s"
                  << " total: " << std::setw(8) << std::setprecision(1) << training_elapsed.count() << "s"
                  << std::endl;

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
        rlt::set_step(device, device.logger, epoch_i);
        rlt::add_scalar(device, device.logger, "training/mse_loss", epoch_loss);
        if(episode_count_tf > 0){
            rlt::add_scalar(device, device.logger, "training/teacher/episode_length", mean_episode_length_tf);
            rlt::add_scalar(device, device.logger, "training/teacher/episodes", static_cast<T>(episode_count_tf));
        }
        if(episode_count_student > 0){
            rlt::add_scalar(device, device.logger, "training/student/episode_length", mean_episode_length_student);
            rlt::add_scalar(device, device.logger, "training/student/episodes", static_cast<T>(episode_count_student));
        }
        rlt::add_scalar(device, device.logger, "training/fps", fps);
        rlt::add_scalar(device, device.logger, "training/epoch_time_s", epoch_elapsed.count());
        rlt::add_scalar(device, device.logger, "training/total_time_s", training_elapsed.count());
        rlt::add_scalar(device, device.logger, "training/teacher_forcing", full_teacher_forcing ? (T)1 : TEACHER_FORCING_FRACTION);
#endif

        if(epoch_i % CHECKPOINT_CADENCE == 0){
            auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, epoch_end_step);
            static constexpr TI CHECKPOINT_BATCH_SIZE = 1;
            using EVAL_TYPE = typename STUDENT_TYPE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>::template CHANGE_BATCH_SIZE<TI, CHECKPOINT_BATCH_SIZE>;
            EVAL_TYPE eval_student;
            rlt::malloc(device, eval_student);
            rlt::copy(device_gpu, device, student_gpu, eval_student);
            { // binary (tar)
                std::filesystem::path checkpoint_path = step_folder / "checkpoint.tar";
                rlt::persist::backends::tar::Writer writer;
                rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> root_group{"", &writer};
                auto actor_group = rlt::create_group(device, root_group, "actor");
                rlt::set_attribute(device, actor_group, "checkpoint_name", step_folder.string().c_str());
                rlt::save(device, eval_student, actor_group);
                rlt::persist::backends::tar::finalize(device, writer);
                std::ofstream f(checkpoint_path, std::ios::binary);
                f.write(writer.buffer.data(), writer.buffer.size());
            }
            { // code (checkpoint.h)
                auto actor_weights = rlt::save_code(device, eval_student, std::string("rl_tools::checkpoint::actor"), true);
                std::stringstream output_ss;
                output_ss << actor_weights;
                output_ss << "\n" << "namespace rl_tools::checkpoint::meta{";
                output_ss << "\n" << "   " << "char name[] = \"" << step_folder.string() << "\";";
                output_ss << "\n" << "   " << "char commit_hash[] = \"" << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
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
    for(TI env_i = 1; env_i < N_ENVIRONMENTS; env_i++){
        envs[env_i].renderer = nullptr;
        envs[env_i].scene = nullptr;
    }
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::free(device, envs[env_i]);
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
    rlt::free(device_gpu, student_gpu);
    rlt::free(device_gpu, raptor_gpu);
    rlt::free(device_gpu, raptor_buffer_gpu);
    rlt::free(device_gpu, raptor_state_gpu);
    rlt::free(device_gpu, gpu_teacher_obs);
    rlt::free(device_gpu, student_buffers);
    rlt::free(device_gpu, optimizer_gpu);
    rlt::free(device_gpu, gpu_all_observations);
    rlt::free(device_gpu, gpu_all_state_observations);
    rlt::free(device_gpu, gpu_all_teacher_actions);
    rlt::free(device_gpu, gpu_d_action_train);
    rlt::free(device_gpu, gpu_actions_eval);
    cudaFree(gpu_dynamics_arr);
    cudaFree(gpu_params_arr);
    cudaFree(gpu_states_arr);
    cudaFree(gpu_terminated_arr);
    cudaFree(gpu_episode_step_arr);
    cudaFree(gpu_teacher_forcing_arr);
    cudaFree(gpu_episode_return_arr);
    cudaFree(gpu_needs_reset);
    cudaFree(gpu_episode_lengths_log);
    cudaFree(gpu_episode_returns_log);
    cudaFree(gpu_episode_tf_log);
#ifdef USE_GRU_TEMPORAL
    rlt::free(device_gpu, student_state_gpu);
    rlt::free(device_gpu, gpu_rollout_branch_a);
    rlt::free(device_gpu, gpu_rollout_branch_b);
    rlt::free(device_gpu, gpu_rollout_concat);
    rlt::free(device_gpu, gpu_rollout_actions);
#elif defined(USE_FRAME_STACKING)
    rlt::free(device_gpu, gpu_stacked_batch);
    cudaFree(gpu_gather_indices);
    cudaFree(gpu_episode_start_step);
    cudaFree(gpu_episode_start_step_per_row);
#endif

    return 0;
}
