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
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/gru/helper_operations_cuda.h>
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
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cuda.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/operations_generic_mixed_imitation.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mux.h>
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

#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_generic_position_hold.h>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <vector>
#include <numeric>
#include <random>
#include <mutex>
#include <sstream>
#include <metra/metra.h>

namespace rlt = rl_tools;
using namespace rl_tools;
using rlt::is_imitation_sample;
using rlt::use_teacher_action;
using rlt::mix_policy_output_gradients;
using rlt::load;
using rlt::copy;

#include "config.h"

namespace rl_tools::nn::group{
    template<>
    struct CodeName<rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups::Fixed>{
        static constexpr const char* VALUE = "rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups::Fixed";
        static constexpr const char* DEFINITION = "namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups{ struct Fixed; }";
    };
    template<>
    struct CodeName<rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups::Visual>{
        static constexpr const char* VALUE = "rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups::Visual";
        static constexpr const char* DEFINITION = "namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups{ struct Visual; }";
    };
    template<>
    struct CodeName<rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups::Readout>{
        static constexpr const char* VALUE = "rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups::Readout";
        static constexpr const char* DEFINITION = "namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups{ struct Readout; }";
    };
    template<>
    struct CodeName<rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups::Exploration>{
        static constexpr const char* VALUE = "rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups::Exploration";
        static constexpr const char* DEFINITION = "namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed::groups{ struct Exploration; }";
    };
}
namespace training_mixed = rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed;
using namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame::mixed;

using DEVICE_GPU_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_GPU_SPEC>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;

static constexpr TI WORLD_HISTORY_LENGTH = FRAME_STACK_STRIDE * (FRAME_STACK_N - 1) + 1;

static_assert(CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X == CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y && CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X == CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z, "The Hyperdrone World randomizes all mount offset axes with one range");
static_assert(CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X == CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y && CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X == CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z, "The Hyperdrone World randomizes all mount rotation axes with one range");
static_assert(CAMERA_PITCH == static_cast<T>(0), "The Hyperdrone World uses the default forward camera mount");
struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, STATIC_PARAMETERS>{
    static constexpr TI N_CAMERA_AGENTS = 1;
    static constexpr TI INSTANCES_PER_ENVIRONMENT = N_ENVIRONMENTS_PER_SCENE;
    static constexpr TI CAM_WIDTH = training_mixed::CAM_WIDTH;
    static constexpr TI CAM_HEIGHT = training_mixed::CAM_HEIGHT;
    static constexpr TI NUM_PROBES = training_mixed::NUM_PROBES;
    using SHADING = RENDER_SHADING;
    static constexpr bool ENABLE_MOTION_BLUR = RENDER_ENABLE_MOTION_BLUR;
    static constexpr TI MOTION_BLUR_SAMPLES = RENDER_MOTION_BLUR_SAMPLES;
    static constexpr bool ENABLE_ANTI_ALIASING = RENDER_ENABLE_ANTI_ALIASING;
    static constexpr TI ANTI_ALIASING_GRID_SIZE = RENDER_ANTI_ALIASING_GRID_SIZE;
    static constexpr TI HISTORY_LENGTH = WORLD_HISTORY_LENGTH;
    static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = training_mixed::CAMERA_FOV_RANDOMIZATION_RANGE;
    static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X;
    static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X;
    static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = training_mixed::BRIGHTNESS_RANDOMIZATION_RANGE;
    static constexpr T SHUTTER_FRACTION_MIN = RENDER_SHUTTER_FRACTION_MIN;
    static constexpr T SHUTTER_FRACTION_MAX = RENDER_SHUTTER_FRACTION_MAX;
};
using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
static_assert(BASE_WORLD::RENDERER_SPEC::SHADING::PBR_SHADING, "Hyperdrone training must use the High renderer profile");
struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD>{
    static constexpr TI IMAGE_STACK_N = FRAME_STACK_N;
    static constexpr TI IMAGE_STACK_STRIDE = FRAME_STACK_STRIDE;
    static constexpr TI PAD_CHANNELS_TO = 8;
    static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = training_mixed::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE;
    static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = training_mixed::TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE;
    // imitation instances reset together every IMITATION_EPISODE_STEPS: one refresh per imitation episode hands each a fresh setup
    static constexpr TI TARGET_BANK_REFRESH_STEPS = IMITATION_EPISODE_STEPS;
};
using TASK_WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
static constexpr TI NUMBER_OF_ENVIRONMENTS = N_ACTIVE_SCENES;
using MULTI_ENVIRONMENT = rlt::rl::environments::hyperdrone::MultiEnvironment<TASK_WORLD, NUMBER_OF_ENVIRONMENTS>;
using ENVIRONMENT = TASK_WORLD;
static constexpr bool RENDER_MOTION_BLUR_ACTIVE = BASE_WORLD::RENDERER_SPEC::ENABLE_MOTION_BLUR;
static constexpr bool RENDER_ANTI_ALIASING_ACTIVE = BASE_WORLD::RENDERER_SPEC::ENABLE_ANTI_ALIASING;
static_assert(BASE_WORLD::Observation::HEIGHT == CAM_HEIGHT && BASE_WORLD::Observation::WIDTH == CAM_WIDTH && BASE_WORLD::Observation::CHANNELS == IMAGE_CHANNELS, "The World must produce the configured camera frames");
static_assert(TASK_IMG_C == TASK_WORLD::OBSERVATION_CHANNELS, "The pairs address the World's composed channel layout");
static_assert(TASK_WORLD::INSTANCES == N_ENVIRONMENTS_PER_SCENE);

using LOOP_CORE_PARAMETERS = LoopParameters<ENVIRONMENT>;
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximators>;
using PPO_SPEC = typename LOOP_CORE_CONFIG::PPO_SPEC;
using PPO_TYPE = typename LOOP_CORE_CONFIG::PPO_TYPE;
using PPO_BUFFERS_TYPE = typename LOOP_CORE_CONFIG::PPO_BUFFERS_TYPE;
using ACTOR_OPTIMIZER = typename LOOP_CORE_CONFIG::NN::ACTOR_OPTIMIZER;
using CRITIC_OPTIMIZER = typename LOOP_CORE_CONFIG::NN::CRITIC_OPTIMIZER;
using ACTOR_BUFFERS = typename LOOP_CORE_CONFIG::ACTOR_BUFFERS;
using CRITIC_BUFFERS = typename LOOP_CORE_CONFIG::CRITIC_BUFFERS;
using ACTOR_TYPE = typename LOOP_CORE_CONFIG::NN::ACTOR_TYPE;

using CAPABILITY_ROLLOUT = rlt::nn::capability::Forward<true>;
using ROLLOUT_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<CAPABILITY_ROLLOUT>::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>;
using ROLLOUT_ACTOR_BUFFERS = typename ROLLOUT_ACTOR_TYPE::template Buffer<true>;
using ACTOR_CONFIG = typename LOOP_CORE_CONFIG::NN::template Actor<CAPABILITY_ROLLOUT>;
using MATCHING_TYPE = typename ACTOR_CONFIG::MATCHING_TYPE::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>;
using MATCHING_BUFFERS = typename MATCHING_TYPE::template Buffer<true>;
using FEATURE_SHAPE = typename MATCHING_TYPE::OUTPUT_SHAPE;
static constexpr TI FEATURE_DIM = rlt::product(rlt::tensor::PopFront<rlt::tensor::PopFront<FEATURE_SHAPE>>{});
using CHECKPOINT_ACTOR_TYPE = typename ACTOR_CONFIG::EXPORT_MODEL::template CHANGE_BATCH_SIZE<TI, N_EXAMPLES>;
using ROLLOUT_POLICY_STATE = typename ROLLOUT_ACTOR_TYPE::template State<true>;
using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<TYPE_POLICY, MULTI_ENVIRONMENT, ROLLOUT_POLICY_STATE, typename BASE_WORLD::Observation, typename BASE_WORLD::ObservationPrivileged, T, T, EPISODE_STEP_LIMIT, LOOP_CORE_PARAMETERS::PPO_PARAMETERS::TRUNCATE_ON_EACH_ITERATION, true, LOOP_CORE_PARAMETERS::PPO_PARAMETERS::BOOTSTRAP_TRUNCATIONS || LOOP_CORE_PARAMETERS::PPO_PARAMETERS::IGNORE_TERMINATION>;
using ON_POLICY_RUNNER = rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
using ON_POLICY_RUNNER_BUFFER = rlt::rl::components::on_policy_runner::Buffer<ON_POLICY_RUNNER_SPEC>;
using ON_POLICY_RUNNER_DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, LOOP_CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV, true>;
using ON_POLICY_RUNNER_DATASET_TYPE = rlt::rl::components::on_policy_runner::Dataset<ON_POLICY_RUNNER_DATASET_SPEC>;

static constexpr TI STEPS_PER_ENV = LOOP_CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV;
static constexpr TI BATCH_SIZE = LOOP_CORE_PARAMETERS::BATCH_SIZE;
static constexpr TI OBSERVATION_DIM = BASE_WORLD::OBSERVATION_DIM;
static constexpr TI TASK_OBSERVATION_DIM = TASK_WORLD::OBSERVATION_DIM;
static constexpr TI OBS_PRIV_DIM = ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED;
static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
static constexpr TI STEPS_TOTAL = ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL;
static constexpr TI N_EPOCHS = LOOP_CORE_PARAMETERS::PPO_PARAMETERS::N_EPOCHS;
static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
using MIXED_SPEC = mixed_imitation::Specification<TYPE_POLICY, TI, BATCH_SIZE, N_ENVIRONMENTS_PER_SCENE, MIXED_PARAMETERS::IMITATION_ENVIRONMENTS_PER_SCENE, MIXED_PARAMETERS>;
static constexpr TI RL_BATCH_SIZE = MIXED_SPEC::RL_BATCH_SIZE;
static constexpr TI RL_STEPS_TOTAL = N_BATCHES * RL_BATCH_SIZE;
static constexpr TI RL_UPDATE_SAMPLES = RL_STEPS_TOTAL * GRADIENT_ACCUMULATION_ROLLOUTS;
static constexpr TI UPDATE_SAMPLES = STEPS_TOTAL * GRADIENT_ACCUMULATION_ROLLOUTS;
static constexpr T GRADIENT_SCALE = static_cast<T>(BATCH_SIZE) / static_cast<T>(UPDATE_SAMPLES);

static_assert(GRADIENT_ACCUMULATION_ROLLOUTS == 1, "The runner-based rollout uses one update per rollout");
static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");
static_assert(STEPS_TOTAL % BATCH_SIZE == 0, "Every rollout sample must be consumed exactly once");
static_assert(N_EPOCHS == 1, "On-policy accumulation does not reuse rollout samples");
static_assert(N_ENVIRONMENTS <= RNG_GPU::NUM_RNGS, "Each environment needs an independent RNG stream");
static_assert(N_EXAMPLES <= N_ENVIRONMENTS, "The checkpoint examples are rows of the rollout's final step");
static_assert(BATCH_SIZE % N_ENVIRONMENTS == 0, "Every mixed minibatch must cover all scenes with the same IL/RL split");
static_assert(ON_POLICY_RUNNER_SPEC::STEP_LIMIT == EPISODE_STEP_LIMIT);
static_assert(ON_POLICY_RUNNER_SPEC::COLLECT_NEXT_OBSERVATIONS, "Truncation bootstrapping requires next observations");
static_assert(MIXED_SPEC::GROUP_SIZE == TASK_WORLD::INSTANCES, "The IL/RL split is defined per World instance block");

struct TrajectoryStep {
    typename ENVIRONMENT::State state;
    typename ENVIRONMENT::Parameters parameters;
    std::string scene_hash;
    T actions[ENVIRONMENT::ACTION_DIM];
    T reward;
    bool terminated;
};
struct EpisodeRecorder {
    std::vector<TrajectoryStep> current_episode;
};

std::string trajectory_episodes_to_json(DEVICE& device, TASK_WORLD& env, const std::vector<std::vector<TrajectoryStep>>& episodes, T dt){
    if(episodes.empty()){
        return "[]";
    }
    TI max_len = 0;
    for(auto& ep : episodes){
        if(ep.size() > max_len){
            max_len = ep.size();
        }
    }
    std::string json = "[";
    for(TI ep_i = 0; ep_i < episodes.size(); ep_i++){
        auto& episode = episodes[ep_i];
        const auto& parameters = episode.front().parameters;
        std::string parameters_json = rlt::json(device, env.dynamics, parameters.agents[0].dynamics);
        parameters_json = parameters_json.substr(0, parameters_json.size() - 1);
        parameters_json += ", \"visual\": {\"scene_translation\": [" + std::to_string(parameters.scene_translation[0]) + ", " + std::to_string(parameters.scene_translation[1]) + ", " + std::to_string(parameters.scene_translation[2]) + "]"
            + ", \"scene_yaw\": " + std::to_string(rlt::math::atan2(device.math, parameters.scene_yaw_sin, parameters.scene_yaw_cos))
            + ", \"scene_hash\": \"" + episode.front().scene_hash + "\""
            + ", \"cam_width\": " + std::to_string(WORLD_SPEC::CAM_WIDTH)
            + ", \"cam_height\": " + std::to_string(WORLD_SPEC::CAM_HEIGHT)
            + ", \"fov\": " + std::to_string(parameters.agents[0].cameras[0].fov) + "}}";
        json += "{\"parameters\": " + parameters_json + ",\n";
        json += "\"trajectory\": [";
        for(TI step_i = 0; step_i < max_len; step_i++){
            auto& s = (step_i < episode.size()) ? episode[step_i] : episode.back();
            json += "{\"state\":" + rlt::json(device, env.dynamics, parameters.agents[0].dynamics, s.state) + ",";
            json += "\"action\":[";
            for(TI a = 0; a < ENVIRONMENT::ACTION_DIM; a++){
                json += std::to_string(s.actions[a]);
                if(a < ENVIRONMENT::ACTION_DIM - 1){
                    json += ",";
                }
            }
            json += "],";
            json += "\"dt\":" + std::to_string(dt) + ",";
            json += "\"reward\":" + std::to_string(s.reward) + ",";
            bool term = (step_i < episode.size()) ? s.terminated : true;
            json += "\"terminated\":" + (term ? std::string("true") : std::string("false"));
            json += "}";
            if(step_i < max_len - 1){
                json += ",";
            }
        }
        json += "]}";
        if(ep_i < episodes.size() - 1){
            json += ",";
        }
    }
    json += "]";
    return json;
}

static constexpr T EPISODE_END_REASON_NONE = 0;
static constexpr T EPISODE_END_REASON_TERMINATED = 1;
static constexpr T EPISODE_END_REASON_TIME_LIMIT = 2;
static constexpr T EPISODE_END_REASON_SCENE_BOUNDARY = 3;

namespace rl_tools{
    // the teacher's velocity term is the finite difference the actor forms from its frame pairs: body positions FRAME_STACK_STRIDE steps apart, the earlier one clamped to the episode start
    template<typename DEVICE, typename COUNTER_SPEC>
    __global__ void geometric_teacher_kernel(DEVICE device, const typename ENVIRONMENT::State* states, const T* observations, const Tensor<COUNTER_SPEC> episode_step, T* body_position_history, T* actions){
        const TI env_i = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        static_assert(STATE_OBS_DIM == 62 && ACTION_HISTORY_LENGTH == 8);
        const T* obs = observations + env_i * STATE_OBS_DIM;
        const auto& state = states[env_i];
        T rotation[3][3], body_position[3] = {};
        using rl::environments::l2f::quaternion_to_rotation_matrix;
        quaternion_to_rotation_matrix<DEVICE, T>(state.orientation, rotation);
        for(TI axis = 0; axis < 3; axis++){
            for(TI world_axis = 0; world_axis < 3; world_axis++){
                body_position[axis] += rotation[world_axis][axis] * state.position[world_axis];
            }
        }
        const TI step = get(device, episode_step, env_i);
        T* history = body_position_history + env_i * (FRAME_STACK_STRIDE + 1) * 3;
        T* current = history + step % (FRAME_STACK_STRIDE + 1) * 3;
        const T* previous = history + (step >= FRAME_STACK_STRIDE ? step - FRAME_STACK_STRIDE : 0) % (FRAME_STACK_STRIDE + 1) * 3;
        T damped[3];
        for(TI axis = 0; axis < 3; axis++){
            current[axis] = body_position[axis];
            damped[axis] = body_position[axis] + POSITION_HOLD_SPEC::DERIVATIVE_GAIN * (body_position[axis] - previous[axis]);
        }
        const T gravity = POSITION_HOLD_SPEC::GRAVITY;
        T action[4];
        action[0] = obs[6] - static_cast<T>(0.02) * (obs[40] - gravity + static_cast<T>(2) * damped[2]);
        action[1] = static_cast<T>(-1.5) * obs[1] + static_cast<T>(3) / gravity * damped[1];
        action[2] = static_cast<T>(1.5) * obs[0] - static_cast<T>(3) / gravity * damped[0];
        action[3] = -math::atan2(device.math, rotation[1][0], rotation[0][0]);
        for(TI axis = 0; axis < 4; axis++){
            actions[env_i * ACTION_DIM + axis] = math::clamp(device.math, action[axis], static_cast<T>(-1), static_cast<T>(1));
        }
    }

    // the actor's state branch: estimated gravity, bias-corrected gyro (calibration residual at reset,
    // low-pass bias tracking in flight) and CTBR/acceleration history
    template<typename DEVICE, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename STATE_OBS_SPEC, typename RNG>
    __global__ void mixed_state_observation_kernel(DEVICE device, typename ENVIRONMENT::DYNAMICS_ENV dynamics, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<RESET_SPEC> reset_mask, Tensor<STATE_OBS_SPEC> state_observations, RNG rng){
        const TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        auto& rng_state = get(rng.states, 0, env_i);
        auto& params = get_ref(device, parameters, env_i);
        auto& state = get_ref(device, states, env_i);
        const bool reset = get(device, reset_mask, env_i);
        auto state_obs_slice = view(device, state_observations, env_i);
        auto base_observations = matrix_view(device, state_obs_slice);
        for(TI axis = 0; axis < 3; axis++){
            T calibration_error = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, static_cast<T>(0), GYRO_CALIBRATION_RESIDUAL_STD, rng_state);
            if(reset){
                state.bias_estimate[axis] = state.gyro_bias[axis] + calibration_error;
            }
        }
        observe(device, dynamics, params.agents[0].dynamics, state, ACTOR_STATE_OBS{}, base_observations, rng_state);
        if(reset){
            set(base_observations, 0, 40, static_cast<T>(9.81));
        }
        for(TI axis = 0; axis < 3; axis++){
            if(reset){
                state.gyro[axis] = get(base_observations, 0, 3 + axis);
            }
            else{
                state.bias_estimate[axis] += (state.gyro[axis] - state.bias_estimate[axis]) * params.agents[0].dynamics.integration.dt / GYRO_BIAS_TRACKING_TAU;
            }
            set(base_observations, 0, 3 + axis, state.gyro[axis] - state.bias_estimate[axis]);
        }
    }

    template<typename DEVICE, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
    __global__ void mixed_sample_actions_kernel(DEVICE device, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, const Matrix<LOG_STD_SPEC> log_std, Tensor<STEP_ACTIONS_SPEC> step_actions, const T* teacher_actions, TI optimizer_updates, TI step_i, RNG rng){
        const TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        auto& rng_state = get(rng.states, 0, env_i);
        const TI pos = step_i * N_ENVIRONMENTS + env_i;
        T action_log_prob = 0;
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            const T action_mean = get(dataset.actions_mean, pos, action_i);
            const T current_action_log_std = get(log_std, 0, action_i);
            const T action_std = math::exp(device.math, current_action_log_std);
            T action_noisy = random::normal_distribution::sample(device.random, action_mean, action_std, rng_state);
            if(is_imitation_sample(device, MIXED_SPEC{}, env_i)){
                const T noise = (action_noisy - action_mean) / action_std;
                if(use_teacher_action(device, MIXED_SPEC{}, env_i, optimizer_updates)){
                    T student_fraction = 0;
                    if(optimizer_updates > MIXED_PARAMETERS::TEACHER_PURE_UPDATES){
                        student_fraction = static_cast<T>(optimizer_updates - MIXED_PARAMETERS::TEACHER_PURE_UPDATES) / static_cast<T>(MIXED_PARAMETERS::TEACHER_FORCING_UPDATES - MIXED_PARAMETERS::TEACHER_PURE_UPDATES);
                    }
                    const T teacher_action = teacher_actions[pos * ACTION_DIM + action_i] + MIXED_PARAMETERS::TEACHER_ACTION_NOISE_STD * noise;
                    action_noisy = student_fraction * action_mean + (static_cast<T>(1) - student_fraction) * teacher_action;
                }
                else{
                    action_noisy = action_mean;
                }
            }
            action_log_prob += random::normal_distribution::log_prob(device.random, action_mean, current_action_log_std, action_noisy);
            set(dataset.actions, pos, action_i, action_noisy);
            set(device, step_actions, action_noisy, env_i, action_i);
        }
        set(dataset.action_log_probs, pos, 0, action_log_prob);
    }

    // imitation instances fly IMITATION_EPISODE_STEPS steps from an independently sampled pose: force the runner's reset flag after the transition has been recorded so the World re-samples parameters, state and target frame
    template<typename DEVICE, typename RESET_SPEC, typename COUNTER_SPEC>
    __global__ void mixed_imitation_reset_kernel(DEVICE device, Tensor<RESET_SPEC> reset, Tensor<COUNTER_SPEC> episode_step){
        const TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        if(is_imitation_sample(device, MIXED_SPEC{}, env_i) && get(device, episode_step, env_i) >= IMITATION_EPISODE_STEPS){
            set(device, reset, true, env_i);
            set(device, episode_step, (TI)0, env_i);
        }
    }

    template<typename DEVICE, typename PARAMETER_SPEC, typename RESET_SPEC>
    __global__ void mixed_imitation_initialization_kernel(DEVICE device, Tensor<PARAMETER_SPEC> parameters, const Tensor<RESET_SPEC> reset){
        const TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        if(is_imitation_sample(device, MIXED_SPEC{}, env_i) && get(device, reset, env_i)){
            auto& params = get_ref(device, parameters, env_i);
            params.agents[0].dynamics.mdp.init.guidance = 0;
            params.agents[0].dynamics.mdp.init.max_position = IMITATION_POSITION_LIMIT;
            params.agents[0].dynamics.mdp.init.max_angle = IMITATION_ANGLE_LIMIT;
            params.agents[0].dynamics.mdp.init.max_linear_velocity = 0;
            params.agents[0].dynamics.mdp.init.max_angular_velocity = 0;
        }
    }

    // N_PAIRS pairs side by side over the World's composed [stack | target | pad] channels: pair 2 * frame + reversed is [stack frame, target] ([target, stack frame] when reversed)
    __global__ void pair_task_observation_kernel(const float* __restrict__ task_observations, float* __restrict__ combined_out){
        const TI global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(global_idx >= N_ENVIRONMENTS * COMBINED_OBS_DIM){
            return;
        }
        const TI sample = global_idx / COMBINED_OBS_DIM;
        const TI paired_pixel = global_idx % COMBINED_OBS_DIM / COMBINED_IMG_C;
        const TI pair = paired_pixel % (N_PAIRS * CAM_WIDTH) / CAM_WIDTH;
        const TI pixel = paired_pixel / (N_PAIRS * CAM_WIDTH) * CAM_WIDTH + paired_pixel % CAM_WIDTH;
        const TI pair_channel = global_idx % COMBINED_IMG_C;
        const TI slot = pair_channel / IMG_C;
        if(slot >= 2){
            combined_out[global_idx] = 0;
            return;
        }
        const bool target = (slot == 1) != (pair % 2 == 1);
        const TI source = target ? FRAME_STACK_N : pair / 2;
        combined_out[global_idx] = task_observations[sample * TASK_OBSERVATION_DIM + pixel * TASK_IMG_C + source * IMG_C + pair_channel % IMG_C];
    }

    // the trained actor consumes the matching prefix's features; checkpoints carry the prefix in front of it
    template<TI OFFSET, TI LAYER_I = 0, typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE, typename TARGET>
    void copy_layers(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const SOURCE& source, TARGET& target){
        if constexpr(LAYER_I < SOURCE::SPEC::NUM_LAYERS){
            copy(source_device, target_device, nn_models::sequential::layer<LAYER_I>(source), nn_models::sequential::layer<OFFSET + LAYER_I>(target));
            copy_layers<OFFSET, LAYER_I + 1>(source_device, target_device, source, target);
        }
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename MATCHING, typename ACTOR, typename EXPORT_ACTOR>
    void assemble_export_actor(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const MATCHING& matching, const ACTOR& actor, EXPORT_ACTOR& export_actor){
        auto& image_branch = get<0>(export_actor.pipelines);
        copy_layers<0>(source_device, target_device, matching, image_branch);
        copy_layers<MATCHING::SPEC::NUM_LAYERS>(source_device, target_device, get<0>(actor.pipelines), image_branch);
        copy(source_device, target_device, get<1>(actor.pipelines), get<1>(export_actor.pipelines));
        copy(source_device, target_device, actor.head, export_actor.head);
    }
}
using rlt::geometric_teacher_kernel;
using rlt::mixed_state_observation_kernel;
using rlt::mixed_sample_actions_kernel;
using rlt::mixed_imitation_reset_kernel;
using rlt::mixed_imitation_initialization_kernel;
using rlt::pair_task_observation_kernel;

int main(int argc, char** argv){
    TI seed = 0;
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <scene_directory or scene.glb> [seed]" << std::endl;
        return 1;
    }
    if(argc > 3){
        std::cerr << "Usage: <scene_directory | scene.glb> [seed]" << std::endl;
        return 1;
    }
    if(argc > 2){
        seed = std::atoi(argv[2]);
    }

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
    } else {
        scene_paths.resize(N_TOTAL_SCENES, scene_arg);
        std::cout << "Replicating single scene across " << N_TOTAL_SCENES << " renderers: " << scene_arg << std::endl;
    }

    DEVICE_GPU device_gpu;
    auto& device = device_gpu.rendering;
    init(device);
    init(device_gpu);
    device.logger.verbosity = rlt::devices::logging::Verbosity::warning;
    rlt::rendering::datasets::procthor::GLB scene_dataset{{}, scene_paths};
    std::vector<TI> scene_permutation(N_TOTAL_SCENES);
    std::iota(scene_permutation.begin(), scene_permutation.end(), 0);

    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = RL_TOOLS_HYPERDRONE_TARGET_NAME;
    extrack_config.population_variates = "algorithm_environments_accumulation";
    extrack_config.population_values = std::string("mixed_") + std::to_string(N_ENVIRONMENTS) + "_" + std::to_string(GRADIENT_ACCUMULATION_ROLLOUTS);
    init(device, extrack_config, extrack_paths, seed);
    metra::log_raw("hyperdrone/target_frame/mixed/target", "\"" RL_TOOLS_HYPERDRONE_TARGET_NAME "\"");
    metra::log("hyperdrone/target_frame/mixed/seed", static_cast<double>(seed));
    {
        nlohmann::json camera_setup;
        camera_setup["nominal_pitch_degrees"] = CAMERA_PITCH * 180 / rlt::math::PI<T>;
        camera_setup["nominal_fov_degrees"] = CAMERA_FOV;
        camera_setup["image_rectification"] = false;
        camera_setup["gravity_direction_source"] = "mahony";
        camera_setup["imitation_position_limit_m"] = IMITATION_POSITION_LIMIT;
        camera_setup["imitation_angle_limit_rad"] = IMITATION_ANGLE_LIMIT;
        camera_setup["diagnostic_teacher_initialization"] = "normal_at_target";
        camera_setup["diagnostic_same_state_teacher_actions"] = true;
        camera_setup["mahony_kp"] = STATIC_PARAMETERS::STATE_BASE_MAHONY::KP;
        camera_setup["mahony_ki"] = STATIC_PARAMETERS::STATE_BASE_MAHONY::KI;
        camera_setup["gyro_bias_correction"] = "mahony_bias_estimate_with_in_flight_tracking";
        camera_setup["gyro_bias_tracking_tau_s"] = GYRO_BIAS_TRACKING_TAU;
        camera_setup["gyro_startup_calibration"] = "stationary_mean_with_gaussian_residual";
        camera_setup["gyro_calibration_residual_std_rad_s"] = GYRO_CALIBRATION_RESIDUAL_STD;
        camera_setup["fov_randomization_degrees"] = CAMERA_FOV_RANDOMIZATION_RANGE;
        camera_setup["width"] = CAM_WIDTH;
        camera_setup["height"] = CAM_HEIGHT;
        camera_setup["anti_aliasing_grid_size"] = RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : static_cast<TI>(1);
        camera_setup["environment"] = "hyperdrone";
        std::ofstream output(extrack_paths.seed / "camera_setup.json");
        output << camera_setup.dump(2) << '\n';
        output.close();
        rlt::utils::assert_exit(device, output.good(), "Could not save camera setup");
    }

    RNG rng;
    malloc(device, rng);
    init(device, rng, seed);
    RNG reward_log_rng;
    malloc(device, reward_log_rng);
    init(device, reward_log_rng, seed + 0xBADC0DE);

    PPO_TYPE ppo;
    PPO_BUFFERS_TYPE ppo_buffers;
    ON_POLICY_RUNNER_DATASET_TYPE dataset;
    ACTOR_OPTIMIZER actor_optimizer;
    CRITIC_OPTIMIZER critic_optimizer;
    malloc(device, ppo);
    malloc(device, ppo_buffers);
    malloc(device, dataset);
    malloc(device, actor_optimizer);
    malloc(device, critic_optimizer);

    auto* env_storage = new MULTI_ENVIRONMENT{};
    MULTI_ENVIRONMENT& env = *env_storage;
    env.shared.annotation_cache.directory = rlt::rendering::datasets::annotations::default_cache_directory();
    malloc(device_gpu, env);
    typename decltype(scene_dataset)::Corpus scene_corpus;
    rlt::rendering::datasets::procthor::enumerate(device, scene_dataset, scene_corpus);
    if(static_cast<TI>(scene_corpus.references.size()) < N_TOTAL_SCENES){
        std::cerr << "Need at least " << N_TOTAL_SCENES << " GLB scenes, found " << scene_corpus.references.size() << std::endl;
        return 1;
    }
    scene_corpus.references.resize(N_TOTAL_SCENES);
    for(TI member_i = 0; member_i < NUMBER_OF_ENVIRONMENTS; member_i++){
        auto& world = env.environments[member_i];
        world.rng_offset = member_i * N_ENVIRONMENTS_PER_SCENE;
        world.parameters.agents[0].cameras[0].fov = CAMERA_FOV;
        rlt::init(device_gpu, world, env.shared, scene_dataset, scene_corpus, (TI)0, N_TOTAL_SCENES, member_i);
    }
    std::cout << "Loaded " << N_TOTAL_SCENES << " scenes on " << NUMBER_OF_ENVIRONMENTS << " Worlds" << std::endl;

    {
        std::string ui = get_ui(device, env.environments[0].dynamics);
        if(!ui.empty()){
            std::filesystem::create_directories(extrack_paths.seed);
            std::ofstream ui_file(extrack_paths.seed / "ui.esm.js");
            ui_file << ui;
        }
    }

    init(device, ppo, actor_optimizer, critic_optimizer, rng);
    MATCHING_TYPE matching;
    malloc(device, matching);
    init(device, matching, ppo.actor, POSITION_HOLD_SPEC{});
    TI environment_step = 0;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, ACTION_DIM>>> teacher_actions_gpu, teacher_actions;
    malloc(device_gpu, teacher_actions_gpu);
    malloc(device, teacher_actions);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, FRAME_STACK_STRIDE + 1, 3>>> teacher_body_positions;
    malloc(device_gpu, teacher_body_positions);

    RNG_GPU rng_gpu;
    malloc(device_gpu, rng_gpu);
    init(device_gpu, rng_gpu, seed);

    PPO_TYPE ppo_gpu;
    ACTOR_BUFFERS actor_buffers;
    CRITIC_BUFFERS critic_buffers;
    rlt::rl::components::on_policy_runner::ValueState<typename PPO_SPEC::CRITIC_TYPE, ON_POLICY_RUNNER_DATASET_SPEC> critic_states_gae;
    rlt::rl::components::on_policy_runner::ValueBuffer<typename PPO_SPEC::CRITIC_TYPE, ON_POLICY_RUNNER_DATASET_SPEC> critic_buffers_gae;
    ON_POLICY_RUNNER_DATASET_TYPE dataset_gpu;
    malloc(device_gpu, ppo_gpu);
    malloc(device_gpu, actor_buffers);
    malloc(device_gpu, critic_buffers);
    malloc(device_gpu, critic_states_gae);
    malloc(device_gpu, critic_buffers_gae);
    malloc(device_gpu, dataset_gpu);

    ROLLOUT_ACTOR_TYPE rollout_actor_gpu;
    ROLLOUT_ACTOR_BUFFERS rollout_actor_buffers;
    malloc(device_gpu, rollout_actor_gpu);
    malloc(device_gpu, rollout_actor_buffers);
    MATCHING_TYPE matching_gpu;
    MATCHING_BUFFERS matching_buffers;
    malloc(device_gpu, matching_gpu);
    malloc(device_gpu, matching_buffers);
    copy(device, device_gpu, matching, matching_gpu);

    ACTOR_OPTIMIZER actor_optimizer_gpu;
    CRITIC_OPTIMIZER critic_optimizer_gpu;
    malloc(device_gpu, actor_optimizer_gpu);
    malloc(device_gpu, critic_optimizer_gpu);
    copy(device, device_gpu, ppo, ppo_gpu);
    copy(device, device_gpu, actor_optimizer, actor_optimizer_gpu);
    copy(device, device_gpu, critic_optimizer, critic_optimizer_gpu);
    copy(device, device_gpu, ppo.actor, rollout_actor_gpu);
    rlt::reset_optimizer_state(device_gpu, actor_optimizer_gpu, ppo_gpu.actor);
    rlt::reset_optimizer_state(device_gpu, critic_optimizer_gpu, ppo_gpu.critic);

    // the runner owns the persistent instance state and episode accounting, its buffer the transient
    // next-state/action/reward storage; the driver adds the mixed target's observer and teacher data
    ON_POLICY_RUNNER gpu_runner;
    ON_POLICY_RUNNER_BUFFER gpu_runner_buffer;
    malloc(device_gpu, gpu_runner);
    malloc(device_gpu, gpu_runner_buffer);
    auto& gpu_env_parameters = gpu_runner.env_parameters;
    auto& gpu_env_states = gpu_runner.states;

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, TASK_OBSERVATION_DIM>>> gpu_task_observations;
    malloc(device_gpu, gpu_task_observations);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, COMBINED_OBS_DIM>>> gpu_rollout_combined;
    malloc(device_gpu, gpu_rollout_combined);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, FEATURE_DIM>>> gpu_all_features;
    malloc(device_gpu, gpu_all_features);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, STATE_OBS_DIM>>> gpu_all_state_observations;
    malloc(device_gpu, gpu_all_state_observations);

    rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ACTION_DIM>> gpu_actions_eval;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_actions_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_d_action_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, OBS_PRIV_DIM>> gpu_critic_obs;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, 1>> gpu_d_critic_output;
    malloc(device_gpu, gpu_actions_eval);
    malloc(device_gpu, gpu_actions_train);
    malloc(device_gpu, gpu_d_action_train);
    malloc(device_gpu, gpu_critic_obs);
    malloc(device_gpu, gpu_d_critic_output);

    EpisodeRecorder episode_recorders[TRAJECTORY_NUM_ENVS];
    std::vector<std::vector<TrajectoryStep>> completed_episodes;
    T simulation_dt = static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY);
    typename ENVIRONMENT::State reward_log_state;
    typename ENVIRONMENT::State reward_log_next_state;
    typename ENVIRONMENT::Parameters reward_log_parameters;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_DIM>> reward_log_action;
    malloc(device, reward_log_action);

    std::vector<typename ENVIRONMENT::State> trajectory_states(STEPS_PER_ENV * TRAJECTORY_NUM_ENVS);
    std::vector<typename ENVIRONMENT::Parameters> trajectory_parameters(STEPS_PER_ENV * TRAJECTORY_NUM_ENVS);

    std::cout << "Starting single-use mixed imitation/on-policy training (visual L2F position hold, Hyperdrone CUDA)" << std::endl;
    std::cout << "  N_ENVIRONMENTS:   " << N_ENVIRONMENTS << std::endl;
    std::cout << "  STEPS_PER_ENV:    " << STEPS_PER_ENV << std::endl;
    std::cout << "  SCENE_SET_STEPS:  " << STEPS_PER_ENV * ROLLOUTS_PER_SCENE_SET << std::endl;
    std::cout << "  STEPS_TOTAL:      " << STEPS_TOTAL << std::endl;
    std::cout << "  BATCH_SIZE:       " << BATCH_SIZE << std::endl;
    std::cout << "  MIXED BATCH:      " << MIXED_SPEC::IMITATION_BATCH_SIZE << " IL + " << RL_BATCH_SIZE << " RL" << std::endl;
    std::cout << "  TEACHER WARMUP:   " << MIXED_PARAMETERS::TEACHER_FORCING_UPDATES << " optimizer updates (IL environments only)" << std::endl;
    {
        using BALANCING_MODE = typename MIXED_PARAMETERS::BALANCING_MODE::MODE;
        std::cout << "  IL BALANCING:     " << (rlt::mode::is<BALANCING_MODE, mixed_imitation::FixedNormRatio> ? "output-gradient norm ratio" : "fixed weight") << std::endl;
        auto log_config = [&](const char* key, T value){
            add_scalar(device, device.logger, (std::string("mixed/config/") + key).c_str(), value);
            metra::log(std::string("hyperdrone/target_frame/mixed/config/") + key, static_cast<double>(value));
        };
        log_config("nominal_camera_pitch_degrees", CAMERA_PITCH * 180 / rlt::math::PI<T>);
        log_config("nominal_camera_fov_degrees", CAMERA_FOV);
        log_config("gyro_bias_tracking_tau_s", GYRO_BIAS_TRACKING_TAU);
        log_config("imitation_share", static_cast<T>(MIXED_SPEC::IMITATION_BATCH_SIZE) / BATCH_SIZE);
        log_config("teacher_forcing_updates", MIXED_PARAMETERS::TEACHER_FORCING_UPDATES);
        log_config("imitation_weight", MIXED_PARAMETERS::IMITATION_WEIGHT);
        log_config("target_norm_ratio", MIXED_PARAMETERS::OUTPUT_GRADIENT_NORM_RATIO);
        log_config("fixed_norm_ratio", rlt::mode::is<BALANCING_MODE, mixed_imitation::FixedNormRatio> ? 1 : 0);
        log_config("max_imitation_weight", MIXED_PARAMETERS::MAX_IMITATION_WEIGHT);
        log_config("norm_epsilon", MIXED_PARAMETERS::NORM_EPSILON);
    }
    std::cout << "  N_BATCHES:        " << N_BATCHES << std::endl;
    std::cout << "  ACCUMULATION:     " << GRADIENT_ACCUMULATION_ROLLOUTS << " rollouts, " << UPDATE_SAMPLES << " samples/update" << std::endl;
    std::cout << "  COMBINED_IMG_C:   " << COMBINED_IMG_C << " (logical " << COMBINED_IMG_C_LOGICAL << ")" << std::endl;
    std::cout << "  STATE_OBS_DIM:    " << STATE_OBS_DIM << std::endl;
    std::cout << "  OBS_PRIV_DIM:     " << OBS_PRIV_DIM << std::endl;
    std::cout << "  PPO STEP_LIMIT:   " << LOOP_CORE_PARAMETERS::STEP_LIMIT << std::endl;
    std::cout << "  RENDER_AA:        " << (RENDER_ANTI_ALIASING_ACTIVE ? "on" : "off") << " grid=" << (RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : (TI)1) << std::endl;
    std::cout << "  RENDER_MOTION_BLUR: " << (RENDER_MOTION_BLUR_ACTIVE ? "on" : "off") << " samples=" << (RENDER_MOTION_BLUR_ACTIVE ? RENDER_MOTION_BLUR_SAMPLES : (TI)1) << " shutter=[" << RENDER_SHUTTER_FRACTION_MIN << ", " << RENDER_SHUTTER_FRACTION_MAX << "]" << std::endl;

    auto training_start = std::chrono::high_resolution_clock::now();
    static constexpr TI N_PPO_STEPS = LOOP_CORE_PARAMETERS::STEP_LIMIT;
    TI accumulated_samples = 0;
    TI optimizer_updates = 0;

    constexpr TI BLOCKSIZE = 32;
    constexpr TI N_BLOCKS = (N_ENVIRONMENTS + BLOCKSIZE - 1) / BLOCKSIZE;
    dim3 grid(N_BLOCKS);
    dim3 block(BLOCKSIZE);
    rlt::devices::cuda::TAG<DEVICE_GPU, true> tag_device{};

    static constexpr TI VIDEO_ENVS_PER_SCENE = N_ENVIRONMENTS_PER_SCENE - MIXED_SPEC::IMITATION_PER_GROUP;
    static_assert(VIDEO_SCENE_COLS > 0 && VIDEO_ENV_COLS > 0);
    static constexpr TI VIDEO_ENV_ROWS = (VIDEO_ENVS_PER_SCENE + VIDEO_ENV_COLS - 1) / VIDEO_ENV_COLS;
    static constexpr TI VIDEO_SCENE_ROWS = (N_ACTIVE_SCENES + VIDEO_SCENE_COLS - 1) / VIDEO_SCENE_COLS;
    static constexpr TI VIDEO_WIDTH = VIDEO_SCENE_COLS * VIDEO_ENV_COLS * CAM_WIDTH * 2;
    static constexpr TI VIDEO_HEIGHT = VIDEO_SCENE_ROWS * VIDEO_ENV_ROWS * CAM_HEIGHT;
    static_assert(VIDEO_WIDTH % 2 == 0 && VIDEO_HEIGHT % 2 == 0, "YUV420 video dimensions must be even");
    FILE* video_pipe = nullptr;
    bool video_enabled = VIDEO_SAVE_INTERVAL_SCENE_SETS > 0;
    std::filesystem::path video_path;
    TI video_step = 0;
    std::vector<T> video_observations, video_targets;
    std::vector<std::uint8_t> video_frame;
    const auto previous_sigpipe_handler = std::signal(SIGPIPE, SIG_IGN);
    auto close_video_pipe = [&](){
        if(video_pipe == nullptr){
            return;
        }
        const int status = pclose(video_pipe);
        video_pipe = nullptr;
        if(status == 0 && video_enabled){
            link_latest_artifact(device, get_latest_folder(device, extrack_paths), video_path, video_path.parent_path(), video_step);
            std::cout << "Training video saved: " << std::filesystem::absolute(video_path) << std::endl;
        }
        else{
            std::cerr << "Video recording disabled after FFmpeg failed for " << video_path << std::endl;
            video_enabled = false;
        }
    };

    std::vector<T> episode_returns(N_ENVIRONMENTS, 0);
    std::vector<TI> episode_lengths(N_ENVIRONMENTS, 0);
    std::vector<TI> episode_end_reasons(N_ENVIRONMENTS, 0);
    for(TI ppo_step_i = 0; ppo_step_i < N_PPO_STEPS; ppo_step_i++){
        auto step_start = std::chrono::high_resolution_clock::now();
        zero_gradient(device_gpu, ppo_gpu.actor);
        zero_gradient(device_gpu, ppo_gpu.critic);
        accumulated_samples = 0;
        set_step(device, device.logger, environment_step);
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
            std::shuffle(scene_permutation.begin(), scene_permutation.end(), scene_rng);
            for(TI member_i = 0; member_i < NUMBER_OF_ENVIRONMENTS; member_i++){
                select_scene(device_gpu, env.environments[member_i], scene_permutation[member_i]);
            }
            if(ppo_step_i == 0){
                init(device_gpu, gpu_runner, env, rng_gpu);
            }
            else{
                reset(device_gpu, gpu_runner, env, rng_gpu);
            }
            if constexpr(VIDEO_SAVE_INTERVAL_SCENE_SETS > 0){
                if(video_enabled && scene_set_i % VIDEO_SAVE_INTERVAL_SCENE_SETS == 0){
                    video_step = environment_step + N_ENVIRONMENTS * STEPS_PER_ENV * std::min(ROLLOUTS_PER_SCENE_SET, N_PPO_STEPS - ppo_step_i);
                    video_path = get_step_folder(device, extrack_config, extrack_paths, video_step) / "video.mp4";
                    const std::string command = "ffmpeg -y -hide_banner -loglevel warning -f rawvideo -pixel_format rgb24 -video_size " + std::to_string(VIDEO_WIDTH) + "x" + std::to_string(VIDEO_HEIGHT) + " -framerate " + std::to_string(SIMULATION_FREQUENCY) + " -i - -an -c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast -movflags +faststart " + rlt::utils::extrack::detail::shell_quote(video_path.string());
                    video_pipe = popen(command.c_str(), "w");
                    if(video_pipe == nullptr){
                        std::cerr << "Video recording disabled: failed to start FFmpeg for " << video_path << std::endl;
                        video_enabled = false;
                    }
                    else{
                        video_observations.resize(N_ENVIRONMENTS * OBSERVATION_DIM);
                        video_targets.resize(N_ENVIRONMENTS * OBSERVATION_DIM);
                        video_frame.resize(VIDEO_WIDTH * VIDEO_HEIGHT * IMAGE_CHANNELS);
                        std::cout << "Recording PPO training episodes (target | current): " << std::filesystem::absolute(video_path) << std::endl;
                    }
                }
            }
        }

        prologue(device_gpu, dataset_gpu, gpu_runner, env, rng_gpu);
        for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
            evaluate_values(device_gpu, dataset_gpu, ppo_gpu.critic, critic_states_gae, critic_buffers_gae, rng_gpu, step_i);
            auto state_observations = view_range(device_gpu, gpu_all_state_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            mixed_state_observation_kernel<<<grid, block, 0, device_gpu.stream>>>(tag_device, env.environments[0].dynamics, gpu_env_parameters, gpu_env_states, gpu_runner.reset, state_observations, rng_gpu);
            check_status(device_gpu);
            geometric_teacher_kernel<<<grid, block, 0, device_gpu.stream>>>(tag_device, data(gpu_env_states), data(state_observations), gpu_runner.episode_step, data(teacher_body_positions), data(teacher_actions_gpu) + step_i * N_ENVIRONMENTS * ACTION_DIM);
            check_status(device_gpu);
            if(video_pipe != nullptr){
                const T* obs_ptr = data(dataset_gpu.all_observations) + step_i * N_ENVIRONMENTS * OBSERVATION_DIM;
                cudaMemcpyAsync(video_observations.data(), obs_ptr, video_observations.size() * sizeof(T), cudaMemcpyDeviceToHost, device_gpu.stream);
                for(TI member_i = 0; member_i < NUMBER_OF_ENVIRONMENTS; member_i++){
                    constexpr TI M = TASK_WORLD::INSTANCES;
                    cudaMemcpyAsync(video_targets.data() + member_i * M * OBSERVATION_DIM, data(env.environments[member_i].target_frames), M * OBSERVATION_DIM * sizeof(T), cudaMemcpyDeviceToHost, device_gpu.stream);
                }
                cudaStreamSynchronize(device_gpu.stream);
                check_status(device_gpu);
                for(TI scene_i = 0; scene_i < N_ACTIVE_SCENES; scene_i++){
                    for(TI local_env = 0; local_env < VIDEO_ENVS_PER_SCENE; local_env++){
                        const TI env_i = scene_i * N_ENVIRONMENTS_PER_SCENE + MIXED_SPEC::IMITATION_PER_GROUP + local_env;
                        const TI cell_x = (scene_i % VIDEO_SCENE_COLS * VIDEO_ENV_COLS + local_env % VIDEO_ENV_COLS) * CAM_WIDTH * 2;
                        const TI cell_y = (scene_i / VIDEO_SCENE_COLS * VIDEO_ENV_ROWS + local_env / VIDEO_ENV_COLS) * CAM_HEIGHT;
                        for(TI y = 0; y < CAM_HEIGHT; y++){
                            for(TI x = 0; x < CAM_WIDTH; x++){
                                const TI source = env_i * OBSERVATION_DIM + (y * CAM_WIDTH + x) * IMAGE_CHANNELS;
                                const TI destination = ((cell_y + y) * VIDEO_WIDTH + cell_x + x) * IMAGE_CHANNELS;
                                for(TI channel = 0; channel < IMAGE_CHANNELS; channel++){
                                    video_frame[destination + channel] = static_cast<std::uint8_t>(std::clamp(video_targets[source + channel] * 255.0f, 0.0f, 255.0f));
                                    video_frame[destination + CAM_WIDTH * IMAGE_CHANNELS + channel] = static_cast<std::uint8_t>(std::clamp(video_observations[source + channel] * 255.0f, 0.0f, 255.0f));
                                }
                            }
                        }
                    }
                }
                if(std::fwrite(video_frame.data(), 1, video_frame.size(), video_pipe) != video_frame.size()){
                    video_enabled = false;
                    close_video_pipe();
                }
            }

            observe(device_gpu, env, gpu_env_parameters, gpu_env_states, typename TASK_WORLD::Observation{}, gpu_task_observations, rng_gpu);
            {
                constexpr TI total_elements = N_ENVIRONMENTS * COMBINED_OBS_DIM;
                pair_task_observation_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(data(gpu_task_observations), data(gpu_rollout_combined));
                check_status(device_gpu);
            }
            {
                using ROLLOUT_IMG_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, IMG_H, IMG_W, COMBINED_IMG_C>;
                auto step_combined_reshaped = reshape_row_major(device_gpu, gpu_rollout_combined, ROLLOUT_IMG_SHAPE{});
                auto step_features = view_range(device_gpu, gpu_all_features, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                auto step_features_reshaped = reshape_row_major(device_gpu, step_features, rlt::tensor::Replace<FEATURE_SHAPE, N_ENVIRONMENTS, 1>{});
                evaluate(device_gpu, matching_gpu, step_combined_reshaped, step_features_reshaped, matching_buffers, rng_gpu);
                using ROLLOUT_STATE_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, STATE_OBS_DIM>;
                auto step_state_reshaped = reshape_row_major(device_gpu, state_observations, ROLLOUT_STATE_SHAPE{});
                auto inputs = rlt::nn_models::parallel::pack_inputs(step_features_reshaped, step_state_reshaped);
                auto gpu_actions_eval_tensor = to_tensor(device_gpu, gpu_actions_eval);
                auto gpu_actions_eval_reshaped = reshape_row_major(device_gpu, gpu_actions_eval_tensor, rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, ACTION_DIM>{});
                evaluate(device_gpu, rollout_actor_gpu, inputs, gpu_actions_eval_reshaped, rollout_actor_buffers, rng_gpu);
            }
            {
                auto actions_mean_view = view(device_gpu, dataset_gpu.actions_mean, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
                copy(device_gpu, device_gpu, gpu_actions_eval, actions_mean_view);
            }
            if(log_reward_components_this_step && step_i == STEPS_PER_ENV - 1){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(&reward_log_state, data(gpu_env_states), sizeof(typename ENVIRONMENT::State), cudaMemcpyDeviceToHost);
                cudaMemcpy(&reward_log_parameters, data(gpu_env_parameters), sizeof(typename ENVIRONMENT::Parameters), cudaMemcpyDeviceToHost);
            }
            {
                auto& last_layer_gpu = ppo_gpu.actor.head;
                auto log_std_gpu = matrix_view(device_gpu, last_layer_gpu.log_std.parameters);
                mixed_sample_actions_kernel<<<grid, block, 0, device_gpu.stream>>>(tag_device, dataset_gpu, log_std_gpu, gpu_runner_buffer.actions, data(teacher_actions_gpu), optimizer_updates, step_i, rng_gpu);
                check_status(device_gpu);
            }
            if(save_extrack_step){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(trajectory_parameters.data() + step_i * TRAJECTORY_NUM_ENVS, data(gpu_env_parameters), TRAJECTORY_NUM_ENVS * sizeof(typename ENVIRONMENT::Parameters), cudaMemcpyDeviceToHost);
            }
            // the runner's epilogue with the mixed target's imitation episodes: after the transition is recorded, expired imitation instances are forced to reset and their initial state distribution is narrowed before the World samples the new episode
            step(device_gpu, env, gpu_env_parameters, gpu_env_states, gpu_runner_buffer.actions, gpu_runner_buffer.next_states, rng_gpu);
            reward(device_gpu, env, gpu_env_parameters, gpu_env_states, gpu_runner_buffer.actions, gpu_runner_buffer.next_states, gpu_runner_buffer.rewards, rng_gpu);
            terminated(device_gpu, env, gpu_env_parameters, gpu_runner_buffer.next_states, gpu_runner_buffer.terminated, rng_gpu);
            copy(device_gpu, device_gpu, gpu_runner_buffer.next_states, gpu_env_states);
            record_transition(device_gpu, dataset_gpu, gpu_runner, gpu_runner_buffer, step_i);
            observe(device_gpu, env, gpu_env_parameters, gpu_env_states, typename ON_POLICY_RUNNER_SPEC::OBSERVATION_PRIVILEGED{}, gpu_runner_buffer.next_observations_privileged, rng_gpu);
            mixed_imitation_reset_kernel<<<grid, block, 0, device_gpu.stream>>>(tag_device, gpu_runner.reset, gpu_runner.episode_step);
            check_status(device_gpu);
            sample_initial_parameters(device_gpu, env, gpu_env_parameters, gpu_runner.reset, rng_gpu);
            mixed_imitation_initialization_kernel<<<grid, block, 0, device_gpu.stream>>>(tag_device, gpu_env_parameters, gpu_runner.reset);
            check_status(device_gpu);
            sample_initial_state(device_gpu, env, gpu_env_parameters, gpu_env_states, gpu_runner.reset, rng_gpu);
            observe_row(device_gpu, dataset_gpu, gpu_runner, env, step_i + 1, rng_gpu);
            evaluate_bootstrap_values(device_gpu, dataset_gpu, gpu_runner_buffer.next_observations_privileged, ppo_gpu.critic, critic_states_gae, critic_buffers_gae, rng_gpu, step_i);
            if(log_reward_components_this_step && step_i == STEPS_PER_ENV - 1){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(&reward_log_next_state, data(gpu_runner_buffer.next_states), sizeof(typename ENVIRONMENT::State), cudaMemcpyDeviceToHost);
            }
            if(save_extrack_step){
                cudaStreamSynchronize(device_gpu.stream);
                std::vector<typename ENVIRONMENT::State> tmp_states(TRAJECTORY_NUM_ENVS);
                cudaMemcpy(tmp_states.data(), data(gpu_runner_buffer.next_states), TRAJECTORY_NUM_ENVS * sizeof(typename ENVIRONMENT::State), cudaMemcpyDeviceToHost);
                for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                    trajectory_states[step_i * TRAJECTORY_NUM_ENVS + env_i] = tmp_states[env_i];
                }
            }
        }

        environment_step += N_ENVIRONMENTS * STEPS_PER_ENV;
        set_step(device, device.logger, environment_step);
        if(scene_set_end || ppo_step_i + 1 == N_PPO_STEPS){
            close_video_pipe();
        }

        evaluate_rollout_values(device_gpu, dataset_gpu, ppo_gpu.critic, critic_buffers_gae, rng_gpu, PPO_SPEC::COLLECTION_MODE{});

        cudaDeviceSynchronize();
        {
            auto gpu_scalar = matrix_view(device_gpu, dataset_gpu.scalar_data);
            auto cpu_scalar = matrix_view(device, dataset.scalar_data);
            copy(device_gpu, device, gpu_scalar, cpu_scalar);
        }
        {
            auto gpu_obs_priv = matrix_view(device_gpu, dataset_gpu.all_observations_privileged);
            auto cpu_obs_priv = matrix_view(device, dataset.all_observations_privileged);
            copy(device_gpu, device, gpu_obs_priv, cpu_obs_priv);
        }
        if(log_reward_components_this_step){
            static constexpr TI REWARD_LOG_POS = (STEPS_PER_ENV - 1) * N_ENVIRONMENTS;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                set(reward_log_action, 0, action_i, get(dataset.actions, REWARD_LOG_POS, action_i));
            }
            log_reward(device, env.environments[0].dynamics, reward_log_parameters.agents[0].dynamics, reward_log_state, reward_log_action, reward_log_next_state, reward_log_rng);
        }
        {
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
            std::array<T, 2> subset_length_sum{};
            std::array<TI, 2> subset_episode_count{};
            for(TI pos = 0; pos < STEPS_TOTAL; pos++){
                T reward_value = get(dataset.rewards, pos, 0);
                reward_sum += reward_value;
                reward_sq_sum += reward_value * reward_value;
                bool terminated_event = get(dataset.terminated, pos, 0) > (T)0.5;
                bool done_event = get(dataset.truncated, pos, 0) > (T)0.5;
                if(terminated_event){
                    rollout_terminated_count++;
                }
                if(done_event){
                    rollout_done_count++;
                    if(!terminated_event){
                        rollout_truncated_count++;
                    }
                }
                const TI env_i = pos % N_ENVIRONMENTS;
                // imitation episodes are fixed-length teacher/student flights, not task episodes
                if(is_imitation_sample(device, MIXED_SPEC{}, env_i)){
                    continue;
                }
                T ep_len = get(dataset.reset, pos, 0) > (T)0.5 && episode_lengths[env_i] > 0 ? static_cast<T>(episode_lengths[env_i]) : (T)-1;
                if(ep_len >= (T)0){
                    const TI subset = is_imitation_sample(device, MIXED_SPEC{}, pos) ? 0 : 1;
                    subset_length_sum[subset] += ep_len;
                    subset_episode_count[subset]++;
                    add_scalar(device, device.logger, "episode/length", ep_len, 100);
                    T ep_return = episode_returns[env_i];
                    add_scalar(device, device.logger, "episode/return", ep_return, 100);
                    length_sum += ep_len;
                    length_sq_sum += ep_len * ep_len;
                    return_sum += ep_return;
                    return_sq_sum += ep_return * ep_return;
                    TI ep_reason = episode_end_reasons[env_i] != 0 ? episode_end_reasons[env_i] : static_cast<TI>(EPISODE_END_REASON_SCENE_BOUNDARY);
                    if(ep_reason == static_cast<TI>(EPISODE_END_REASON_TERMINATED)){
                        episode_end_terminated_count++;
                        length_sum_terminated += ep_len;
                        length_sum_task += ep_len;
                        add_scalar(device, device.logger, "episode/length/terminated", ep_len, 100);
                    } else if(ep_reason == static_cast<TI>(EPISODE_END_REASON_TIME_LIMIT)){
                        episode_end_time_limit_count++;
                        length_sum_time_limit += ep_len;
                        length_sum_task += ep_len;
                        add_scalar(device, device.logger, "episode/length/time_limit", ep_len, 100);
                    } else if(ep_reason == static_cast<TI>(EPISODE_END_REASON_SCENE_BOUNDARY)){
                        episode_end_scene_boundary_count++;
                        length_sum_scene_boundary += ep_len;
                        add_scalar(device, device.logger, "episode/length/scene_boundary", ep_len, 100);
                    }
                    count++;
                    episode_returns[env_i] = 0;
                    episode_lengths[env_i] = 0;
                    episode_end_reasons[env_i] = 0;
                }
                episode_returns[env_i] += reward_value;
                episode_lengths[env_i]++;
                if(done_event){
                    episode_end_reasons[env_i] = static_cast<TI>(terminated_event ? EPISODE_END_REASON_TERMINATED : EPISODE_END_REASON_TIME_LIMIT);
                }
            }
            rollout_reward_mean = reward_sum / static_cast<T>(STEPS_TOTAL);
            rollout_reward_std = rlt::math::sqrt(device.math, rlt::math::max(device.math, (T)0, reward_sq_sum / static_cast<T>(STEPS_TOTAL) - rollout_reward_mean * rollout_reward_mean));
            rollout_terminated_share = rollout_done_count > 0 ? static_cast<T>(rollout_terminated_count) / static_cast<T>(rollout_done_count) : (T)0;
            for(TI subset = 0; subset < 2; subset++){
                if(subset_episode_count[subset] > 0){
                    const T mean = subset_length_sum[subset] / subset_episode_count[subset];
                    const char* key = subset == 0 ? "mixed/il_episode_length" : "mixed/rl_episode_length";
                    add_scalar(device, device.logger, key, mean);
                    std::cout << "  " << key << ": " << mean << std::endl;
                }
            }
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
                add_scalar(device, device.logger, "training/episode_length", rollout_episode_length_mean);
                add_scalar(device, device.logger, "training/episode_length/mean", rollout_episode_length_mean);
                add_scalar(device, device.logger, "training/episode_length/std", rollout_episode_length_std);
                add_scalar(device, device.logger, "training/return/mean", rollout_return_mean);
                add_scalar(device, device.logger, "training/return/std", rollout_return_std);
                add_scalar(device, device.logger, "training/episodes", static_cast<T>(rollout_episode_count));
            }
            TI episode_end_task_count = episode_end_terminated_count + episode_end_time_limit_count;
            TI episode_end_count = episode_end_task_count + episode_end_scene_boundary_count;
            rlt::utils::assert_exit(device, episode_end_count == count, "Every completed episode must have an end reason");
            add_scalar(device, device.logger, "training/episode_end/terminated", static_cast<T>(episode_end_terminated_count));
            add_scalar(device, device.logger, "training/episode_end/time_limit", static_cast<T>(episode_end_time_limit_count));
            add_scalar(device, device.logger, "training/episode_end/scene_boundary", static_cast<T>(episode_end_scene_boundary_count));
            add_scalar(device, device.logger, "training/episode_end/task", static_cast<T>(episode_end_task_count));
            add_scalar(device, device.logger, "training/time_limit_episodes", static_cast<T>(episode_end_time_limit_count));
            add_scalar(device, device.logger, "training/scene_boundary_resets", static_cast<T>(episode_end_scene_boundary_count));
            add_scalar(device, device.logger, "training/task_episodes", static_cast<T>(episode_end_task_count));
            if(episode_end_terminated_count > 0){
                add_scalar(device, device.logger, "training/episode_length/terminated", length_sum_terminated / static_cast<T>(episode_end_terminated_count));
            }
            if(episode_end_time_limit_count > 0){
                add_scalar(device, device.logger, "training/episode_length/time_limit", length_sum_time_limit / static_cast<T>(episode_end_time_limit_count));
            }
            if(episode_end_scene_boundary_count > 0){
                add_scalar(device, device.logger, "training/episode_length/scene_boundary", length_sum_scene_boundary / static_cast<T>(episode_end_scene_boundary_count));
            }
            if(episode_end_task_count > 0){
                T task_episode_count = static_cast<T>(episode_end_task_count);
                T termination_rate_task = static_cast<T>(episode_end_terminated_count) / task_episode_count;
                add_scalar(device, device.logger, "training/episode_length/task", length_sum_task / task_episode_count);
                add_scalar(device, device.logger, "training/episode_length/excluding_scene_boundary", length_sum_task / task_episode_count);
                add_scalar(device, device.logger, "training/termination_rate/task_episodes", termination_rate_task);
                add_scalar(device, device.logger, "training/termination_rate/excluding_scene_boundary", termination_rate_task);
                add_scalar(device, device.logger, "training/time_limit_rate/task_episodes", static_cast<T>(episode_end_time_limit_count) / task_episode_count);
            }
            if(episode_end_count > 0){
                T all_episode_end_count = static_cast<T>(episode_end_count);
                add_scalar(device, device.logger, "training/termination_rate/all_episode_ends", static_cast<T>(episode_end_terminated_count) / all_episode_end_count);
                add_scalar(device, device.logger, "training/scene_boundary_rate/all_episode_ends", static_cast<T>(episode_end_scene_boundary_count) / all_episode_end_count);
            }
            add_scalar(device, device.logger, "training/reward/mean", rollout_reward_mean);
            add_scalar(device, device.logger, "training/reward/std", rollout_reward_std);
            add_scalar(device, device.logger, "training/terminated_share", rollout_terminated_share);
            add_scalar(device, device.logger, "training/terminated_episodes", static_cast<T>(rollout_terminated_count));
            add_scalar(device, device.logger, "training/truncated_episodes", static_cast<T>(rollout_truncated_count));
            add_scalar(device, device.logger, "training/complete_episodes", static_cast<T>(rollout_done_count));
        }

        if(save_extrack_step){
            for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                episode_recorders[env_i].current_episode.clear();
                for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
                    TI pos = step_i * N_ENVIRONMENTS + env_i;
                    TrajectoryStep ts;
                    ts.state = trajectory_states[step_i * TRAJECTORY_NUM_ENVS + env_i];
                    ts.parameters = trajectory_parameters[step_i * TRAJECTORY_NUM_ENVS + env_i];
                    const auto& world = env.environments[env_i / N_ENVIRONMENTS_PER_SCENE];
                    ts.scene_hash = world.slots[world.active_slot].metadata.content_hash;
                    for(TI a = 0; a < ACTION_DIM; a++){
                        ts.actions[a] = get(dataset.actions, pos, a);
                    }
                    ts.reward = get(dataset.rewards, pos, 0);
                    ts.terminated = (get(dataset.terminated, pos, 0) > (T)0.5);
                    episode_recorders[env_i].current_episode.push_back(ts);
                    if(ts.terminated || get(dataset.truncated, pos, 0) > (T)0.5){
                        completed_episodes.push_back(std::move(episode_recorders[env_i].current_episode));
                        episode_recorders[env_i].current_episode.clear();
                        if(completed_episodes.size() > TRAJECTORY_MAX_EPISODES){
                            completed_episodes.erase(completed_episodes.begin());
                        }
                    }
                }
            }
            if(!completed_episodes.empty()){
                auto step_folder = get_step_folder(device, extrack_config, extrack_paths, environment_step);
                std::string trajectories_json = trajectory_episodes_to_json(device, env.environments[0], completed_episodes, simulation_dt);
                std::vector<uint8_t> compressed;
                if(rlt::compress_zlib(trajectories_json, compressed)){
                    std::filesystem::path trajectories_path = step_folder / "trajectories.json.gz";
                    std::ofstream f(trajectories_path, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
                    f.close();
                    auto latest_folder = get_latest_folder(device, extrack_paths);
                    link_latest_artifact(device, latest_folder, trajectories_path, step_folder, environment_step);
                }
                std::cout << "  Saved " << completed_episodes.size() << " trajectory episodes to " << step_folder << std::endl;
                completed_episodes.clear();
            }
        }

        rlt::estimate_generalized_advantages(device, dataset, dataset.bootstrap_values, typename PPO_TYPE::SPEC::PARAMETERS{});

        {
            auto& last_layer_gpu = ppo_gpu.actor.head;
            auto& last_layer_cpu = ppo.actor.head;
            copy(device_gpu, device, last_layer_gpu.log_std.parameters, last_layer_cpu.log_std.parameters);
            copy(device_gpu, device, last_layer_gpu.log_std.gradient, last_layer_cpu.log_std.gradient);
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
        TI imitation_update_samples = 0;
        T imitation_mse_sum = 0;
        copy(device_gpu, device, teacher_actions_gpu, teacher_actions);

        for(TI epoch_i = 0; epoch_i < N_EPOCHS; epoch_i++){
            TI batch_order[N_BATCHES];
            for(TI i = 0; i < N_BATCHES; i++){
                batch_order[i] = i;
            }
            for(TI i = N_BATCHES - 1; i > 0; i--){
                TI j = rlt::random::uniform_int_distribution(device.random, (TI)0, i, rng);
                std::swap(batch_order[i], batch_order[j]);
            }

            for(TI batch_idx = 0; batch_idx < N_BATCHES; batch_idx++){
                TI batch_i = batch_order[batch_idx];
                TI batch_offset = batch_i * BATCH_SIZE;

                auto batch_features = view_range(device_gpu, gpu_all_features, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto batch_state_obs = view_range(device_gpu, gpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                using TRAIN_STATE_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>;
                auto batch_features_reshaped = reshape_row_major(device_gpu, batch_features, rlt::tensor::Replace<FEATURE_SHAPE, BATCH_SIZE, 1>{});
                auto batch_state_reshaped = reshape_row_major(device_gpu, batch_state_obs, TRAIN_STATE_SHAPE{});
                auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(batch_features_reshaped, batch_state_reshaped);

                auto gpu_actions_train_tensor = to_tensor(device_gpu, gpu_actions_train);
                auto gpu_actions_train_reshaped = reshape_row_major(device_gpu, gpu_actions_train_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                forward(device_gpu, ppo_gpu.actor, fwd_inputs, gpu_actions_train_reshaped, actor_buffers, rng_gpu);
                cudaDeviceSynchronize();

                copy(device_gpu, device, gpu_actions_train, ppo_buffers.current_batch_actions);
                auto& last_layer_gpu = ppo_gpu.actor.head;
                auto& last_layer_cpu = ppo.actor.head;
                copy(device_gpu, device, last_layer_gpu.log_std.parameters, last_layer_cpu.log_std.parameters);
                copy(device_gpu, device, last_layer_gpu.log_std.gradient, last_layer_cpu.log_std.gradient);

                auto batch_actions = view(device, dataset.actions, rlt::matrix::ViewSpec<BATCH_SIZE, ACTION_DIM>(), batch_offset, 0);
                auto batch_action_log_probs = view(device, dataset.action_log_probs, rlt::matrix::ViewSpec<BATCH_SIZE, 1>(), batch_offset, 0);
                auto batch_advantages = view(device, dataset.advantages, rlt::matrix::ViewSpec<BATCH_SIZE, 1>(), batch_offset, 0);
                auto batch_target_values = view(device, dataset.target_values, rlt::matrix::ViewSpec<BATCH_SIZE, 1>(), batch_offset, 0);

                T advantage_mean = 0, advantage_std = 0;
                for(TI i = 0; i < BATCH_SIZE; i++){
                    if(is_imitation_sample(device, MIXED_SPEC{}, i)){
                        continue;
                    }
                    T adv = get(batch_advantages, i, 0);
                    advantage_mean += adv;
                    advantage_std += adv * adv;
                }
                advantage_mean /= RL_BATCH_SIZE;
                advantage_std /= RL_BATCH_SIZE;
                advantage_std = rlt::math::sqrt(device.math, rlt::math::max(device.math, (T)0, advantage_std - advantage_mean * advantage_mean));
                ppo_advantage_mean_sum += advantage_mean;
                ppo_advantage_std_sum += advantage_std;

                T batch_log_std_gradient[ACTION_DIM] = {};
                for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
                    if(is_imitation_sample(device, MIXED_SPEC{}, batch_step_i)){
                        imitation_update_samples++;
                        continue;
                    }
                    T action_log_prob = 0;
                    T action_entropy = 0;
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        T current_action = get(ppo_buffers.current_batch_actions, batch_step_i, action_i);
                        T rollout_action = get(batch_actions, batch_step_i, action_i);
                        T current_action_log_std = get(device, last_layer_cpu.log_std.parameters, action_i);
                        action_log_prob += rlt::random::normal_distribution::log_prob(device.random, current_action, current_action_log_std, rollout_action);
                        action_entropy += current_action_log_std + rlt::math::log(device.math, static_cast<T>(2) * rlt::math::PI<T>) / static_cast<T>(2) + static_cast<T>(0.5);
                        set(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, rlt::random::normal_distribution::d_log_prob_d_mean(device.random, current_action, current_action_log_std, rollout_action));
                        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                            T d_entropy_loss_d_current_action_log_std = -(T)1/RL_UPDATE_SAMPLES * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT;
                            rlt::increment(device, last_layer_cpu.log_std.gradient, d_entropy_loss_d_current_action_log_std, action_i);
                            batch_log_std_gradient[action_i] += d_entropy_loss_d_current_action_log_std;
                            set(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i, rlt::random::normal_distribution::d_log_prob_d_log_std(device.random, current_action, current_action_log_std, rollout_action));
                        }
                    }
                    T rollout_action_log_prob = get(batch_action_log_probs, batch_step_i, 0);
                    T advantage = get(batch_advantages, batch_step_i, 0);
                    if(PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE){
                        advantage = (advantage - advantage_mean) / (advantage_std + PPO_SPEC::PARAMETERS::ADVANTAGE_EPSILON);
                    }
                    T log_ratio = action_log_prob - rollout_action_log_prob;
                    T ratio = rlt::math::exp(device.math, log_ratio);
                    T clipped_ratio = rlt::math::clamp(device.math, ratio, 1 - PPO_SPEC::PARAMETERS::EPSILON_CLIP, 1 + PPO_SPEC::PARAMETERS::EPSILON_CLIP);
                    bool clipped = ratio != clipped_ratio;
                    ppo_actor_loss_sum += -action_log_prob * advantage;
                    ppo_entropy_sum += action_entropy;
                    ppo_approx_kl_sum += (ratio - (T)1) - log_ratio;
                    ppo_ratio_sum += ratio;
                    ppo_clipped_samples += clipped ? 1 : 0;
                    ppo_update_samples++;
                    T d_loss_d_action_log_prob = -advantage / static_cast<T>(RL_UPDATE_SAMPLES);
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        rlt::multiply(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, d_loss_d_action_log_prob);
                        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                            T current_d = get(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i);
                            rlt::increment(device, last_layer_cpu.log_std.gradient, d_loss_d_action_log_prob * current_d, action_i);
                            batch_log_std_gradient[action_i] += d_loss_d_action_log_prob * current_d;
                        }
                    }
                }
                {
                    auto batch_teacher_actions = view_range(device, teacher_actions, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                    auto batch_teacher_matrix = matrix_view(device, batch_teacher_actions);
                    mixed_imitation::Metrics<MIXED_SPEC> metrics;
                    mix_policy_output_gradients(device, MIXED_SPEC{}, ppo_buffers.current_batch_actions, batch_teacher_matrix, ppo_buffers.d_action_log_prob_d_action, metrics, GRADIENT_SCALE, MIXED_PARAMETERS::BALANCING_MODE{});
                    imitation_mse_sum += metrics.imitation_mse;
                    T log_std_squared_norm = 0;
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        log_std_squared_norm += batch_log_std_gradient[action_i] * batch_log_std_gradient[action_i];
                    }
                    TI gradient_batch = ppo_step_i * N_BATCHES + batch_idx;
                    set_step(device, device.logger, gradient_batch);
                    auto log_mixed = [&](const char* key, T value){
                        rlt::utils::assert_exit(device, rlt::math::is_finite(device.math, value), "Non-finite mixed-learning metric");
                        add_scalar(device, device.logger, std::string("mixed/") + key, value);
                        if(gradient_batch % (20 * N_BATCHES) == 0){
                            metra::log(std::string("hyperdrone/target_frame/mixed/") + key, static_cast<double>(value));
                        }
                    };
                    log_mixed("gradient_batch", gradient_batch);
                    log_mixed("environment_step", environment_step);
                    log_mixed("teacher_forcing", optimizer_updates < MIXED_PARAMETERS::TEACHER_FORCING_UPDATES ? 1 : 0);
                    log_mixed("imitation_mse", metrics.imitation_mse);
                    log_mixed("rl_output_gradient_norm", metrics.rl_output_gradient_norm);
                    log_mixed("imitation_output_gradient_norm", metrics.imitation_output_gradient_norm);
                    log_mixed("weighted_imitation_output_gradient_norm", metrics.weighted_imitation_output_gradient_norm);
                    log_mixed("rl_output_gradient_rms", metrics.rl_output_gradient_norm / rlt::math::sqrt(device.math, static_cast<T>(RL_BATCH_SIZE * ACTION_DIM)));
                    log_mixed("imitation_output_gradient_rms", metrics.imitation_output_gradient_norm / rlt::math::sqrt(device.math, static_cast<T>(MIXED_SPEC::IMITATION_BATCH_SIZE * ACTION_DIM)));
                    log_mixed("rl_log_std_gradient_norm", rlt::math::sqrt(device.math, log_std_squared_norm));
                    log_mixed("imitation_weight", metrics.imitation_weight);
                    log_mixed("output_gradient_norm_ratio", metrics.output_gradient_norm_ratio);
                    log_mixed("ratio_valid", metrics.ratio_valid ? 1 : 0);
                    log_mixed("balancing_fallback", metrics.balancing_fallback ? 1 : 0);
                    log_mixed("weight_clamped", metrics.weight_clamped ? 1 : 0);
                    set_step(device, device.logger, environment_step);
                }
                copy(device, device_gpu, last_layer_cpu.log_std.parameters, last_layer_gpu.log_std.parameters);
                copy(device, device_gpu, last_layer_cpu.log_std.gradient, last_layer_gpu.log_std.gradient);
                copy(device, device_gpu, ppo_buffers.d_action_log_prob_d_action, gpu_d_action_train);
                auto gpu_d_action_tensor = to_tensor(device_gpu, gpu_d_action_train);
                auto gpu_d_action_reshaped = reshape_row_major(device_gpu, gpu_d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(batch_features_reshaped, batch_state_reshaped);

                backward(device_gpu, ppo_gpu.actor, bwd_inputs, gpu_d_action_reshaped, actor_buffers, TRAINING_MODE{});

                cudaDeviceSynchronize();

                {
                    auto batch_obs_priv = view_range(device, dataset.all_observations_privileged, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                    auto batch_obs_priv_matrix = matrix_view(device, batch_obs_priv);
                    copy(device, device_gpu, batch_obs_priv_matrix, gpu_critic_obs);
                    auto gpu_critic_obs_tensor = to_tensor(device_gpu, gpu_critic_obs);
                    using OBS_PRIV_SHAPE = typename ON_POLICY_RUNNER_DATASET_TYPE::OBS_PRIV_SHAPE;
                    using CRITIC_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, BATCH_SIZE>, (TI)1>;
                    auto gpu_critic_obs_reshaped = reshape_row_major(device_gpu, gpu_critic_obs_tensor, CRITIC_INPUT_SHAPE{});
                    forward(device_gpu, ppo_gpu.critic, gpu_critic_obs_reshaped, critic_buffers, rng_gpu);
                    cudaDeviceSynchronize();
                    {
                        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, 1>> cpu_critic_output, cpu_d_critic;
                        malloc(device, cpu_critic_output);
                        malloc(device, cpu_d_critic);
                        auto critic_output_tensor = rlt::output(device_gpu, ppo_gpu.critic);
                        auto critic_output_matrix = matrix_view(device_gpu, critic_output_tensor);
                        copy(device_gpu, device, critic_output_matrix, cpu_critic_output);
                        T critic_loss = 0;
                        for(TI row_i = 0; row_i < BATCH_SIZE; row_i++){
                            if(is_imitation_sample(device, MIXED_SPEC{}, row_i)){
                                set(cpu_d_critic, row_i, 0, (T)0);
                            }
                            else{
                                T error = get(cpu_critic_output, row_i, 0) - get(batch_target_values, row_i, 0);
                                critic_loss += error * error / static_cast<T>(RL_BATCH_SIZE);
                                set(cpu_d_critic, row_i, 0, error / static_cast<T>(RL_UPDATE_SAMPLES));
                            }
                        }
                        ppo_critic_loss_sum += critic_loss;
                        ppo_critic_batches++;
                        copy(device, device_gpu, cpu_d_critic, gpu_d_critic_output);
                        free(device, cpu_critic_output);
                        free(device, cpu_d_critic);
                    }
                    auto gpu_d_critic_tensor = to_tensor(device_gpu, gpu_d_critic_output);
                    auto gpu_d_critic_reshaped = reshape_row_major(device_gpu, gpu_d_critic_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 1>{});
                    backward(device_gpu, ppo_gpu.critic, gpu_critic_obs_reshaped, gpu_d_critic_reshaped, critic_buffers);
                    cudaDeviceSynchronize();
                }
                ppo_update_batches++;
            }
        }

        rlt::utils::assert_exit(device, ppo_update_samples == RL_STEPS_TOTAL, "Every RL rollout sample must be used exactly once");
        rlt::utils::assert_exit(device, imitation_update_samples == STEPS_TOTAL - RL_STEPS_TOTAL, "Every IL rollout sample must be used exactly once");
        accumulated_samples += ppo_update_samples + imitation_update_samples;
        rlt::utils::assert_exit(device, accumulated_samples == UPDATE_SAMPLES, "Incorrect accumulated update size");
        step(device_gpu, actor_optimizer_gpu, ppo_gpu.actor, TRAINING_MODE{});
        step(device_gpu, critic_optimizer_gpu, ppo_gpu.critic);
        cudaDeviceSynchronize();
        optimizer_updates++;

        copy(device_gpu, device_gpu, ppo_gpu.actor, rollout_actor_gpu);
        copy(device_gpu, device, ppo_gpu.actor.head.log_std.parameters, ppo.actor.head.log_std.parameters);
        copy(device_gpu, device, actor_optimizer_gpu, actor_optimizer);
        copy(device_gpu, device, critic_optimizer_gpu, critic_optimizer);
        rlt::utils::assert_exit(device, get(device, actor_optimizer.age, 0) == optimizer_updates + 1, "Actor updated before accumulation completed");
        rlt::utils::assert_exit(device, get(device, critic_optimizer.age, 0) == optimizer_updates + 1, "Critic updated before accumulation completed");
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> training_elapsed = now - training_start;
        std::chrono::duration<T> step_elapsed = now - step_start;
        T sps_lifetime = environment_step / training_elapsed.count();
        T sps_current = N_ENVIRONMENTS * STEPS_PER_ENV / step_elapsed.count();
        std::cout << "PPO step " << std::setw(5) << ppo_step_i
                  << "  env_step " << std::setw(10) << environment_step
                  << "  elapsed " << std::setw(7) << std::setprecision(3) << training_elapsed.count() << "s"
                  << "  (sps lifetime " << std::setw(6) << std::setprecision(0) << std::fixed << sps_lifetime
                  << ", current " << std::setw(6) << std::setprecision(0) << sps_current << ")" << std::defaultfloat << std::endl;
        std::cout << "  optimizer updates: " << optimizer_updates << "  accumulated samples: " << accumulated_samples << "/" << UPDATE_SAMPLES << std::endl;
        std::cout << "  mixed samples: " << imitation_update_samples << " IL + " << ppo_update_samples << " RL; imitation MSE: " << std::setprecision(6) << imitation_mse_sum / N_BATCHES << std::endl;

        add_scalar(device, device.logger, "ppo/step", ppo_step_i);
        add_scalar(device, device.logger, "training/optimizer_updates", optimizer_updates);
        add_scalar(device, device.logger, "training/accumulated_samples", accumulated_samples);
        add_scalar(device, device.logger, "training/update_samples", UPDATE_SAMPLES);
        add_scalar(device, device.logger, "training/sample_uses", N_EPOCHS);
        const std::string metra_prefix = "hyperdrone/target_frame/mixed/";
        if(optimizer_updates % 20 == 0){
            metra::log(metra_prefix + "episode_length", static_cast<double>(rollout_episode_length_mean));
            metra::log(metra_prefix + "optimizer_updates", static_cast<double>(optimizer_updates));
            metra::log(metra_prefix + "steps_per_second", static_cast<double>(sps_current));
        }
        add_scalar(device, device.logger, "ppo/actor_learning_rate", get(device, actor_optimizer.parameters, 0).alpha);
        add_scalar(device, device.logger, "ppo/critic_learning_rate", get(device, critic_optimizer.parameters, 0).alpha);
        if(ppo_update_samples > 0){
            T inv_samples = static_cast<T>(1) / static_cast<T>(ppo_update_samples);
            T approx_kl = ppo_approx_kl_sum * inv_samples;
            add_scalar(device, device.logger, "ppo/actor_loss", ppo_actor_loss_sum * inv_samples);
            add_scalar(device, device.logger, "ppo/entropy", ppo_entropy_sum * inv_samples);
            add_scalar(device, device.logger, "ppo/approx_kl", approx_kl);
            add_scalar(device, device.logger, "ppo/policy_kl", approx_kl);
            add_scalar(device, device.logger, "ppo/clip_fraction", static_cast<T>(ppo_clipped_samples) * inv_samples);
            add_scalar(device, device.logger, "ppo/ratio_mean", ppo_ratio_sum * inv_samples);
        }
        if(ppo_update_batches > 0){
            T inv_batches = static_cast<T>(1) / static_cast<T>(ppo_update_batches);
            add_scalar(device, device.logger, "ppo/advantage/mean", ppo_advantage_mean_sum * inv_batches);
            add_scalar(device, device.logger, "ppo/advantage/std", ppo_advantage_std_sum * inv_batches);
        }
        if(ppo_critic_batches > 0){
            add_scalar(device, device.logger, "ppo/critic_loss", ppo_critic_loss_sum / static_cast<T>(ppo_critic_batches));
        }
        add_scalar(device, device.logger, "steps_per_second", sps_current);
        add_scalar(device, device.logger, "timing/steps_per_second", sps_current);
        add_scalar(device, device.logger, "timing/steps_per_second_lifetime", sps_lifetime);
        add_scalar(device, device.logger, "timing/step_time_s", step_elapsed.count());
        add_scalar(device, device.logger, "timing/total_time_s", training_elapsed.count());
        add_scalar(device, device.logger, "rendering/anti_aliasing_grid_size", RENDER_ANTI_ALIASING_ACTIVE ? static_cast<T>(RENDER_ANTI_ALIASING_GRID_SIZE) : static_cast<T>(1));
        add_scalar(device, device.logger, "rendering/motion_blur_samples", RENDER_MOTION_BLUR_ACTIVE ? static_cast<T>(RENDER_MOTION_BLUR_SAMPLES) : static_cast<T>(1));
        if constexpr(RENDER_MOTION_BLUR_ACTIVE){
            add_scalar(device, device.logger, "rendering/shutter_fraction_min", RENDER_SHUTTER_FRACTION_MIN);
            add_scalar(device, device.logger, "rendering/shutter_fraction_max", RENDER_SHUTTER_FRACTION_MAX);
        }
        add_scalar(device, device.logger, "rendering/target_frame_roll_pitch_randomization_range", TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE);
        add_scalar(device, device.logger, "rendering/target_frame_brightness_mismatch_range", TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE);
        add_scalar(device, device.logger, "rendering/camera_mount_offset_randomization_range_x", CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X);
        add_scalar(device, device.logger, "rendering/camera_mount_offset_randomization_range_y", CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y);
        add_scalar(device, device.logger, "rendering/camera_mount_offset_randomization_range_z", CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z);
        add_scalar(device, device.logger, "rendering/camera_mount_rotation_randomization_range_x", CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X);
        add_scalar(device, device.logger, "rendering/camera_mount_rotation_randomization_range_y", CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y);
        add_scalar(device, device.logger, "rendering/camera_mount_rotation_randomization_range_z", CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z);

        const bool training_complete = ppo_step_i + 1 == N_PPO_STEPS;
        save_extrack_step = save_extrack_step || training_complete;
        if(save_extrack_step){
            auto step_folder = get_step_folder(device, extrack_config, extrack_paths, environment_step);
            auto latest_folder = get_latest_folder(device, extrack_paths);
            std::filesystem::create_directories(step_folder);

            CHECKPOINT_ACTOR_TYPE eval_actor;
            malloc(device, eval_actor);
            rlt::assemble_export_actor(device_gpu, device, matching_gpu, ppo_gpu.actor, eval_actor);

            char fov_buf[32];
            std::snprintf(fov_buf, sizeof(fov_buf), "%.6g", (double)CAMERA_FOV);
            std::string state_obs_string = rlt::string(device, env.environments[0].dynamics, ACTOR_STATE_OBS{});
            const auto gyro_observation = state_obs_string.find("AngularVelocity");
            if(gyro_observation == std::string::npos){
                throw std::runtime_error("Missing gyro observation metadata");
            }
            state_obs_string.replace(gyro_observation, std::string("AngularVelocity").size(), "GyroscopeMahonyBiasCorrected");
            std::string image_obs_string = std::string("CameraRGBStackedWithTarget(") + fov_buf + ", "
                + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ", "
                + std::to_string(FRAME_STACK_STRIDE) + ", " + std::to_string(FRAME_STACK_N) + ")";
            std::string obs_string = image_obs_string + ", " + state_obs_string;
            std::string rendering_string = std::string("{\"imitation_position_limit_m\": ") + (std::to_string(IMITATION_POSITION_LIMIT) + ", \"imitation_angle_limit_rad\": " + std::to_string(IMITATION_ANGLE_LIMIT)) + std::string(", \"gyro_bias_correction\": \"mahony_bias_estimate_with_in_flight_tracking\", \"gyro_startup_calibration\": \"stationary_mean_with_gaussian_residual\", \"gyro_calibration_residual_std_rad_s\": ") + std::to_string(GYRO_CALIBRATION_RESIDUAL_STD) + ", \"gyro_bias_tracking_tau_s\": " + std::to_string(GYRO_BIAS_TRACKING_TAU) + ", \"image_rectification\": false, \"anti_aliasing\": " + (RENDER_ANTI_ALIASING_ACTIVE ? "true" : "false")
                + ", \"anti_aliasing_grid_size\": " + std::to_string(RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : (TI)1)
                + ", \"motion_blur\": " + (RENDER_MOTION_BLUR_ACTIVE ? "true" : "false")
                + ", \"motion_blur_samples\": " + std::to_string(RENDER_MOTION_BLUR_ACTIVE ? RENDER_MOTION_BLUR_SAMPLES : (TI)1)
                + ", \"shutter_fraction_min\": " + std::to_string(RENDER_SHUTTER_FRACTION_MIN)
                + ", \"shutter_fraction_max\": " + std::to_string(RENDER_SHUTTER_FRACTION_MAX)
                + ", \"target_frame_roll_pitch_randomization_range\": " + std::to_string(TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE)
                + ", \"target_frame_brightness_mismatch_range\": " + std::to_string(TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE)
                + ", \"camera_mount_offset_randomization_range\": [" + std::to_string(CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X)
                + ", " + std::to_string(CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y)
                + ", " + std::to_string(CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z) + "]"
                + ", \"camera_mount_rotation_randomization_range\": [" + std::to_string(CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X)
                + ", " + std::to_string(CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y)
                + ", " + std::to_string(CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z) + "]}";
            std::string meta = "{\"environment\": {\"name\": \"hyperdrone\", \"observation\": \"" + obs_string + "\", \"output\": \"ActionMean\", \"rendering\": " + rendering_string + "}}";

            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, N_EXAMPLES, IMG_H, IMG_W, COMBINED_IMG_C>, true>> example_input_0_image;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, N_EXAMPLES, STATE_OBS_DIM>, true>> example_input_1_state;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_EXAMPLES, ACTION_DIM>, true>> example_output;
            malloc(device, example_input_0_image);
            malloc(device, example_input_1_state);
            malloc(device, example_output);

            static constexpr TI EXAMPLE_ROW_OFFSET = STEPS_TOTAL - N_EXAMPLES;
            {
                auto src_combined = view_range(device_gpu, gpu_rollout_combined, N_ENVIRONMENTS - N_EXAMPLES, rlt::tensor::ViewSpec<0, N_EXAMPLES>{});
                auto src_state = view_range(device_gpu, gpu_all_state_observations, EXAMPLE_ROW_OFFSET, rlt::tensor::ViewSpec<0, N_EXAMPLES>{});
                auto dst_image_2d = reshape_row_major(device, example_input_0_image, rlt::tensor::Shape<TI, N_EXAMPLES, COMBINED_OBS_DIM>{});
                auto dst_state_2d = reshape_row_major(device, example_input_1_state, rlt::tensor::Shape<TI, N_EXAMPLES, STATE_OBS_DIM>{});
                copy(device_gpu, device, src_combined, dst_image_2d);
                copy(device_gpu, device, src_state, dst_state_2d);
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
                typename rlt::utils::typing::remove_reference_t<decltype(get<0>(eval_actor.pipelines))>::template Buffer<true> buffer_0;
                typename rlt::utils::typing::remove_reference_t<decltype(get<1>(eval_actor.pipelines))>::template Buffer<true> buffer_1;
                malloc(device, branch_0_out);
                malloc(device, branch_1_out);
                malloc(device, concat_out);
                malloc(device, buffer_0);
                malloc(device, buffer_1);
                rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
                auto image_eval_view = rlt::view_memory<rlt::tensor::Shape<TI, N_EXAMPLES, IMG_H, IMG_W, COMBINED_IMG_C>>(device, example_input_0_image);
                auto state_eval_view = rlt::view_memory<rlt::tensor::Shape<TI, N_EXAMPLES, STATE_OBS_DIM>>(device, example_input_1_state);
                evaluate(device, get<0>(eval_actor.pipelines), image_eval_view, branch_0_out, buffer_0, rng, eval_mode);
                evaluate(device, get<1>(eval_actor.pipelines), state_eval_view, branch_1_out, buffer_1, rng, eval_mode);
                auto concat_0 = view_range(device, concat_out, (TI)0, rlt::tensor::ViewSpec<1, BRANCH_0_DIM>{});
                auto concat_1 = view_range(device, concat_out, (TI)BRANCH_0_DIM, rlt::tensor::ViewSpec<1, BRANCH_1_DIM>{});
                copy(device, device, branch_0_out, concat_0);
                copy(device, device, branch_1_out, concat_1);
                typename decltype(eval_actor.head)::template Buffer<true> head_buffer;
                malloc(device, head_buffer);
                evaluate(device, eval_actor.head, concat_out, example_output, head_buffer, rng, eval_mode);
                free(device, head_buffer);
                free(device, branch_0_out);
                free(device, branch_1_out);
                free(device, concat_out);
                free(device, buffer_0);
                free(device, buffer_1);
            }
#ifdef RL_TOOLS_HYPERDRONE_MIXED_SMALL
            {
                auto features = view_range(device_gpu, gpu_all_features, (STEPS_PER_ENV - 1) * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                auto state = view_range(device_gpu, gpu_all_state_observations, (STEPS_PER_ENV - 1) * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                auto inputs = rlt::nn_models::parallel::pack_inputs(reshape_row_major(device_gpu, features, rlt::tensor::Replace<FEATURE_SHAPE, N_ENVIRONMENTS, 1>{}), reshape_row_major(device_gpu, state, rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, STATE_OBS_DIM>{}));
                auto actions = reshape_row_major(device_gpu, to_tensor(device_gpu, gpu_actions_eval), rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, ACTION_DIM>{});
                evaluate(device_gpu, rollout_actor_gpu, inputs, actions, rollout_actor_buffers, rng_gpu);
                rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ACTION_DIM>> trained_actions;
                malloc(device, trained_actions);
                copy(device_gpu, device, gpu_actions_eval, trained_actions);
                T max_difference = 0;
                for(TI example_i = 0; example_i < N_EXAMPLES; example_i++){
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        max_difference = rlt::math::max(device.math, max_difference, rlt::math::abs(device.math, get(trained_actions, N_ENVIRONMENTS - N_EXAMPLES + example_i, action_i) - get(device, example_output, example_i, action_i)));
                    }
                }
                free(device, trained_actions);
                std::cout << "Exported actor vs trained actor on cached features: max action difference " << max_difference << std::endl;
                rlt::utils::assert_exit(device, max_difference < static_cast<T>(1e-2), "The exported actor differs from the trained actor on the cached matching features");
            }
#endif

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
                auto example_output_canonical = reshape_row_major(device, example_output, rlt::tensor::Shape<TI, 1, N_EXAMPLES, ACTION_DIM>{});
                rlt::save(device, example_output_canonical, outputs_group, "0");
                rlt::persist::backends::tar::finalize(device, writer);
                std::ofstream f(checkpoint_path, std::ios::binary);
                f.write(writer.buffer.data(), writer.buffer.size());
                f.close();
                link_latest_artifact(device, latest_folder, checkpoint_path, step_folder, environment_step);
            }
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
            auto save_hdf5 = [&](auto batch_size_tag){
                static constexpr TI EXAMPLE_BATCH_SIZE = decltype(batch_size_tag)::value;
                using SIZED_EVAL_ACTOR_TYPE = typename CHECKPOINT_ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, EXAMPLE_BATCH_SIZE>;
                SIZED_EVAL_ACTOR_TYPE sized_eval_actor;
                malloc(device, sized_eval_actor);
                rlt::assemble_export_actor(device_gpu, device, matching_gpu, ppo_gpu.actor, sized_eval_actor);
                std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
                std::filesystem::path checkpoint_path = step_folder / (std::string("checkpoint_") + std::to_string(EXAMPLE_BATCH_SIZE) + "examples.h5");
                rlt::persist::backends::hdf5::File root_file(checkpoint_path.string(), rlt::persist::backends::hdf5::Mode::WRITE);
                auto actor_group = rlt::create_group(device, root_file, "actor");
                rlt::set_attribute(device, actor_group, "checkpoint_name", step_folder.string().c_str());
                rlt::set_attribute(device, actor_group, "meta", meta.c_str());
                rlt::save(device, sized_eval_actor, actor_group);
                auto example_group = rlt::create_group(device, root_file, "example");
                auto inputs_group = rlt::create_group(device, example_group, "inputs");
                auto example_input_0_image_view = view_range(device, example_input_0_image, (TI)0, rlt::tensor::ViewSpec<1, EXAMPLE_BATCH_SIZE>{});
                auto example_input_1_state_view = view_range(device, example_input_1_state, (TI)0, rlt::tensor::ViewSpec<1, EXAMPLE_BATCH_SIZE>{});
                rlt::save(device, example_input_0_image_view, inputs_group, "0");
                rlt::save(device, example_input_1_state_view, inputs_group, "1");
                auto outputs_group = rlt::create_group(device, example_group, "outputs");
                auto example_output_canonical = reshape_row_major(device, example_output, rlt::tensor::Shape<TI, 1, N_EXAMPLES, ACTION_DIM>{});
                auto example_output_view = view_range(device, example_output_canonical, (TI)0, rlt::tensor::ViewSpec<1, EXAMPLE_BATCH_SIZE>{});
                rlt::save(device, example_output_view, outputs_group, "0");
                free(device, sized_eval_actor);
                return checkpoint_path;
            };
            auto reduced_checkpoint_path = save_hdf5(rlt::utils::typing::integral_constant<TI, REDUCED_BATCH_SIZE>{});
            link_latest_artifact(device, latest_folder, reduced_checkpoint_path, step_folder, environment_step);
            if constexpr(N_EXAMPLES != REDUCED_BATCH_SIZE){
                auto full_checkpoint_path = save_hdf5(rlt::utils::typing::integral_constant<TI, N_EXAMPLES>{});
                link_latest_artifact(device, latest_folder, full_checkpoint_path, step_folder, environment_step);
            }
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
                    auto example_output_canonical = reshape_row_major(device, example_output, rlt::tensor::Shape<TI, 1, N_EXAMPLES, ACTION_DIM>{});
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
                    link_latest_artifact(device, latest_folder, checkpoint_code_path, step_folder, environment_step);
                }
#endif
                {
                    std::filesystem::path checkpoint_code_path = step_folder / "checkpoint.h";
                    std::ofstream f(checkpoint_code_path);
                    f << output_string;
                    f.close();
                    link_latest_artifact(device, latest_folder, checkpoint_code_path, step_folder, environment_step);
                }
            }
            free(device, example_input_0_image);
            free(device, example_input_1_state);
            free(device, example_output);
            free(device, eval_actor);
            std::cerr << "Checkpoint saved: " << std::filesystem::absolute(step_folder) << std::endl;
        }

        {
            auto& actor_log_std = ppo.actor.head;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                T log_std_val = get(device, actor_log_std.log_std.parameters, action_i);
                add_scalar(device, device.logger, "actor/log_std", log_std_val);
                add_scalar(device, device.logger, std::string("actor/action_std/") + std::to_string(action_i), rlt::math::exp(device.math, log_std_val));
            }
        }
        if(training_complete){
            break;
        }
    }

    std::signal(SIGPIPE, previous_sigpipe_handler);
    std::cout << "Training finished at env step " << environment_step << std::endl;

    free(device, ppo);
    free(device, ppo_buffers);
    free(device, dataset);
    free(device, actor_optimizer);
    free(device, critic_optimizer);
    free(device_gpu, teacher_actions_gpu);
    free(device_gpu, teacher_body_positions);
    free(device, teacher_actions);
    free(device, reward_log_action);
    free(device, reward_log_rng);
    free(device, rng);
    free(device_gpu, rng_gpu);

    free(device_gpu, ppo_gpu);
    free(device_gpu, actor_buffers);
    free(device_gpu, critic_buffers);
    free(device_gpu, critic_states_gae);
    free(device_gpu, critic_buffers_gae);
    free(device_gpu, dataset_gpu);
    free(device_gpu, rollout_actor_gpu);
    free(device_gpu, rollout_actor_buffers);
    free(device_gpu, actor_optimizer_gpu);
    free(device_gpu, critic_optimizer_gpu);
    free(device_gpu, gpu_task_observations);
    free(device_gpu, gpu_rollout_combined);
    free(device_gpu, gpu_all_features);
    free(device_gpu, matching_gpu);
    free(device_gpu, matching_buffers);
    free(device, matching);
    free(device_gpu, gpu_all_state_observations);
    free(device_gpu, gpu_actions_eval);
    free(device_gpu, gpu_actions_train);
    free(device_gpu, gpu_d_action_train);
    free(device_gpu, gpu_critic_obs);
    free(device_gpu, gpu_d_critic_output);
    free(device_gpu, gpu_runner);
    free(device_gpu, gpu_runner_buffer);
    free(device_gpu, env);
    delete env_storage;

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
    free(device, device.logger);
#endif

    return 0;
}
