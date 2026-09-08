#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <metra/metra.h>
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
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cuda.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/operations_generic.h>
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

#include <array>
#include <deque>
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
using rlt::add_scalar;
using rlt::reset;
using rlt::prologue;
using rlt::sample_actions;
using rlt::epilogue;
using rlt::evaluate_values;
using rlt::evaluate_rollout_values;
using rlt::evaluate_bootstrap_values;

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
using DEVICE_GPU_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_GPU_SPEC>;

#include "training_hyperdrone_config.h"
#include "training_hyperdrone_scenes.h"
#include "training_hyperdrone_loss.h"
using rlt::resolve_training_scenes;
using rlt::select_scene;
using rlt::zero_gradient;
using rlt::step;
using rlt::training_hyperdrone_actor_loss_gradient;
using namespace rlt::rl::environments::l2f_visual::training;

static constexpr TI FRAME_STACK_HISTORY_LENGTH = FRAME_STACK_STRIDE * (FRAME_STACK_N - 1) + ROLLOUT_STEPS_PER_ENV + 1;

struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, STATIC_PARAMETERS>{
    static constexpr TI INSTANCES_PER_ENVIRONMENT = N_ENVIRONMENTS_PER_SCENE;
    static constexpr TI CAM_WIDTH = ::CAM_WIDTH;
    static constexpr TI CAM_HEIGHT = ::CAM_HEIGHT;
    static constexpr TI NUM_PROBES = ::NUM_PROBES;
    using SHADING = RENDER_SHADING;
    static constexpr bool ENABLE_MOTION_BLUR = RENDER_ENABLE_MOTION_BLUR;
    static constexpr TI MOTION_BLUR_SAMPLES = RENDER_MOTION_BLUR_SAMPLES;
    static constexpr bool ENABLE_ANTI_ALIASING = RENDER_ENABLE_ANTI_ALIASING;
    static constexpr TI ANTI_ALIASING_GRID_SIZE = RENDER_ANTI_ALIASING_GRID_SIZE;
    static constexpr TI HISTORY_LENGTH = FRAME_STACK_HISTORY_LENGTH;
    static constexpr T CAMERA_FOV = ::CAMERA_FOV;
    static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = ::CAMERA_FOV_RANDOMIZATION_RANGE;
    // this configuration randomizes all three mount axes with equal ranges
    static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X;
    static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X;
    static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = ::BRIGHTNESS_RANDOMIZATION_RANGE;
    static constexpr T SHUTTER_FRACTION_MIN = RENDER_SHUTTER_FRACTION_MIN;
    static constexpr T SHUTTER_FRACTION_MAX = RENDER_SHUTTER_FRACTION_MAX;
};
using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
static_assert(BASE_WORLD::RENDERER_SPEC::SHADING::PBR_SHADING, "l2f visual training must use the High renderer profile");
struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD>{
    static constexpr TI IMAGE_STACK_N = FRAME_STACK_N;
    static constexpr TI IMAGE_STACK_STRIDE = FRAME_STACK_STRIDE;
    static constexpr TI PAD_CHANNELS_TO = 8;  // cuDNN tensor-core fast path
    static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = ::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE;
    static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = ::TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE;
};
using TASK_WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
static constexpr TI NUMBER_OF_ENVIRONMENTS = N_ACTIVE_SCENES;
using MULTI_ENVIRONMENT = rlt::rl::environments::hyperdrone::MultiEnvironment<TASK_WORLD, NUMBER_OF_ENVIRONMENTS>;
using ENVIRONMENT = TASK_WORLD;
static constexpr bool RENDER_MOTION_BLUR_ACTIVE = BASE_WORLD::RENDERER_SPEC::ENABLE_MOTION_BLUR;
static constexpr bool RENDER_ANTI_ALIASING_ACTIVE = BASE_WORLD::RENDERER_SPEC::ENABLE_ANTI_ALIASING;

// Mosaic layout: each env cell shows (target | actual) pair, arranged in an ENV_GRID_SIDE×ENV_GRID_SIDE grid per active scene.
static constexpr TI ENV_GRID_SIDE = 4;
static constexpr TI SCENE_GRID_COLS = N_ACTIVE_SCENES;
static constexpr TI SCENE_GRID_ROWS = (N_ACTIVE_SCENES + SCENE_GRID_COLS - 1) / SCENE_GRID_COLS;
static_assert(ENV_GRID_SIDE * ENV_GRID_SIDE == N_ENVIRONMENTS_PER_SCENE, "ENV_GRID_SIDE^2 must equal N_ENVIRONMENTS_PER_SCENE for the mosaic layout");

// =========================================================================
// Frame stacking + target-channel concatenation
// =========================================================================
static_assert(COMBINED_IMG_C == TASK_WORLD::OBSERVATION_CHANNELS);
static_assert(COMBINED_OBS_DIM == TASK_WORLD::OBSERVATION_DIM);

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

std::string trajectory_episodes_to_json(DEVICE& device, TASK_WORLD& env,
    const std::vector<std::vector<TrajectoryStep>>& episodes, T dt){
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
        std::string parameters_json = rlt::json(device, env.dynamics, parameters.dynamics);
        parameters_json = parameters_json.substr(0, parameters_json.size() - 1);
        parameters_json += ", \"visual\": {\"scene_translation\": [" + std::to_string(parameters.scene_translation[0]) + ", " + std::to_string(parameters.scene_translation[1]) + ", " + std::to_string(parameters.scene_translation[2]) + "]"
            + ", \"scene_yaw\": " + std::to_string(rlt::math::atan2(device.math, parameters.scene_yaw_sin, parameters.scene_yaw_cos))
            + ", \"scene_hash\": \"" + episode.front().scene_hash + "\""
            + ", \"cam_width\": " + std::to_string(WORLD_SPEC::CAM_WIDTH)
            + ", \"cam_height\": " + std::to_string(WORLD_SPEC::CAM_HEIGHT)
            + ", \"fov\": " + std::to_string(parameters.fov) + "}}";
        json += "{\"parameters\": " + parameters_json + ",\n";
        json += "\"trajectory\": [";
        for(TI step_i = 0; step_i < max_len; step_i++){
            auto& s = (step_i < episode.size()) ? episode[step_i] : episode.back();
            json += "{\"state\":" + rlt::json(device, env.dynamics, parameters.dynamics, s.state) + ",";
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

// =========================================================================
// PPO configuration
// =========================================================================
using LOOP_CORE_PARAMETERS = rlt::rl::environments::l2f_visual::training::LoopParameters<ENVIRONMENT>;
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
using ACTOR_OPTIMIZER = typename LOOP_CORE_CONFIG::NN::ACTOR_OPTIMIZER;
using CRITIC_OPTIMIZER = typename LOOP_CORE_CONFIG::NN::CRITIC_OPTIMIZER;
using ACTOR_BUFFERS = typename LOOP_CORE_CONFIG::ACTOR_BUFFERS;
using CRITIC_BUFFERS = typename LOOP_CORE_CONFIG::CRITIC_BUFFERS;
using ACTOR_TYPE = typename LOOP_CORE_CONFIG::NN::ACTOR_TYPE;

// Rollout actor: forward-only with batch size N_ENVIRONMENTS
using CAPABILITY_ROLLOUT = rlt::nn::capability::Forward<true>;
using ROLLOUT_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<CAPABILITY_ROLLOUT>::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>;
using ROLLOUT_ACTOR_BUFFERS = typename ROLLOUT_ACTOR_TYPE::template Buffer<true>;
using CHECKPOINT_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<CAPABILITY_ROLLOUT>::template CHANGE_BATCH_SIZE<TI, N_EXAMPLES>;
using ROLLOUT_POLICY_STATE = typename ROLLOUT_ACTOR_TYPE::template State<true>;
using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<TYPE_POLICY, MULTI_ENVIRONMENT, ROLLOUT_POLICY_STATE, typename BASE_WORLD::Observation, typename BASE_WORLD::ObservationPrivileged, T, T, EPISODE_STEP_LIMIT, LOOP_CORE_PARAMETERS::PPO_PARAMETERS::TRUNCATE_ON_EACH_ITERATION, true, LOOP_CORE_PARAMETERS::PPO_PARAMETERS::BOOTSTRAP_TRUNCATIONS || LOOP_CORE_PARAMETERS::PPO_PARAMETERS::IGNORE_TERMINATION>;
using ON_POLICY_RUNNER = rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
using ON_POLICY_RUNNER_BUFFER = rlt::rl::components::on_policy_runner::Buffer<ON_POLICY_RUNNER_SPEC>;
using ON_POLICY_RUNNER_DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, LOOP_CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV, true>;
using ON_POLICY_RUNNER_DATASET_TYPE = rlt::rl::components::on_policy_runner::Dataset<ON_POLICY_RUNNER_DATASET_SPEC>;

// Constants
static constexpr TI STEPS_PER_ENV = LOOP_CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV;
static constexpr TI BATCH_SIZE = LOOP_CORE_PARAMETERS::BATCH_SIZE;
static constexpr TI OBSERVATION_DIM = BASE_WORLD::OBSERVATION_DIM;
static constexpr TI OBS_PRIV_DIM = ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED;
static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
static constexpr TI STEPS_TOTAL = ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL;
static constexpr TI STEPS_TOTAL_ALL = ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL;
static constexpr TI N_EPOCHS = LOOP_CORE_PARAMETERS::PPO_PARAMETERS::N_EPOCHS;
static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
static constexpr TI UPDATE_SAMPLES = STEPS_TOTAL * GRADIENT_ACCUMULATION_ROLLOUTS;
static constexpr T GRADIENT_SCALE = static_cast<T>(BATCH_SIZE) / static_cast<T>(UPDATE_SAMPLES);
static_assert(GRADIENT_ACCUMULATION_ROLLOUTS == 1);
static_assert(STEPS_TOTAL % BATCH_SIZE == 0);
static_assert(N_EPOCHS == 1);
static_assert(N_ENVIRONMENTS <= RNG_GPU::NUM_RNGS);
static constexpr TI IMG_H = BASE_WORLD::Observation::HEIGHT;
static constexpr TI IMG_W = BASE_WORLD::Observation::WIDTH;
static constexpr TI IMG_C = BASE_WORLD::Observation::CHANNELS;

static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");
static_assert(N_EXAMPLES <= BATCH_SIZE, "N_EXAMPLES must fit the reusable combined-observation batch buffer");
static_assert(N_EXAMPLES <= STEPS_TOTAL, "N_EXAMPLES must fit one PPO rollout dataset");

static_assert(ON_POLICY_RUNNER_SPEC::STEP_LIMIT == EPISODE_STEP_LIMIT);

// =========================================================================
// Custom CUDA kernels
// =========================================================================

// Build combined (stacked frames + target frame) training batches by gathering from the World's
// frame history (one launch per World; rows of other Worlds' instances exit early). Row layout:
// (step_i, env_i) with env_i member-major; the per-World history is [slot][local_env][frame].
// The history is indexed relative to the World's latest render (the rollout's final observation,
// frame_step_end), so the extra render of a scene rotation between rollouts does not shift the stack.
__global__ void build_frame_stacked_with_target_from_dataset_kernel(
    const float* __restrict__ history_obs,
    const float* __restrict__ all_target_obs,
    const TI* __restrict__ episode_start_step_per_row,
    float* __restrict__ combined_out,
    int obs_dim, int img_c, int n_frames, int frame_stride, int combined_img_c, int combined_obs_dim,
    TI frame_step_start, TI frame_step_end, TI history_step_end, int batch_offset, int batch_size, int n_envs,
    int base_env, int envs_per_world
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx >= batch_size * combined_obs_dim){
        return;
    }
    int sample = global_idx / combined_obs_dim;
    int offset = global_idx % combined_obs_dim;
    int pixel = offset / combined_img_c;
    int frame_channel = offset % combined_img_c;
    int row = batch_offset + sample;
    int env_i = row % n_envs;
    if(env_i < base_env || env_i >= base_env + envs_per_world){
        return;
    }
    int local_env = env_i - base_env;
    int step_i_local = row / n_envs;
    int logical_channels = n_frames * img_c + img_c;
    if(frame_channel < n_frames * img_c){
        int frame = frame_channel / img_c;
        int channel = frame_channel % img_c;
        TI episode_start = episode_start_step_per_row[row];
        TI back = static_cast<TI>(frame) * static_cast<TI>(frame_stride);
        TI frame_step_i = frame_step_start + static_cast<TI>(step_i_local);
        TI desired_step = frame_step_i >= back ? frame_step_i - back : episode_start;
        if(desired_step < episode_start){
            desired_step = episode_start;
        }
        TI renders_back = frame_step_end - desired_step;
        TI history_slot = (history_step_end - 1 - renders_back) % FRAME_STACK_HISTORY_LENGTH;
        TI src_row = history_slot * envs_per_world + local_env;
        combined_out[global_idx] = history_obs[src_row * obs_dim + pixel * img_c + channel];
    } else if(frame_channel < logical_channels) {
        int channel = frame_channel - n_frames * img_c;
        TI src_row = (TI)row;
        combined_out[global_idx] = all_target_obs[src_row * obs_dim + pixel * img_c + channel];
    } else {
        combined_out[global_idx] = 0.0f;
    }
}

#ifdef RL_TOOLS_L2F_VISUAL_TRAINING_SMOKE
static constexpr TI SMOKE_PIXELS = 5;
static constexpr TI SMOKE_FEATURES = SMOKE_PIXELS * COMBINED_IMG_C;
__global__ void capture_observation_probes_kernel(const T* observations, T* probes){
    const TI env_i = blockIdx.x;
    const TI feature = threadIdx.x;
    if(feature < SMOKE_FEATURES){
        const TI pixel = (feature / COMBINED_IMG_C) * (CAM_WIDTH * CAM_HEIGHT - 1) / (SMOKE_PIXELS - 1);
        probes[env_i * SMOKE_FEATURES + feature] = observations[env_i * COMBINED_OBS_DIM + pixel * COMBINED_IMG_C + feature % COMBINED_IMG_C];
    }
}
__global__ void compare_observation_probes_kernel(const T* observations, const T* probes, bool* matches){
    const TI row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < BATCH_SIZE){
        bool match = true;
        for(TI feature = 0; feature < SMOKE_FEATURES; feature++){
            const TI pixel = (feature / COMBINED_IMG_C) * (CAM_WIDTH * CAM_HEIGHT - 1) / (SMOKE_PIXELS - 1);
            match = match && observations[row * COMBINED_OBS_DIM + pixel * COMBINED_IMG_C + feature % COMBINED_IMG_C] == probes[row * SMOKE_FEATURES + feature];
        }
        matches[row] = match;
    }
}
#endif

// per-row episode start (in frame steps) for the dataset re-assembly: a reset instance starts at
// the current frame step
__global__ void record_episode_start_kernel(const bool* reset_mask, TI* episode_start_step, TI* episode_start_step_per_row, TI step_i, TI frame_step_i){
    TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
    if(env_i >= N_ENVIRONMENTS){
        return;
    }
    if(reset_mask[env_i]){
        episode_start_step[env_i] = frame_step_i;
    }
    episode_start_step_per_row[step_i * N_ENVIRONMENTS + env_i] = episode_start_step[env_i];
}

// =========================================================================
// Main
// =========================================================================
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
    // Resolve scenes: a directory of .glb scenes (this run trains on the first N_TOTAL_SCENES
    // of the corpus, ordered by numeric suffix) or a single .glb replicated across all Worlds
    // ---------------------------------------------------------------------
    DEVICE_GPU device_gpu;
    auto& device = device_gpu.rendering;
    rlt::init(device);
    rlt::init(device_gpu);
    rlt::rendering::datasets::procthor::GLB scene_dataset{{}, resolve_training_scenes(device, argv[1])};
    std::mt19937 scene_rng(seed);
    std::vector<TI> scene_permutation(N_TOTAL_SCENES);
    std::iota(scene_permutation.begin(), scene_permutation.end(), 0);

    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = "l2f_visual_training_hyperdrone";
    extrack_config.population_variates = "algorithm_environments_accumulation";
    extrack_config.population_values = "on-policy_" + std::to_string(N_ENVIRONMENTS) + "_" + std::to_string(GRADIENT_ACCUMULATION_ROLLOUTS);
    rlt::init(device, extrack_config, extrack_paths, seed);
    metra::log_raw("l2f_visual_training_on_policy/target", "\"hyperdrone\"");
    metra::log("l2f_visual_training_on_policy/seed", static_cast<double>(seed));

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
    ON_POLICY_RUNNER_DATASET_TYPE dataset;
    ACTOR_OPTIMIZER actor_optimizer;
    CRITIC_OPTIMIZER critic_optimizer;
    rlt::malloc(device, ppo);
    rlt::malloc(device, ppo_buffers);
    rlt::malloc(device, dataset);
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, critic_optimizer);

    // ---------------------------------------------------------------------
    // Environment: MultiEnvironment of target-frame task Worlds over the shared scene set
    // ---------------------------------------------------------------------
    auto* env_storage = new MULTI_ENVIRONMENT{};
    MULTI_ENVIRONMENT& env = *env_storage;
    rlt::malloc(device_gpu, env);
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
        rlt::init(device_gpu, world, env.shared, scene_dataset, scene_corpus, (TI)0, N_TOTAL_SCENES, member_i);
    }
    std::cout << "Loaded " << N_TOTAL_SCENES << " scenes" << std::endl;

    {
        std::string ui = rlt::get_ui(device, env.environments[0].dynamics);
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
    TI global_env_step = 0;
    TI optimizer_updates = 0;

    // ---------------------------------------------------------------------
    // GPU init
    // ---------------------------------------------------------------------
    RNG_GPU rng_gpu;
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_gpu, rng_gpu, seed);

    PPO_TYPE ppo_gpu;
    ACTOR_BUFFERS actor_buffers;
    CRITIC_BUFFERS critic_buffers;
    rlt::rl::components::on_policy_runner::ValueState<typename PPO_SPEC::CRITIC_TYPE, ON_POLICY_RUNNER_DATASET_SPEC> critic_states_gae;
    rlt::rl::components::on_policy_runner::ValueBuffer<typename PPO_SPEC::CRITIC_TYPE, ON_POLICY_RUNNER_DATASET_SPEC> critic_buffers_gae;
    ON_POLICY_RUNNER_DATASET_TYPE dataset_gpu;
    rlt::malloc(device_gpu, ppo_gpu);
    rlt::malloc(device_gpu, actor_buffers);
    rlt::malloc(device_gpu, critic_buffers);
    rlt::malloc(device_gpu, critic_states_gae); rlt::malloc(device_gpu, critic_buffers_gae);
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

    // ---------------------------------------------------------------------
    // GPU-resident instance data (the environment's vectorized axis)
    // ---------------------------------------------------------------------
    // the runner owns persistent rollout state and episode accounting; its buffer owns transient
    // next-state/action/reward storage, and the dataset owns per-rollout episode completions
    ON_POLICY_RUNNER gpu_runner;
    ON_POLICY_RUNNER_BUFFER gpu_runner_buffer;
    rlt::malloc(device_gpu, gpu_runner);
    rlt::malloc(device_gpu, gpu_runner_buffer);
    auto& gpu_env_parameters = gpu_runner.env_parameters;
    auto& gpu_env_states = gpu_runner.states;
    auto& gpu_step_rewards = gpu_runner_buffer.rewards;
    auto& gpu_step_actions = gpu_runner_buffer.actions;
    // ---------------------------------------------------------------------
    // Auxiliary GPU buffers for visual pipeline
    // ---------------------------------------------------------------------
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, OBSERVATION_DIM>>> gpu_all_target_observations;
    rlt::malloc(device_gpu, gpu_all_target_observations);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, COMBINED_OBS_DIM>>> gpu_rollout_combined;
    rlt::malloc(device_gpu, gpu_rollout_combined);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, COMBINED_OBS_DIM>>> gpu_combined_batch;
    rlt::malloc(device_gpu, gpu_combined_batch);
#ifdef RL_TOOLS_L2F_VISUAL_TRAINING_SMOKE
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, SMOKE_FEATURES>>> observation_probes;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, BATCH_SIZE>>> observation_matches_gpu, observation_matches;
    rlt::malloc(device_gpu, observation_probes);
    rlt::malloc(device_gpu, observation_matches_gpu);
    rlt::malloc(device, observation_matches);
#endif

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, STATE_OBS_DIM>>> gpu_all_state_observations;
    rlt::malloc(device_gpu, gpu_all_state_observations);

    rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ACTION_DIM>> gpu_actions_eval;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_actions_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_d_action_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, OBS_PRIV_DIM>> gpu_critic_obs;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, 1>> gpu_d_critic_output;
    rlt::malloc(device_gpu, gpu_actions_eval);
    rlt::malloc(device_gpu, gpu_actions_train);
    rlt::malloc(device_gpu, gpu_d_action_train);
    rlt::malloc(device_gpu, gpu_critic_obs);
    rlt::malloc(device_gpu, gpu_d_critic_output);

    TI* gpu_episode_start_step = nullptr;
    TI* gpu_episode_start_step_per_row = nullptr;
    cudaMalloc(&gpu_episode_start_step, N_ENVIRONMENTS * sizeof(TI));
    cudaMalloc(&gpu_episode_start_step_per_row, STEPS_TOTAL * sizeof(TI));
    cudaMemset(gpu_episode_start_step, 0, N_ENVIRONMENTS * sizeof(TI));

    EpisodeRecorder episode_recorders[TRAJECTORY_NUM_ENVS];
    std::vector<std::vector<TrajectoryStep>> completed_episodes;
    T simulation_dt = static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY);
    typename TASK_WORLD::State reward_log_state;
    typename TASK_WORLD::State reward_log_next_state;
    typename TASK_WORLD::Parameters reward_log_parameters;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_DIM>> reward_log_action;
    rlt::malloc(device, reward_log_action);

    // Trajectory state buffer for post-collect reconstruction
    std::vector<typename TASK_WORLD::State> trajectory_states(STEPS_PER_ENV * TRAJECTORY_NUM_ENVS);
    std::vector<typename TASK_WORLD::Parameters> trajectory_parameters(STEPS_PER_ENV * TRAJECTORY_NUM_ENVS);

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
    std::cout << "Starting single-use on-policy training (visual L2F target navigation, Hyperdrone CUDA)" << std::endl;
    std::cout << "  N_ENVIRONMENTS:   " << N_ENVIRONMENTS << std::endl;
    std::cout << "  STEPS_PER_ENV:    " << STEPS_PER_ENV << std::endl;
    std::cout << "  SCENE_SET_STEPS:  " << STEPS_PER_ENV * ROLLOUTS_PER_SCENE_SET << std::endl;
    std::cout << "  STEPS_TOTAL:      " << STEPS_TOTAL << std::endl;
    std::cout << "  BATCH_SIZE:       " << BATCH_SIZE << std::endl;
    std::cout << "  N_BATCHES:        " << N_BATCHES << std::endl;
    std::cout << "  COMBINED_IMG_C:   " << COMBINED_IMG_C << std::endl;
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

    std::vector<T> episode_returns(N_ENVIRONMENTS, 0);
    std::vector<TI> episode_lengths(N_ENVIRONMENTS, 0);
    std::vector<TI> episode_end_reasons(N_ENVIRONMENTS, 0);
    for(TI ppo_step_i = 0; ppo_step_i < N_PPO_STEPS; ppo_step_i++){
        auto step_start = std::chrono::high_resolution_clock::now();
        zero_gradient(device_gpu, ppo_gpu.actor);
        zero_gradient(device_gpu, ppo_gpu.critic);
        rlt::set_step(device, device.logger, global_env_step);

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
            std::shuffle(scene_permutation.begin(), scene_permutation.end(), scene_rng);
            for(TI member_i = 0; member_i < NUMBER_OF_ENVIRONMENTS; member_i++){
                select_scene(device_gpu, env.environments[member_i], scene_permutation[member_i]);
            }
            if(ppo_step_i == 0){
                rlt::init(device_gpu, gpu_runner, env, rng_gpu);
            } else {
                reset(device_gpu, gpu_runner, env, rng_gpu);
            }
        }

        // =================================================================
        // Optional video recording (mosaic of target | actual frames)
        // =================================================================
        if(scene_set_boundary){
            record_video_scene_set = scene_set_i % VIDEO_SAVE_INTERVAL_SCENE_SETS == 0;
        }
        if(record_video_scene_set && ffmpeg_pipe == nullptr){
            TI video_step = global_env_step + N_ENVIRONMENTS * STEPS_PER_ENV * ROLLOUTS_PER_SCENE_SET;
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
        // row 0: the current raw frames and privileged observations, plus the dataset's reset column
        prologue(device_gpu, dataset_gpu, gpu_runner, env, rng_gpu);
        for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
            evaluate_values(device_gpu, dataset_gpu, ppo_gpu.critic, critic_states_gae, critic_buffers_gae, rng_gpu, step_i);
            TI frame_step_i = frame_step_start + step_i;
            // 1. Driver-side per-row data: the episode start (frame-stack guard), the actor's state
            // branch observation and the cached target frame
            record_episode_start_kernel<<<grid, block, 0, device_gpu.stream>>>(rlt::data(gpu_runner.reset), gpu_episode_start_step, gpu_episode_start_step_per_row, step_i, frame_step_i);
            rlt::check_status(device_gpu);
            {
                auto state_observations = rlt::view_range(device_gpu, gpu_all_state_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                rlt::observe(device_gpu, env, gpu_env_parameters, gpu_env_states, ACTOR_STATE_OBS{}, state_observations, rng_gpu);
            }
            T* obs_ptr = rlt::data(dataset_gpu.all_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
            T* target_obs_ptr = rlt::data(gpu_all_target_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
            for(TI member_i = 0; member_i < NUMBER_OF_ENVIRONMENTS; member_i++){
                auto& world = env.environments[member_i];
                constexpr TI M = TASK_WORLD::INSTANCES;
                cudaMemcpyAsync(target_obs_ptr + member_i * M * OBSERVATION_DIM, rlt::data(world.target_frames), M * OBSERVATION_DIM * sizeof(float), cudaMemcpyDeviceToDevice, device_gpu.stream);
            }
            rlt::check_status(device_gpu);

            // 2. The composed [stack | target | pad] rollout observation for the actor
            rlt::observe(device_gpu, env, gpu_env_parameters, gpu_env_states, typename TASK_WORLD::Observation{}, gpu_rollout_combined, rng_gpu);
#ifdef RL_TOOLS_L2F_VISUAL_TRAINING_SMOKE
            capture_observation_probes_kernel<<<N_ENVIRONMENTS, 256, 0, device_gpu.stream>>>(rlt::data(gpu_rollout_combined), rlt::data(observation_probes) + step_i * N_ENVIRONMENTS * SMOKE_FEATURES);
#endif

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

            // 7. Gaussian action sampling into the dataset and the runner's step actions, then the
            // runner's epilogue: step, reward, episode accounting, autoreset, next row
            if(log_reward_components_this_step && step_i == STEPS_PER_ENV - 1){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(&reward_log_state, rlt::data(gpu_env_states), sizeof(typename TASK_WORLD::State), cudaMemcpyDeviceToHost);
                cudaMemcpy(&reward_log_parameters, rlt::data(gpu_env_parameters), sizeof(typename TASK_WORLD::Parameters), cudaMemcpyDeviceToHost);
            }
            {
                auto& last_layer_gpu = ppo_gpu.actor.head;
                auto log_std_gpu = rlt::matrix_view(device_gpu, last_layer_gpu.log_std.parameters);
                sample_actions(device_gpu, dataset_gpu, log_std_gpu, gpu_step_actions, step_i, rng_gpu);
            }
            if(save_extrack_step){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(trajectory_parameters.data() + step_i * TRAJECTORY_NUM_ENVS, rlt::data(gpu_env_parameters), TRAJECTORY_NUM_ENVS * sizeof(typename TASK_WORLD::Parameters), cudaMemcpyDeviceToHost);
            }
            epilogue(device_gpu, dataset_gpu, gpu_runner, gpu_runner_buffer, env, rng_gpu, step_i);
            evaluate_bootstrap_values(device_gpu, dataset_gpu, gpu_runner_buffer.next_observations_privileged, ppo_gpu.critic, critic_states_gae, critic_buffers_gae, rng_gpu, step_i);
            if(log_reward_components_this_step && step_i == STEPS_PER_ENV - 1){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(&reward_log_next_state, rlt::data(gpu_runner_buffer.next_states), sizeof(typename TASK_WORLD::State), cudaMemcpyDeviceToHost);
            }

            // 8. Pull state for trajectory recording
            if(save_extrack_step){
                cudaStreamSynchronize(device_gpu.stream);
                std::vector<typename TASK_WORLD::State> tmp_states(TRAJECTORY_NUM_ENVS);
                cudaMemcpy(tmp_states.data(), rlt::data(gpu_runner_buffer.next_states), TRAJECTORY_NUM_ENVS * sizeof(typename TASK_WORLD::State), cudaMemcpyDeviceToHost);
                for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                    trajectory_states[step_i * TRAJECTORY_NUM_ENVS + env_i] = tmp_states[env_i];
                }
            }
        }

        if(ffmpeg_pipe && rollout_in_scene_set + 1 == ROLLOUTS_PER_SCENE_SET){
            close_video_pipe();
        }

        global_env_step += N_ENVIRONMENTS * STEPS_PER_ENV;
        rlt::set_step(device, device.logger, global_env_step);

        evaluate_rollout_values(device_gpu, dataset_gpu, ppo_gpu.critic, critic_buffers_gae, rng_gpu, PPO_SPEC::COLLECTION_MODE{});

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
            rlt::log_reward(device, env.environments[0].dynamics, reward_log_parameters.dynamics, reward_log_state, reward_log_action, reward_log_next_state, reward_log_rng);
        }

        // Episode statistics + log
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
                const TI env_i = pos % N_ENVIRONMENTS;
                T ep_len = rlt::get(dataset.reset, pos, 0) > (T)0.5 && episode_lengths[env_i] > 0 ? static_cast<T>(episode_lengths[env_i]) : (T)-1;
                if(ep_len >= (T)0){
                    rlt::add_scalar(device, device.logger, "episode/length", ep_len, 100);
                    T ep_return = episode_returns[env_i];
                    rlt::add_scalar(device, device.logger, "episode/return", ep_return, 100);
                    length_sum += ep_len;
                    length_sq_sum += ep_len * ep_len;
                    return_sum += ep_return;
                    return_sq_sum += ep_return * ep_return;
                    TI ep_reason = episode_end_reasons[env_i] != 0 ? episode_end_reasons[env_i] : static_cast<TI>(EPISODE_END_REASON_SCENE_BOUNDARY);
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
            rlt::utils::assert_exit(device, episode_end_count == count, "Every completed episode must have an end reason");
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
                    ts.parameters = trajectory_parameters[step_i * TRAJECTORY_NUM_ENVS + env_i];
                    const auto& world = env.environments[env_i / N_ENVIRONMENTS_PER_SCENE];
                    ts.scene_hash = world.slots[world.active_slot].metadata.content_hash;
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
                auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, global_env_step);
                std::string trajectories_json = trajectory_episodes_to_json(device, env.environments[0], completed_episodes, simulation_dt);
                std::vector<uint8_t> compressed;
                if(rlt::compress_zlib(trajectories_json, compressed)){
                    std::filesystem::path trajectories_path = step_folder / "trajectories.json.gz";
                    std::ofstream f(trajectories_path, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
                    f.close();
                    auto latest_folder = rlt::get_latest_folder(device, extrack_paths);
                    rlt::link_latest_artifact(device, latest_folder, trajectories_path, step_folder, global_env_step);
                }
                std::cout << "  Saved " << completed_episodes.size() << " trajectory episodes to " << step_folder << std::endl;
                completed_episodes.clear();
            }
        }

        // =================================================================
        // GAE
        // =================================================================
        rlt::estimate_generalized_advantages(device, dataset, dataset.bootstrap_values, typename PPO_TYPE::SPEC::PARAMETERS{});

        // =================================================================
        // Accumulate gradients over the fresh rollout before one optimizer update
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


                // Build combined batch observation by gathering from each World's frame history
                {
                    int total_elements = BATCH_SIZE * COMBINED_OBS_DIM;
                    for(TI member_i = 0; member_i < NUMBER_OF_ENVIRONMENTS; member_i++){
                        build_frame_stacked_with_target_from_dataset_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                            rlt::data(env.environments[member_i].history),
                            rlt::data(gpu_all_target_observations),
                            gpu_episode_start_step_per_row,
                            rlt::data(gpu_combined_batch),
                            OBSERVATION_DIM, IMG_C, FRAME_STACK_N, FRAME_STACK_STRIDE, COMBINED_IMG_C, COMBINED_OBS_DIM,
                            frame_step_start, frame_step_start + STEPS_PER_ENV, env.environments[member_i].history_step, (int)batch_offset, (int)BATCH_SIZE, (int)N_ENVIRONMENTS,
                            (int)(member_i * TASK_WORLD::INSTANCES), (int)TASK_WORLD::INSTANCES);
                    }
                    rlt::check_status(device_gpu);
                }

#ifdef RL_TOOLS_L2F_VISUAL_TRAINING_SMOKE
                compare_observation_probes_kernel<<<(BATCH_SIZE + 255) / 256, 256, 0, device_gpu.stream>>>(rlt::data(gpu_combined_batch), rlt::data(observation_probes) + batch_offset * SMOKE_FEATURES, rlt::data(observation_matches_gpu));
                rlt::copy(device_gpu, device, observation_matches_gpu, observation_matches);
                for(TI row = 0; row < BATCH_SIZE; row++){
                    rlt::utils::assert_exit(device, rlt::get(device, observation_matches, row), "Training frame stack differs from rollout observation");
                }
#endif
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

                // On-policy loss using freshly-forwarded action means and the fixed log_std
                rlt::copy(device_gpu, device, gpu_actions_train, ppo_buffers.current_batch_actions);
                auto& last_layer_gpu = ppo_gpu.actor.head;
                auto& last_layer_cpu = ppo.actor.head;
                rlt::copy(device_gpu, device, last_layer_gpu.log_std.parameters, last_layer_cpu.log_std.parameters);
                rlt::copy(device_gpu, device, last_layer_gpu.log_std.gradient, last_layer_cpu.log_std.gradient);

#ifdef RL_TOOLS_L2F_VISUAL_TRAINING_SMOKE
                for(TI row = 0; row < BATCH_SIZE; row++){
                    for(TI action = 0; action < ACTION_DIM; action++){
                        const T difference = rlt::math::abs(device.math, rlt::get(ppo_buffers.current_batch_actions, row, action) - rlt::get(dataset.actions_mean, batch_offset + row, action));
                        rlt::utils::assert_exit(device, difference < (T)1e-3, "Actor changed before accumulation completed");
                    }
                }
#endif
                auto batch_actions = rlt::view(device, dataset.actions, rlt::matrix::ViewSpec<BATCH_SIZE, ACTION_DIM>(), batch_offset, 0);
                auto batch_action_log_probs = rlt::view(device, dataset.action_log_probs, rlt::matrix::ViewSpec<BATCH_SIZE, 1>(), batch_offset, 0);
                auto batch_advantages = rlt::view(device, dataset.advantages, rlt::matrix::ViewSpec<BATCH_SIZE, 1>(), batch_offset, 0);
                auto batch_target_values = rlt::view(device, dataset.target_values, rlt::matrix::ViewSpec<BATCH_SIZE, 1>(), batch_offset, 0);

                auto statistics = training_hyperdrone_actor_loss_gradient(device, ppo, ppo_buffers, batch_actions, batch_action_log_probs, batch_advantages, rlt::utils::typing::integral_constant<TI, UPDATE_SAMPLES>{});
                ppo_actor_loss_sum += statistics.actor_loss;
                ppo_entropy_sum += statistics.entropy;
                ppo_approx_kl_sum += statistics.approx_kl;
                ppo_ratio_sum += statistics.ratio;
                ppo_advantage_mean_sum += statistics.advantage_mean;
                ppo_advantage_std_sum += statistics.advantage_std;
                ppo_update_samples += statistics.samples;
                ppo_clipped_samples += statistics.clipped_samples;
                // Sync accumulated log_std gradients and this minibatch's action derivatives.
                rlt::copy(device, device_gpu, last_layer_cpu.log_std.parameters, last_layer_gpu.log_std.parameters);
                rlt::copy(device, device_gpu, last_layer_cpu.log_std.gradient, last_layer_gpu.log_std.gradient);
                rlt::copy(device, device_gpu, ppo_buffers.d_action_log_prob_d_action, gpu_d_action_train);
                auto gpu_d_action_tensor = rlt::to_tensor(device_gpu, gpu_d_action_train);
                auto gpu_d_action_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(batch_combined_reshaped, batch_state_reshaped);
                rlt::backward(device_gpu, ppo_gpu.actor, bwd_inputs, gpu_d_action_reshaped, actor_buffers);
                cudaDeviceSynchronize();

                // Critic forward + MSE loss + backward on GPU
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
                        rlt::nn::loss_functions::mse::gradient(device, cpu_critic_output, batch_target_values, cpu_d_critic, (T)0.5 * GRADIENT_SCALE);
                        rlt::copy(device, device_gpu, cpu_d_critic, gpu_d_critic_output);
                        rlt::free(device, cpu_critic_output);
                        rlt::free(device, cpu_d_critic);
                    }
                    auto gpu_d_critic_tensor = rlt::to_tensor(device_gpu, gpu_d_critic_output);
                    auto gpu_d_critic_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_critic_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 1>{});
                    rlt::backward(device_gpu, ppo_gpu.critic, gpu_critic_obs_reshaped, gpu_d_critic_reshaped, critic_buffers);
                    cudaDeviceSynchronize();
                }
                ppo_update_batches++;
            }
        }

        rlt::utils::assert_exit(device, ppo_update_samples == UPDATE_SAMPLES, "Every rollout sample must be used exactly once");
        step(device_gpu, actor_optimizer_gpu, ppo_gpu.actor);
        step(device_gpu, critic_optimizer_gpu, ppo_gpu.critic);
        cudaDeviceSynchronize();
        optimizer_updates++;

        // Sync trained actor weights → rollout actor for next data collection
        rlt::copy(device_gpu, device_gpu, ppo_gpu.actor, rollout_actor_gpu);
        rlt::copy(device_gpu, device, ppo_gpu.actor.head.log_std.parameters, ppo.actor.head.log_std.parameters);
        rlt::copy(device_gpu, device, actor_optimizer_gpu, actor_optimizer);
        rlt::copy(device_gpu, device, critic_optimizer_gpu, critic_optimizer);
        rlt::utils::assert_exit(device, rlt::get(device, actor_optimizer.age, 0) == optimizer_updates + 1, "Unexpected actor optimizer age");
        rlt::utils::assert_exit(device, rlt::get(device, critic_optimizer.age, 0) == optimizer_updates + 1, "Unexpected critic optimizer age");

        // Logging
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> training_elapsed = now - training_start;
        std::chrono::duration<T> step_elapsed = now - step_start;
        T sps_lifetime = global_env_step / training_elapsed.count();
        T sps_current = N_ENVIRONMENTS * STEPS_PER_ENV / step_elapsed.count();
        std::cout << "PPO step " << std::setw(5) << ppo_step_i
                  << "  env_step " << std::setw(10) << global_env_step
                  << "  elapsed " << std::setw(7) << std::setprecision(3) << training_elapsed.count() << "s"
                  << "  (sps lifetime " << std::setw(6) << std::setprecision(0) << std::fixed << sps_lifetime
                  << ", current " << std::setw(6) << std::setprecision(0) << sps_current << ")" << std::defaultfloat << std::endl;

        std::cout << "  optimizer updates: " << optimizer_updates << "  accumulated samples: " << ppo_update_samples << "/" << UPDATE_SAMPLES << std::endl;
        rlt::add_scalar(device, device.logger, "ppo/step", ppo_step_i);
        rlt::add_scalar(device, device.logger, "training/optimizer_updates", optimizer_updates);
        rlt::add_scalar(device, device.logger, "training/accumulated_samples", ppo_update_samples);
        rlt::add_scalar(device, device.logger, "training/update_samples", UPDATE_SAMPLES);
        rlt::add_scalar(device, device.logger, "training/sample_uses", N_EPOCHS);
        metra::log("l2f_visual_training_on_policy/episode_length", static_cast<double>(rollout_episode_length_mean));
        metra::log("l2f_visual_training_on_policy/episode_return", static_cast<double>(rollout_return_mean));
        metra::log("l2f_visual_training_on_policy/optimizer_updates", static_cast<double>(optimizer_updates));
        metra::log("l2f_visual_training_on_policy/steps_per_second", static_cast<double>(sps_current));
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
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_offset_randomization_range_x", CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_offset_randomization_range_y", CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_offset_randomization_range_z", CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_rotation_randomization_range_x", CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_rotation_randomization_range_y", CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_rotation_randomization_range_z", CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z);

        if(save_extrack_step){
            {
                auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, global_env_step);
                auto latest_folder = rlt::get_latest_folder(device, extrack_paths);
                std::filesystem::create_directories(step_folder);

                CHECKPOINT_ACTOR_TYPE eval_actor;
                rlt::malloc(device, eval_actor);
                rlt::copy(device_gpu, device, ppo_gpu.actor, eval_actor);

                char fov_buf[32];
                std::snprintf(fov_buf, sizeof(fov_buf), "%.6g", (double)WORLD_SPEC::CAMERA_FOV);
                std::string state_obs_string = rlt::string(device, env.environments[0].dynamics, ACTOR_STATE_OBS{});
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
                    + ", \"target_frame_brightness_mismatch_range\": " + std::to_string(TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE)
                    + ", \"camera_mount_offset_randomization_range\": [" + std::to_string(CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X)
                    + ", " + std::to_string(CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y)
                    + ", " + std::to_string(CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z) + "]"
                    + ", \"camera_mount_rotation_randomization_range\": [" + std::to_string(CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X)
                    + ", " + std::to_string(CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y)
                    + ", " + std::to_string(CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z) + "]}";
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
                    for(TI member_i = 0; member_i < NUMBER_OF_ENVIRONMENTS; member_i++){
                        build_frame_stacked_with_target_from_dataset_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                            rlt::data(env.environments[member_i].history),
                            rlt::data(gpu_all_target_observations),
                            gpu_episode_start_step_per_row,
                            rlt::data(gpu_combined_batch),
                            OBSERVATION_DIM, IMG_C, FRAME_STACK_N, FRAME_STACK_STRIDE, COMBINED_IMG_C, COMBINED_OBS_DIM,
                            frame_step_start, frame_step_start + STEPS_PER_ENV, env.environments[member_i].history_step, (int)EXAMPLE_ROW_OFFSET, (int)N_EXAMPLES, (int)N_ENVIRONMENTS,
                            (int)(member_i * TASK_WORLD::INSTANCES), (int)TASK_WORLD::INSTANCES);
                    }
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
                    rlt::link_latest_artifact(device, latest_folder, checkpoint_path, step_folder, global_env_step);
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
                rlt::link_latest_artifact(device, latest_folder, reduced_checkpoint_path, step_folder, global_env_step);
                auto full_checkpoint_path = save_hdf5(rlt::utils::typing::integral_constant<TI, N_EXAMPLES>{});
                rlt::link_latest_artifact(device, latest_folder, full_checkpoint_path, step_folder, global_env_step);
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
                        rlt::link_latest_artifact(device, latest_folder, checkpoint_code_path, step_folder, global_env_step);
                    }
#endif
                    {
                        std::filesystem::path checkpoint_code_path = step_folder / "checkpoint.h";
                        std::ofstream f(checkpoint_code_path);
                        f << output_string;
                        f.close();
                        rlt::link_latest_artifact(device, latest_folder, checkpoint_code_path, step_folder, global_env_step);
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

    std::cout << "Training finished at env step " << global_env_step << std::endl;

    // ---------------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------------
    rlt::free(device, ppo);
    rlt::free(device, ppo_buffers);
    rlt::free(device, dataset);
    rlt::free(device, actor_optimizer);
    rlt::free(device, critic_optimizer);
    rlt::free(device, reward_log_action);
    rlt::free(device, reward_log_rng);
    rlt::free(device, rng);
    rlt::free(device_gpu, rng_gpu);

    rlt::free(device_gpu, ppo_gpu);
    rlt::free(device_gpu, actor_buffers);
    rlt::free(device_gpu, critic_buffers);
    rlt::free(device_gpu, critic_states_gae); rlt::free(device_gpu, critic_buffers_gae);
    rlt::free(device_gpu, dataset_gpu);
    rlt::free(device_gpu, rollout_actor_gpu);
    rlt::free(device_gpu, rollout_actor_buffers);
    rlt::free(device_gpu, actor_optimizer_gpu);
    rlt::free(device_gpu, critic_optimizer_gpu);
    rlt::free(device_gpu, gpu_all_target_observations);
    rlt::free(device_gpu, gpu_rollout_combined);
    rlt::free(device_gpu, gpu_combined_batch);
#ifdef RL_TOOLS_L2F_VISUAL_TRAINING_SMOKE
    rlt::free(device_gpu, observation_probes);
    rlt::free(device_gpu, observation_matches_gpu);
    rlt::free(device, observation_matches);
#endif
    rlt::free(device_gpu, gpu_all_state_observations);
    rlt::free(device_gpu, gpu_actions_eval);
    rlt::free(device_gpu, gpu_actions_train);
    rlt::free(device_gpu, gpu_d_action_train);
    rlt::free(device_gpu, gpu_critic_obs);
    rlt::free(device_gpu, gpu_d_critic_output);
    rlt::free(device_gpu, gpu_runner);
    rlt::free(device_gpu, gpu_runner_buffer);
    cudaFree(gpu_episode_start_step);
    cudaFree(gpu_episode_start_step_per_row);
    rlt::free(device_gpu, env);
    delete env_storage;

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
    rlt::free(device, device.logger);
#endif

    return 0;
}
