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
#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cuda.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>
#include <rl_tools/rendering/datasets/annotations/operations_cpu.h>

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

#include <rl_tools/rl/environments/l2f_visual/operations_generic_inertial_velocity.h>
#include <rl_tools/rl/environments/l2f_visual/operations_generic_level_camera.h>
#include <rl_tools/rl/environments/l2f_visual/operations_generic_position_hold.h>

#include <array>
#include <deque>
#include <chrono>
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

#include <rl_tools/nn_models/sequential/operations_generic_trainable_tail.h>

namespace rl_tools::rl::environments::l2f_visual::evaluation{
    template<typename T_BASE = mode::Evaluation<>, typename T_SPEC = bool>
    struct Teacher: T_BASE{
        using BASE = T_BASE;
        using SPEC = T_SPEC;
    };
    template<typename T_BASE = mode::Evaluation<>, typename T_SPEC = bool>
    struct Blind: T_BASE{
        using BASE = T_BASE;
        using SPEC = T_SPEC;
    };
    template<typename T_BASE = mode::Evaluation<>, typename T_SPEC = bool>
    struct Recovery: T_BASE{
        using BASE = T_BASE;
        using SPEC = T_SPEC;
    };
}
namespace hold_evaluation = rlt::rl::environments::l2f_visual::evaluation;
namespace rl_tools::mode::l2f_visual{
    template<typename T_BASE = Default<>, typename T_SPEC = bool>
    struct IndependentPose: T_BASE{
        using BASE = T_BASE;
        using SPEC = T_SPEC;
    };
}

#include "training_mixed_config.h"
using namespace rl_tools::rl::environments::l2f_visual::training_mixed;

namespace rl_tools{
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
        json += "{\"parameters\": " + rlt::json(device, env, parameters) + ",\n";
        json += "\"trajectory\": [";
        for(TI step_i = 0; step_i < max_len; step_i++){
            auto& s = (step_i < episode.size()) ? episode[step_i] : episode.back();
            json += "{\"state\":" + rlt::json(device, env, parameters, s.state) + ",";
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

}

namespace rl_tools{
    template<typename DEVICE>
    __global__ void geometric_teacher_kernel(DEVICE device, const ENVIRONMENT::Parameters*, const ENVIRONMENT::State* states, const T* observations, T* actions){
        const TI env_i = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        static_assert(STATE_OBS_DIM == 65 && ACTION_HISTORY_LENGTH == 8);
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
        const T gravity = INERTIAL_SPEC::GRAVITY;
        T action[4];
        action[0] = obs[6] - static_cast<T>(0.02) * (obs[40] - gravity + static_cast<T>(2) * obs[64] + static_cast<T>(2) * body_position[2]);
        action[1] = static_cast<T>(-1.5) * obs[1] + static_cast<T>(3) / gravity * (obs[63] + body_position[1]);
        action[2] = static_cast<T>(1.5) * obs[0] - static_cast<T>(3) / gravity * (obs[62] + body_position[0]);
        action[3] = -math::atan2(device.math, rotation[1][0], rotation[0][0]);
        for(TI axis = 0; axis < 4; axis++){
            actions[env_i * ACTION_DIM + axis] = math::clamp(device.math, action[axis], static_cast<T>(-1), static_cast<T>(1));
        }
    }
}
using rlt::geometric_teacher_kernel;

using COMPATIBILITY_OBSERVATION = obs::Position<obs::PositionSpecification<T, TI,
        obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
        obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
        obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI,
        obs::ActionHistory<obs::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
static constexpr T EPISODE_END_REASON_NONE = 0;
static constexpr T EPISODE_END_REASON_TERMINATED = 1;
static constexpr T EPISODE_END_REASON_TIME_LIMIT = 2;
static constexpr T EPISODE_END_REASON_SCENE_BOUNDARY = 3;

namespace rl_tools{

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

    template<bool ENABLE_MOTION_BLUR, typename DEVICE, typename RENDERER>
    void l2f_visual_set_active_scene_camera_open_buffer(DEVICE& device, void*& camera_open_buffer, RENDERER* renderer){
        if constexpr(ENABLE_MOTION_BLUR){
            camera_open_buffer = (void*)data(cameras_open(device, *renderer));
        }
    }

    template<typename DEVICE, typename OBS_PRIV_SPEC, typename STATE_OBS_SPEC, typename EPISODE_STAT_SPEC, typename RNG, typename MODE>
    __global__
    void l2f_visual_prologue_kernel(
        DEVICE device,
        ENVIRONMENT* envs, typename ENVIRONMENT::Parameters* env_params, typename ENVIRONMENT::State* states,
        bool* truncated_arr, TI* episode_step_arr, T* episode_return_arr,
        T* episode_end_reason_arr,
        bool* render_reset_arr,
        T* shutter_fraction_arr,
        Tensor<OBS_PRIV_SPEC> observations_privileged,
        Tensor<STATE_OBS_SPEC> state_observations,
        INERTIAL_STATE* inertial_states,
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
        T* positions_ptr, TI* num_positions_ptr, TI* env_scene_ptr, TI max_indoor_pos,
        TI* episode_start_step,
        RNG rng, TI step_i, TI frame_step_i, TI rng_offset, Mode<MODE>
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        TI rng_i = (env_i / N_ENVIRONMENTS_PER_SCENE) * N_ENVIRONMENTS_PER_SCENE * GRADIENT_ACCUMULATION_ROLLOUTS + rng_offset + env_i % N_ENVIRONMENTS_PER_SCENE;
        auto& rng_state = get(rng.states, 0, rng_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        TI pos = step_i * N_ENVIRONMENTS + env_i;
        bool reset_for_render = truncated_arr[env_i];
        if constexpr(mode::is<MODE, mode::l2f_visual::IndependentPose>){
            if(is_imitation_sample(device, MIXED_SPEC{}, env_i)){
                reset_for_render = true;
                episode_step_arr[env_i] = 0;
                episode_return_arr[env_i] = 0;
            }
        }
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
            if constexpr(mode::is<MODE, mode::Evaluation>){
                if constexpr(mode::is<MODE, hold_evaluation::Teacher>){
                    params.dynamics.mdp.init.guidance = 0;
                    params.dynamics.mdp.init.max_position = 0.2;
                    params.dynamics.mdp.init.max_linear_velocity = 0.2;
                }
                else{
                    params.dynamics.mdp.init.guidance = 1;
                }
            }
            if constexpr(mode::is<MODE, mode::l2f_visual::IndependentPose>){
                if(is_imitation_sample(device, MIXED_SPEC{}, env_i)){
                    params.dynamics.mdp.init.guidance = 0;
                    params.dynamics.mdp.init.max_position = static_cast<T>(0.03);
                    params.dynamics.mdp.init.max_angle = static_cast<T>(0.015);
                    params.dynamics.mdp.init.max_linear_velocity = 0;
                    params.dynamics.mdp.init.max_angular_velocity = 0;
                }
            }
            sample_initial_state(device, env, params, state, rng_state);
            if constexpr(mode::is<MODE, hold_evaluation::Recovery>){
                for(TI axis = 0; axis < 3; axis++){
                    state.position[axis] = random::uniform_real_distribution(device.random, static_cast<T>(-0.2), static_cast<T>(0.2), rng_state);
                    state.linear_velocity[axis] = random::uniform_real_distribution(device.random, static_cast<T>(-0.2), static_cast<T>(0.2), rng_state);
                }
            }
            TI scene_idx = env_scene_ptr[env_i];
            TI num_pos = num_positions_ptr[scene_idx];
            TI pos_idx = random::uniform_int_distribution(device.random, (TI)0, num_pos - 1, rng_state);
            T* p = positions_ptr + (scene_idx * max_indoor_pos + pos_idx) * INDOOR_POSITION_DIM;
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
        {
            auto obs_priv_slice = view(device, observations_privileged, env_i);
            auto obs_priv_flat = view_memory<tensor::Shape<TI, ENVIRONMENT::ObservationPrivileged::DIM>>(device, obs_priv_slice);
            auto obs_priv_matrix = matrix_view(device, obs_priv_flat);
            observe(device, env.dynamics, params.dynamics, state, typename ENVIRONMENT::ObservationPrivileged{}, obs_priv_matrix, rng_state);
        }
        {
            auto state_obs_slice = view(device, state_observations, env_i);
            auto state_obs_matrix = matrix_view(device, state_obs_slice);
            auto base_observations = view(device, state_obs_matrix, matrix::ViewSpec<1, ACTOR_STATE_OBS::DIM>{}, 0, 0);
            observe(device, env.dynamics, params.dynamics, state, ACTOR_STATE_OBS{}, base_observations, rng_state);
            if(reset_for_render){
                set(base_observations, 0, 40, static_cast<T>(9.81));
            }
            auto& inertial_state = inertial_states[env_i];
            if(reset_for_render){
                init(device, inertial_state, state.gyro);
            }
            else{
                T world_z_body[3];
                for(TI axis = 0; axis < 3; axis++){
                    world_z_body[axis] = get(base_observations, 0, axis);
                }
                step(device, inertial_state, world_z_body, state.accelerometer, state.gyro, params.dynamics.integration.dt);
            }
            for(TI axis = 0; axis < 3; axis++){
                set(state_obs_matrix, 0, ACTOR_STATE_OBS::DIM + axis, inertial_state.velocity[axis]);
            }
        }
        // Preserve the recorded seed sequence: the former teacher observation consumed noise draws.
        Matrix<matrix::Specification<T, TI, 1, COMPATIBILITY_OBSERVATION::DIM, false>> discarded_observation;
        observe(device, env.dynamics, params.dynamics, state, COMPATIBILITY_OBSERVATION{}, discarded_observation, rng_state);
    }

    template<typename DEVICE, typename ACTIONS_MEAN_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RUNNER_SPEC, typename RNG, typename MODE>
    __global__
    void l2f_visual_epilogue_kernel(
        DEVICE device,
        ENVIRONMENT* envs, typename ENVIRONMENT::Parameters* env_params, typename ENVIRONMENT::State* states,
        bool* truncated_arr, TI* episode_step_arr, T* episode_return_arr,
        T* episode_end_reason_arr,
        Matrix<ACTIONS_MEAN_SPEC> actions_mean,
        Matrix<ACTIONS_SPEC> actions,
        Matrix<ACTION_LOG_STD_SPEC> action_log_std,
        rl::components::on_policy_runner::Dataset<RUNNER_SPEC> dataset,
        const T* teacher_actions, TI optimizer_updates,
        RNG rng, TI step_i, TI episode_step_limit, TI rng_offset, Mode<MODE>
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        TI rng_i = (env_i / N_ENVIRONMENTS_PER_SCENE) * N_ENVIRONMENTS_PER_SCENE * GRADIENT_ACCUMULATION_ROLLOUTS + rng_offset + env_i % N_ENVIRONMENTS_PER_SCENE;
        auto& rng_state = get(rng.states, 0, rng_i);
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
            if constexpr(MIXED_TRAINING){
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
            }
            if constexpr(mode::is<MODE, mode::Evaluation>){
                action_noisy = action_mean;
            }
            if constexpr(mode::is<MODE, hold_evaluation::Teacher>){
                static_assert(MIXED_TRAINING, "Teacher evaluation requires the mixed target");
                action_noisy = teacher_actions[pos * ACTION_DIM + action_i];
            }
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
        if constexpr(MIXED_TRAINING && !mode::is<MODE, mode::Evaluation>){
            if(use_teacher_action(device, MIXED_SPEC{}, env_i, optimizer_updates)){
                episode_step_limit = MIXED_PARAMETERS::TEACHER_EPISODE_STEPS;
            }
        }
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

    __global__ void l2f_visual_force_scene_boundary_reset_kernel(bool* truncated_arr, TI* episode_step_arr, T* episode_end_reason_arr){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        if(episode_end_reason_arr[env_i] == EPISODE_END_REASON_NONE && episode_step_arr[env_i] > 0){
            episode_end_reason_arr[env_i] = EPISODE_END_REASON_SCENE_BOUNDARY;
        }
        truncated_arr[env_i] = true;
    }

    template<typename DEVICE, typename OBS_PRIV_SPEC, typename RNG>
    __global__
    void l2f_visual_final_priv_obs_kernel(
        DEVICE device,
        ENVIRONMENT* envs, typename ENVIRONMENT::Parameters* env_params, typename ENVIRONMENT::State* states,
        Tensor<OBS_PRIV_SPEC> observations_privileged,
        RNG rng, TI rng_offset
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        TI rng_i = (env_i / N_ENVIRONMENTS_PER_SCENE) * N_ENVIRONMENTS_PER_SCENE * GRADIENT_ACCUMULATION_ROLLOUTS + rng_offset + env_i % N_ENVIRONMENTS_PER_SCENE;
        auto& rng_state = get(rng.states, 0, rng_i);
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
    void l2f_visual_make_cameras_kernel(
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
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
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
    void l2f_visual_make_target_cameras_kernel(
        DEVICE device,
        typename ENVIRONMENT::Parameters* env_params,
        CAMERA_DATA* target_cameras,
        T aspect,
        const T* target_frame_roll_arr,
        const T* target_frame_pitch_arr,
        T* scene_translation_arr, T* scene_yaw_cos_arr, T* scene_yaw_sin_arr
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS){
            return;
        }
        const auto& params = env_params[env_i];
        target_cameras[env_i] = rl::environments::l2f_visual::cuda::make_target_camera<DEVICE, VISUAL_SPEC>(
            device, params, aspect,
            scene_translation_arr + env_i * 3,
            scene_yaw_cos_arr[env_i], scene_yaw_sin_arr[env_i],
            target_frame_roll_arr[env_i], target_frame_pitch_arr[env_i]
        );
    }
}

namespace rl_tools{
template <bool APPLY_BRIGHTNESS>
__global__ void scatter_pixel_to_float_kernel(
    const float* __restrict__ observation, float* __restrict__ output,
    const float* __restrict__ brightness_scales,
    int base_env, int n_envs,
    int pixels_per_camera, int obs_dim
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_envs * pixels_per_camera;
    if(tid >= total){
        return;
    }
    int local_env = tid / pixels_per_camera;
    int pixel_idx = tid % pixels_per_camera;
    int in_base = (local_env * pixels_per_camera + pixel_idx) * 3;
    float r = observation[in_base + 0];
    float g = observation[in_base + 1];
    float b = observation[in_base + 2];
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

}

namespace rl_tools{
    template<typename DEVICE>
    __global__ void init_level_cameras_kernel(DEVICE device, const ENVIRONMENT::Parameters* parameters, const T* observations, LEVEL_CAMERA* cameras){
        const TI sample = blockIdx.x * blockDim.x + threadIdx.x;
        if(sample >= N_ENVIRONMENTS){
            return;
        }
        const auto& calibration = parameters[sample];
        init(device, cameras[sample], observations + sample * STATE_OBS_DIM, calibration.camera_mount.forward_body, calibration.camera_mount.up_body, calibration.fov);
    }
    template<typename DEVICE>
    __global__ void rectify_images_kernel(DEVICE device, const LEVEL_CAMERA* cameras, const T* input, T* output){
        const TI index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= N_ENVIRONMENTS * CAM_HEIGHT * CAM_WIDTH){
            return;
        }
        const TI sample = index / (CAM_HEIGHT * CAM_WIDTH);
        const TI pixel = index % (CAM_HEIGHT * CAM_WIDTH);
        T source_x, source_y;
        project(device, cameras[sample], static_cast<T>(pixel % CAM_WIDTH), static_cast<T>(pixel / CAM_WIDTH), source_x, source_y);
        source_x = math::clamp(device.math, source_x, static_cast<T>(0), static_cast<T>(CAM_WIDTH - 1));
        source_y = math::clamp(device.math, source_y, static_cast<T>(0), static_cast<T>(CAM_HEIGHT - 1));
        const TI ix = static_cast<TI>(math::min(device.math, source_x, static_cast<T>(CAM_WIDTH - 2)));
        const TI iy = static_cast<TI>(math::min(device.math, source_y, static_cast<T>(CAM_HEIGHT - 2)));
        const T fx = source_x - static_cast<T>(ix);
        const T fy = source_y - static_cast<T>(iy);
        for(TI channel = 0; channel < 3; channel++){
            T color = 0;
            for(TI dy = 0; dy < 2; dy++){
                for(TI dx = 0; dx < 2; dx++){
                    const T weight = (dx == 0 ? 1 - fx : fx) * (dy == 0 ? 1 - fy : fy);
                    color += weight * input[(sample * CAM_HEIGHT * CAM_WIDTH + (iy + dy) * CAM_WIDTH + ix + dx) * 3 + channel];
                }
            }
            output[index * 3 + channel] = color;
        }
    }

}

namespace rl_tools{
__global__ void build_frame_stacked_with_target_from_history_kernel(
    const float* __restrict__ history_obs,
    const float* __restrict__ target_obs,
    const TI* __restrict__ episode_start_step,
    TI step_i,
    float* __restrict__ combined_out,
    int obs_dim, int img_c, int n_frames, int frame_stride, int combined_img_c, int combined_obs_dim, int num_envs
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx >= num_envs * combined_obs_dim){
        return;
    }
    int sample = global_idx / combined_obs_dim;
    int offset = global_idx % combined_obs_dim;
    const int paired_pixel = offset / combined_img_c;
    const bool reversed = paired_pixel % (2 * CAM_WIDTH) >= CAM_WIDTH;
    int pixel = paired_pixel / (2 * CAM_WIDTH) * CAM_WIDTH + paired_pixel % CAM_WIDTH;
    int frame_channel = offset % combined_img_c;
    if(reversed && frame_channel < 6){
        frame_channel = (frame_channel + 3) % 6;
    }
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

__global__ void build_frame_stacked_with_target_from_dataset_kernel(
    const float* __restrict__ history_obs,
    const float* __restrict__ all_target_obs,
    const TI* __restrict__ episode_start_step_per_row,
    float* __restrict__ combined_out,
    int obs_dim, int img_c, int n_frames, int frame_stride, int combined_img_c, int combined_obs_dim,
    TI frame_step_start, int batch_offset, int batch_size, int n_envs
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx >= batch_size * combined_obs_dim){
        return;
    }
    int sample = global_idx / combined_obs_dim;
    int offset = global_idx % combined_obs_dim;
    const int paired_pixel = offset / combined_img_c;
    const bool reversed = paired_pixel % (2 * CAM_WIDTH) >= CAM_WIDTH;
    int pixel = paired_pixel / (2 * CAM_WIDTH) * CAM_WIDTH + paired_pixel % CAM_WIDTH;
    int frame_channel = offset % combined_img_c;
    if(reversed && frame_channel < 6){
        frame_channel = (frame_channel + 3) % 6;
    }
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
        if(desired_step < episode_start){
            desired_step = episode_start;
        }
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
    if(env_i >= N_ENVIRONMENTS){
        return;
    }
    episode_start_step_per_row[step_i * N_ENVIRONMENTS + env_i] = episode_start_step[env_i];
}

}

static bool parse_hex_hash(const char* hex, unsigned char* out, unsigned len){
    for(unsigned i = 0; i < len; i++){
        unsigned byte = 0;
        for(int nibble = 0; nibble < 2; nibble++){
            char c = hex[i * 2 + nibble];
            if(c >= '0' && c <= '9'){
                byte = (byte << 4) | (c - '0');
            }
            else if(c >= 'a' && c <= 'f'){
                byte = (byte << 4) | (c - 'a' + 10);
            }
            else if(c >= 'A' && c <= 'F'){
                byte = (byte << 4) | (c - 'A' + 10);
            }
            else{
                return false;
            }
        }
        out[i] = static_cast<unsigned char>(byte);
    }
    return true;
}

template<typename MODE>
int run(int argc, char** argv, rlt::Mode<MODE> mode){
    using namespace rlt;
    constexpr bool EVALUATION = rlt::mode::is<MODE, rlt::mode::Evaluation>;
    TI seed = 0;
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <scene_directory or scene.glb> [seed] [evaluation_checkpoint.h5|teacher [seconds [blind|recovery]]]" << std::endl;
        return 1;
    }
    if(argc > 2){
        seed = std::atoi(argv[2]);
    }
    TI evaluation_steps = 600 * SIMULATION_FREQUENCY;
    if constexpr(EVALUATION){
        if(argc > 4){
            const double seconds = std::atof(argv[4]);
            if(!std::isfinite(seconds) || seconds <= 0 || seconds > 86400){
                std::cerr << "Evaluation duration must be in (0, 86400] seconds" << std::endl;
                return 1;
            }
            evaluation_steps = static_cast<TI>(seconds * SIMULATION_FREQUENCY);
        }
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

    DEVICE device;
    DEVICE_GPU device_gpu;
    init(device);

    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = "l2f_visual_patch64_down45_fov110_aa2train_aa4eval";
    if constexpr(EVALUATION){
        extrack_config.name += "_evaluation";
    }
    extrack_config.population_variates = "algorithm_environments_accumulation";
    extrack_config.population_values = std::string(MIXED_TRAINING ? "mixed_" : "on-policy_") + std::to_string(N_ENVIRONMENTS) + "_" + std::to_string(GRADIENT_ACCUMULATION_ROLLOUTS);
    init(device, extrack_config, extrack_paths, seed);
    {
        nlohmann::json camera_setup;
        camera_setup["nominal_pitch_degrees"] = CAMERA_PITCH * 180 / rlt::math::PI<T>;
        camera_setup["nominal_fov_degrees"] = CAMERA_FOV;
        camera_setup["fov_randomization_degrees"] = CAMERA_FOV_RANDOMIZATION_RANGE;
        camera_setup["width"] = CAM_WIDTH;
        camera_setup["height"] = CAM_HEIGHT;
        camera_setup["observer_time_constant_seconds"] = INERTIAL_SPEC::TIME_CONSTANT;
        camera_setup["anti_aliasing_grid_size"] = RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : static_cast<TI>(1);
        camera_setup["training_default_anti_aliasing_grid_size"] = RENDER_ENABLE_ANTI_ALIASING ? RENDER_ANTI_ALIASING_GRID_SIZE_TRAINING : static_cast<TI>(1);
        camera_setup["evaluation_default_anti_aliasing_grid_size"] = RENDER_ENABLE_ANTI_ALIASING ? RENDER_ANTI_ALIASING_GRID_SIZE_EVALUATION : static_cast<TI>(1);
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

    using RENDERER_SPEC = typename ENVIRONMENT::SPEC::RENDERER_SPEC;
    using RENDERER_TYPE = rlt::rendering::raytracing::Renderer<RENDERER_SPEC>;
    using ANNOTATIONS_TYPE = rlt::rendering::datasets::annotations::FreeSpace<typename ENVIRONMENT::SPEC::ANNOTATIONS_SPEC>;
    using LIBRARY_TYPE = rlt::rendering::raytracing::AssetLibrary<RENDERER_SPEC>;

    std::vector<ENVIRONMENT> envs(N_ENVIRONMENTS);
    std::vector<typename ENVIRONMENT::Parameters> env_parameters(N_ENVIRONMENTS);

    auto* library = new LIBRARY_TYPE{};
    malloc(device, *library);
    auto* renderer_storage = new std::array<RENDERER_TYPE, N_TOTAL_SCENES>{};
    std::array<RENDERER_TYPE*, N_TOTAL_SCENES> renderers{};
    std::deque<ANNOTATIONS_TYPE> procthor_annotations;
    std::array<ANNOTATIONS_TYPE*, N_TOTAL_SCENES> annotations{};
    {
        for(TI s = 0; s < N_TOTAL_SCENES; s++){
            std::cout << "Loading scene [" << s << "]: " << std::filesystem::path(scene_paths[s]).filename().string() << std::flush;
            renderers[s] = &(*renderer_storage)[s];
            malloc(device, *renderers[s], *library);
            rlt::rendering::Bundle<T> bundle;
            load<typename RENDERER_SPEC::SHADING, RENDERER_SPEC::HAS_RGB>(device, bundle, scene_paths[s]);
            TI scene_id = insert(device, *library, bundle);
            init(device, *renderers[s], *library, scene_id);
            generate_probe_directions(device, *renderers[s]);
            if(scene_id == (TI)procthor_annotations.size()){
                procthor_annotations.emplace_back();
                rlt::rendering::datasets::annotations::FreeSpaceParameters<T, TI> free_space_parameters{};
                rlt::rendering::datasets::annotations::annotate(device, procthor_annotations[scene_id], bundle.metadata, *renderers[s], free_space_parameters);
            }
            TI num_pos = procthor_annotations[scene_id].num_positions;
            std::cout << " — " << num_pos << " indoor positions" << std::endl;
            if(num_pos == 0){
                std::cerr << "Scene has no valid positions with 1m clearance: " << scene_paths[s] << std::endl;
                return 1;
            }
            annotations[s] = &procthor_annotations[scene_id];
        }
        std::cout << "Loaded " << N_TOTAL_SCENES << " scenes" << std::endl;
    }

    static constexpr TI MAX_INDOOR_POS = 256;
    T* gpu_positions = nullptr;
    TI* gpu_num_positions = nullptr;
    TI* gpu_env_scene = nullptr;
    cudaMalloc(&gpu_positions, N_TOTAL_SCENES * MAX_INDOOR_POS * INDOOR_POSITION_DIM * sizeof(T));
    cudaMalloc(&gpu_num_positions, N_TOTAL_SCENES * sizeof(TI));
    cudaMalloc(&gpu_env_scene, N_ENVIRONMENTS * sizeof(TI));
    {
        std::vector<T> all_positions(N_TOTAL_SCENES * MAX_INDOOR_POS * INDOOR_POSITION_DIM, 0);
        std::vector<TI> all_counts(N_TOTAL_SCENES);
        for(TI s = 0; s < N_TOTAL_SCENES; s++){
            all_counts[s] = annotations[s]->num_positions;
            for(TI i = 0; i < all_counts[s]; i++){
                TI base = (s * MAX_INDOOR_POS + i) * INDOOR_POSITION_DIM;
                all_positions[base + 0] = annotations[s]->positions[i].position[0];
                all_positions[base + 1] = annotations[s]->positions[i].position[1];
                all_positions[base + 2] = annotations[s]->positions[i].position[2];
            }
        }
        cudaMemcpy(gpu_positions, all_positions.data(), all_positions.size() * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_num_positions, all_counts.data(), N_TOTAL_SCENES * sizeof(TI), cudaMemcpyHostToDevice);
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

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        TI scene_i = env_i / N_ENVIRONMENTS_PER_SCENE;
        malloc(device, envs[env_i].dynamics);
        envs[env_i].renderer = renderers[scene_i];
        envs[env_i].annotations = annotations[scene_i];
        envs[env_i].use_target_mode = true;
        envs[env_i].parameters.fov = CAMERA_FOV;
        const T nominal_camera_pitch_axis[3] = {0, 1, 0};
        rotate_camera_mount(device, envs[env_i].parameters.camera_mount, nominal_camera_pitch_axis, CAMERA_PITCH);
        envs[env_i].parameters.camera_randomization.fov_range = CAMERA_FOV_RANDOMIZATION_RANGE;
        envs[env_i].parameters.camera_randomization.offset_body_range[0] = CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X;
        envs[env_i].parameters.camera_randomization.offset_body_range[1] = CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y;
        envs[env_i].parameters.camera_randomization.offset_body_range[2] = CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z;
        envs[env_i].parameters.camera_randomization.rotation_body_range[0] = CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X;
        envs[env_i].parameters.camera_randomization.rotation_body_range[1] = CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y;
        envs[env_i].parameters.camera_randomization.rotation_body_range[2] = CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z;
        initial_parameters(device, envs[env_i], env_parameters[env_i]);
        env_parameters[env_i].scene_translation[0] = 0;
        env_parameters[env_i].scene_translation[1] = 0;
        env_parameters[env_i].scene_translation[2] = 0;
    }

    {
        std::string ui = get_ui(device, envs[0].dynamics);
        if(!ui.empty()){
            std::filesystem::create_directories(extrack_paths.seed);
            std::ofstream ui_file(extrack_paths.seed / "ui.esm.js");
            ui_file << ui;
        }
    }

    init(device_gpu);
    RNG_GPU rng_gpu;
    malloc(device_gpu, rng_gpu);
    init(device_gpu, rng_gpu, seed);
    init(device, ppo, actor_optimizer, critic_optimizer, rng);
    init(device, ppo.actor, rlt::rl::environments::l2f_visual::position_hold::Specification<TYPE_POLICY, TI>{});
    if constexpr(EVALUATION && !rlt::mode::is<MODE, hold_evaluation::Teacher>){
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
        using VALIDATION_ACTOR = typename CHECKPOINT_ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, REDUCED_BATCH_SIZE>;
        VALIDATION_ACTOR checkpoint_actor;
        malloc(device, checkpoint_actor);
        rlt::persist::backends::hdf5::File file(argv[3], rlt::persist::backends::hdf5::Mode::READ);
        auto actor_group = get_group(device, file, "actor");
        rlt::utils::assert_exit(device, load(device, checkpoint_actor, actor_group), "Could not load evaluation actor");
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, REDUCED_BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>>> example_image;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, REDUCED_BATCH_SIZE, STATE_OBS_DIM>>> example_state;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, REDUCED_BATCH_SIZE, ACTION_DIM>>> expected_actions, actual_actions;
        typename VALIDATION_ACTOR::template Buffer<true> validation_buffer;
        malloc(device, example_image);
        malloc(device, example_state);
        malloc(device, expected_actions);
        malloc(device, actual_actions);
        malloc(device, validation_buffer);
        auto inputs_group = get_group(device, file, "example/inputs");
        auto outputs_group = get_group(device, file, "example/outputs");
        rlt::utils::assert_exit(device, load(device, example_image, inputs_group, "0") && load(device, example_state, inputs_group, "1") && load(device, expected_actions, outputs_group, "0"), "Evaluation requires the checkpoint with two canonical examples");
        auto example_inputs = rlt::nn_models::parallel::pack_inputs(example_image, example_state);
        evaluate(device, checkpoint_actor, example_inputs, actual_actions, validation_buffer, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
        auto check_examples = [&](const char* backend, T tolerance){
            T example_error = 0;
            for(TI index = 0; index < REDUCED_BATCH_SIZE * ACTION_DIM; index++){
                const T actual = data(actual_actions)[index];
                const T expected = data(expected_actions)[index];
                std::uint32_t actual_bits, expected_bits;
                std::memcpy(&actual_bits, &actual, sizeof(actual_bits));
                std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
                rlt::utils::assert_exit(device, (actual_bits & 0x7f800000u) != 0x7f800000u && (expected_bits & 0x7f800000u) != 0x7f800000u, "Non-finite checkpoint example");
                example_error = std::max(example_error, std::abs(actual - expected));
            }
            std::cout << "Checkpoint " << backend << " example maximum error: " << example_error << std::endl;
            rlt::utils::assert_exit(device, example_error < tolerance, "Checkpoint examples disagree with this evaluator's model configuration");
            return example_error;
        };
        check_examples("CPU", 1e-4f);
        VALIDATION_ACTOR gpu_validation_actor;
        typename VALIDATION_ACTOR::template Buffer<true> gpu_validation_buffer;
        decltype(example_image) gpu_example_image;
        decltype(example_state) gpu_example_state;
        decltype(actual_actions) gpu_example_actions;
        malloc(device_gpu, gpu_validation_actor);
        malloc(device_gpu, gpu_validation_buffer);
        malloc(device_gpu, gpu_example_image);
        malloc(device_gpu, gpu_example_state);
        malloc(device_gpu, gpu_example_actions);
        copy(device, device_gpu, checkpoint_actor, gpu_validation_actor);
        copy(device, device_gpu, example_image, gpu_example_image);
        copy(device, device_gpu, example_state, gpu_example_state);
        auto gpu_example_inputs = rlt::nn_models::parallel::pack_inputs(gpu_example_image, gpu_example_state);
        evaluate(device_gpu, gpu_validation_actor, gpu_example_inputs, gpu_example_actions, gpu_validation_buffer, rng_gpu, rlt::Mode<rlt::mode::Evaluation<>>{});
        copy(device_gpu, device, gpu_example_actions, actual_actions);
        const T gpu_example_error = check_examples("GPU", 1e-3f);
        metra::log("l2f_visual_hold/checkpoint_gpu_max_abs_error", gpu_example_error);
        free(device_gpu, gpu_example_actions);
        free(device_gpu, gpu_example_state);
        free(device_gpu, gpu_example_image);
        free(device_gpu, gpu_validation_buffer);
        free(device_gpu, gpu_validation_actor);
        free(device, validation_buffer);
        free(device, actual_actions);
        free(device, expected_actions);
        free(device, example_state);
        free(device, example_image);
        copy(device, device, checkpoint_actor, ppo.actor);
        free(device, checkpoint_actor);
#else
        std::cerr << "Checkpoint evaluation requires HDF5" << std::endl;
        return 1;
#endif
    }
    TI environment_step = 0;
    if constexpr(!EVALUATION){
        // Preserve the CPU RNG sequence after removing the unused 4-to-15 auxiliary Dense initializer.
        for(TI draw = 0; draw < 75; draw++){
            random::uniform_real_distribution(device.random, static_cast<T>(-1), static_cast<T>(1), rng);
        }
    }
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, ACTION_DIM>>> teacher_actions_gpu, teacher_actions;
    malloc(device_gpu, teacher_actions_gpu);
    malloc(device, teacher_actions);

    PPO_TYPE ppo_gpu;
    ACTOR_BUFFERS actor_buffers;
    CRITIC_BUFFERS critic_buffers;
    rlt::rl::components::on_policy_runner::ValueState<typename PPO_SPEC::CRITIC_TYPE, ON_POLICY_RUNNER_DATASET_SPEC> critic_states_gae;
    rlt::rl::components::on_policy_runner::ValueBuffer<typename PPO_SPEC::CRITIC_TYPE, ON_POLICY_RUNNER_DATASET_SPEC> critic_buffers_gae;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, OBS_PRIV_DIM>>> next_privileged_observations;
    ON_POLICY_RUNNER_DATASET_TYPE dataset_gpu;
    malloc(device_gpu, ppo_gpu);
    malloc(device_gpu, actor_buffers);
    malloc(device_gpu, critic_buffers);
    malloc(device_gpu, critic_states_gae);
    malloc(device_gpu, critic_buffers_gae);
    if constexpr(ON_POLICY_RUNNER_SPEC::COLLECT_NEXT_OBSERVATIONS){
        malloc(device_gpu, next_privileged_observations);
    }
    malloc(device_gpu, dataset_gpu);

    ROLLOUT_ACTOR_TYPE rollout_actor_gpu;
    ROLLOUT_ACTOR_BUFFERS rollout_actor_buffers;
    malloc(device_gpu, rollout_actor_gpu);
    malloc(device_gpu, rollout_actor_buffers);

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

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, OBSERVATION_DIM>>> gpu_all_target_observations;
    malloc(device_gpu, gpu_all_target_observations);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, FRAME_STACK_HISTORY_LENGTH * N_ENVIRONMENTS, OBSERVATION_DIM>>> gpu_frame_stack_history;
    malloc(device_gpu, gpu_frame_stack_history);
    Tensor<tensor::Specification<LEVEL_CAMERA, TI, tensor::Shape<TI, N_ENVIRONMENTS>>> level_cameras_gpu;
    malloc(device_gpu, level_cameras_gpu);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, COMBINED_OBS_DIM>>> gpu_rollout_combined;
    malloc(device_gpu, gpu_rollout_combined);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, COMBINED_OBS_DIM>>> gpu_combined_batch;
    malloc(device_gpu, gpu_combined_batch);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, STATE_OBS_DIM>>> gpu_all_state_observations;
    malloc(device_gpu, gpu_all_state_observations);
    rlt::Tensor<rlt::tensor::Specification<INERTIAL_STATE, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS>>> inertial_states_gpu;
    malloc(device_gpu, inertial_states_gpu);

    rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ACTION_DIM>> gpu_actions_eval;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_actions_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_d_action_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, OBS_PRIV_DIM>> gpu_critic_obs;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, 1>> gpu_d_critic_output;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> gpu_episode_lengths_log;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> gpu_episode_returns_log;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> gpu_episode_end_reasons_log;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> cpu_episode_lengths_log;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> cpu_episode_returns_log;
    rlt::Matrix<rlt::matrix::Specification<T, TI, STEPS_TOTAL, 1>> cpu_episode_end_reasons_log;
    malloc(device_gpu, gpu_actions_eval);
    malloc(device_gpu, gpu_actions_train);
    malloc(device_gpu, gpu_d_action_train);
    malloc(device_gpu, gpu_critic_obs);
    malloc(device_gpu, gpu_d_critic_output);
    malloc(device_gpu, gpu_episode_lengths_log);
    malloc(device_gpu, gpu_episode_returns_log);
    malloc(device_gpu, gpu_episode_end_reasons_log);
    malloc(device, cpu_episode_lengths_log);
    malloc(device, cpu_episode_returns_log);
    malloc(device, cpu_episode_end_reasons_log);

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
    malloc(device, reward_log_action);

    std::vector<typename ENVIRONMENT::State> trajectory_states(STEPS_PER_ENV * TRAJECTORY_NUM_ENVS);

    static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    static constexpr TI VIDEO_FRAME_STRIDE = 4;
    static_assert(SIMULATION_FREQUENCY % VIDEO_FRAME_STRIDE == 0);
    static constexpr TI MOSAIC_W = SCENE_GRID_COLS * ENV_GRID_SIDE * CAM_WIDTH * 2;
    static constexpr TI MOSAIC_H = SCENE_GRID_ROWS * ENV_GRID_SIDE * CAM_HEIGHT;
    std::vector<uint8_t> mosaic_frame(MOSAIC_W * MOSAIC_H * 3);
    std::vector<float> cpu_obs_video(N_ENVIRONMENTS * OBSERVATION_DIM);
    std::vector<float> cpu_target_obs_video(N_ENVIRONMENTS * OBSERVATION_DIM);

    std::cout << "Starting single-use on-policy training (visual L2F target navigation, CUDA)" << std::endl;
    std::cout << "  N_ENVIRONMENTS:   " << N_ENVIRONMENTS << std::endl;
    std::cout << "  STEPS_PER_ENV:    " << STEPS_PER_ENV << std::endl;
    std::cout << "  SCENE_SET_STEPS:  " << STEPS_PER_ENV * ROLLOUTS_PER_SCENE_SET << std::endl;
    std::cout << "  STEPS_TOTAL:      " << STEPS_TOTAL << std::endl;
    std::cout << "  BATCH_SIZE:       " << BATCH_SIZE << std::endl;
    if constexpr(MIXED_TRAINING){
        std::cout << "  MIXED BATCH:      " << MIXED_SPEC::IMITATION_BATCH_SIZE << " IL + " << RL_BATCH_SIZE << " RL" << std::endl;
        std::cout << "  TEACHER WARMUP:   " << MIXED_PARAMETERS::TEACHER_FORCING_UPDATES << " optimizer updates (IL environments only)" << std::endl;
        using BALANCING_MODE = typename MIXED_PARAMETERS::BALANCING_MODE::MODE;
        std::cout << "  IL BALANCING:     " << (rlt::mode::is<BALANCING_MODE, mixed_imitation::FixedNormRatio> ? "output-gradient norm ratio" : "fixed weight") << std::endl;
        auto log_config = [&](const char* key, T value){
            add_scalar(device, device.logger, (std::string("mixed/config/") + key).c_str(), value);
            metra::log(std::string("l2f_visual_training_mixed/config/") + key, static_cast<double>(value));
        };
        log_config("nominal_camera_pitch_degrees", CAMERA_PITCH * 180 / rlt::math::PI<T>);
        log_config("nominal_camera_fov_degrees", CAMERA_FOV);
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
    static constexpr TI N_PPO_STEPS = ((LOOP_CORE_PARAMETERS::STEP_LIMIT + GRADIENT_ACCUMULATION_ROLLOUTS - 1) / GRADIENT_ACCUMULATION_ROLLOUTS) * GRADIENT_ACCUMULATION_ROLLOUTS;
    TI accumulated_samples = 0;
    TI optimizer_updates = 0;
    std::array<std::array<TI, N_ENVIRONMENTS>, GRADIENT_ACCUMULATION_ROLLOUTS> group_episode_steps{};
    std::array<std::array<T, N_ENVIRONMENTS>, GRADIENT_ACCUMULATION_ROLLOUTS> group_episode_returns{};
    std::array<std::array<T, N_ENVIRONMENTS>, GRADIENT_ACCUMULATION_ROLLOUTS> group_episode_end_reasons{};

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
            int video_status = pclose(ffmpeg_pipe);
            ffmpeg_pipe = nullptr;
            if(video_status != 0){
                throw std::runtime_error("Video encoder failed");
            }
            if(!current_video_path.empty()){
                const auto temporary_path = current_video_path.parent_path() / "video.partial.mp4";
                if(!std::filesystem::is_regular_file(temporary_path) || std::filesystem::file_size(temporary_path) < 1024){
                    throw std::runtime_error("Video encoder produced an empty or invalid artifact");
                }
                std::filesystem::rename(temporary_path, current_video_path);
                auto latest_folder = get_latest_folder(device, extrack_paths);
                link_latest_artifact(device, latest_folder, current_video_path, current_video_path.parent_path(), current_video_step);
                current_video_path.clear();
                current_video_step = 0;
            }
            ffmpeg_pipe = nullptr;
            record_video_scene_set = false;
        }
    };

    std::array<TI, N_ENVIRONMENTS> first_failure_step;
    first_failure_step.fill(evaluation_steps + 1);
    std::array<double, N_ENVIRONMENTS> position_squared_sum{}, speed_squared_sum{}, position_max{};
    std::array<std::array<double, 3>, N_ENVIRONMENTS> position_sum{}, initial_position_sum{}, final_position_sum{};
    std::array<TI, N_ENVIRONMENTS> initial_window_samples{}, final_window_samples{};
    std::array<TI, N_ENVIRONMENTS> evaluation_samples{};
    std::ofstream evaluation_trace;
    if constexpr(EVALUATION){
        evaluation_trace.open(extrack_paths.seed / "evaluation.csv");
        evaluation_trace << "time,environment,x,y,z,r00,r01,r02,r10,r11,r12,r20,r21,r22,vx,vy,vz,wx,wy,wz,action0,action1,action2,action3,terminated\n";
    }
    const TI run_steps = EVALUATION ? (evaluation_steps + STEPS_PER_ENV - 1) / STEPS_PER_ENV : N_PPO_STEPS;
    for(TI ppo_step_i = 0; ppo_step_i < run_steps; ppo_step_i++){
        auto step_start = std::chrono::high_resolution_clock::now();
        TI accumulation_i = ppo_step_i % GRADIENT_ACCUMULATION_ROLLOUTS;
        TI rng_offset = accumulation_i * N_ENVIRONMENTS_PER_SCENE;
        bool update_ready = accumulation_i + 1 == GRADIENT_ACCUMULATION_ROLLOUTS;
        if(accumulation_i == 0){
            zero_gradient(device_gpu, ppo_gpu.actor);
            zero_gradient(device_gpu, ppo_gpu.critic);

            accumulated_samples = 0;
        }
        cudaMemcpy(gpu_episode_step_arr, group_episode_steps[accumulation_i].data(), N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_return_arr, group_episode_returns[accumulation_i].data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_end_reason_arr, group_episode_end_reasons[accumulation_i].data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
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
        bool scene_set_boundary = EVALUATION ? ppo_step_i == 0 : ppo_step_i % ROLLOUTS_PER_SCENE_SET == 0;
        bool scene_set_end = rollout_in_scene_set + 1 == ROLLOUTS_PER_SCENE_SET;
        bool save_extrack_step = !EVALUATION && scene_set_end && scene_set_i % CHECKPOINT_CADENCE_SCENE_SETS == 0;
        bool log_reward_components_this_step = ppo_step_i % REWARD_COMPONENT_LOG_INTERVAL_PPO_STEPS == 0;
        if(scene_set_boundary){
            close_video_pipe();
            l2f_visual_force_scene_boundary_reset_kernel<<<grid, block, 0, device_gpu.stream>>>(gpu_truncated_arr, gpu_episode_step_arr, gpu_episode_end_reason_arr);
            check_status(device_gpu);
            cudaStreamSynchronize(device_gpu.stream);
            check_status(device_gpu);
            if(accumulation_i == 0){
                std::shuffle(scene_permutation.begin(), scene_permutation.end(), scene_rng);
                for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                    active_scene_indices[active_scene_i] = scene_permutation[active_scene_i];
                }
                upload_active_scenes();
            }
        }

        std::array<RENDERER_TYPE*, N_ACTIVE_SCENES> active_renderers{};
        std::array<cudaStream_t, N_ACTIVE_SCENES> active_scene_render_streams{};
        std::array<void*, N_ACTIVE_SCENES> active_scene_camera_buffers{};
        std::array<void*, N_ACTIVE_SCENES> active_scene_camera_open_buffers{};
        std::array<const float*, N_ACTIVE_SCENES> active_scene_observation_ptrs{};
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            TI actual_scene_i = active_scene_indices[active_scene_i];
            auto* renderer = renderers[actual_scene_i];
            active_renderers[active_scene_i] = renderer;
            active_scene_render_streams[active_scene_i] = stream(device, *renderer);
            active_scene_camera_buffers[active_scene_i] = (void*)data(cameras(device, *renderer));
            l2f_visual_set_active_scene_camera_open_buffer<RENDER_MOTION_BLUR_ACTIVE>(device, active_scene_camera_open_buffers[active_scene_i], renderer);
            active_scene_observation_ptrs[active_scene_i] = data(observation(device, *renderer));
        }

        {
            std::vector<unsigned char> truncated_host(N_ENVIRONMENTS);
            cudaMemcpy(truncated_host.data(), gpu_truncated_arr, N_ENVIRONMENTS * sizeof(bool), cudaMemcpyDeviceToHost);
            std::vector<T> reset_host(N_ENVIRONMENTS);
            for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                reset_host[env_i] = truncated_host[env_i] ? (T)1 : (T)0;
            }
            cudaMemcpy(dataset_gpu.reset._data, reset_host.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        }

        if(scene_set_boundary){
            record_video_scene_set = EVALUATION && !rlt::mode::is<MODE, hold_evaluation::Recovery> && !rlt::mode::is<MODE, hold_evaluation::Teacher>;
        }
        if(record_video_scene_set && ffmpeg_pipe == nullptr){
            TI video_step = EVALUATION ? evaluation_steps * N_ENVIRONMENTS : environment_step + N_ENVIRONMENTS * STEPS_PER_ENV * ROLLOUTS_PER_SCENE_SET;
            auto step_folder = get_step_folder(device, extrack_config, extrack_paths, video_step);
            std::filesystem::create_directories(step_folder);
            auto video_path = step_folder / "video.mp4";
            current_video_path = video_path;
            current_video_step = video_step;
            const auto temporary_path = step_folder / "video.partial.mp4";
            std::ostringstream ffmpeg_cmd;
            ffmpeg_cmd << "ffmpeg -y -threads 1 -filter_threads 1 -f rawvideo -pixel_format rgb24 -video_size " << MOSAIC_W << "x" << MOSAIC_H << " -framerate " << SIMULATION_FREQUENCY / VIDEO_FRAME_STRIDE << " -i - -c:v libx264 -threads 1 -pix_fmt yuv420p -crf 18 -preset fast -movflags +faststart -loglevel warning " << utils::extrack::detail::shell_quote(temporary_path.string());
            ffmpeg_pipe = popen(ffmpeg_cmd.str().c_str(), "w");
            if(!ffmpeg_pipe){
                throw std::runtime_error("Failed to open video encoder for " + video_path.string());
            }
        }
        bool record_video = record_video_scene_set && ffmpeg_pipe != nullptr;

        T cam_aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
        TI frame_step_start = ppo_step_i * STEPS_PER_ENV;
        for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
            TI frame_step_i = frame_step_start + step_i;
            auto observations_privileged = view_range(device_gpu, dataset_gpu.all_observations_privileged, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            auto state_observations = view_range(device_gpu, gpu_all_state_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            l2f_visual_prologue_kernel<<<grid, block, 0, device_gpu.stream>>>(
                tag_device, gpu_envs_arr, gpu_params_arr, gpu_states_arr,
                gpu_truncated_arr, gpu_episode_step_arr, gpu_episode_return_arr,
                gpu_episode_end_reason_arr,
                gpu_render_reset_arr,
                gpu_shutter_fraction_arr,
                observations_privileged, state_observations,
                data(inertial_states_gpu),
                gpu_episode_lengths_log, gpu_episode_returns_log, gpu_episode_end_reasons_log,
                gpu_brightness_scale_arr,
                gpu_target_brightness_scale_arr,
                gpu_target_frame_roll_arr,
                gpu_target_frame_pitch_arr,
                gpu_scene_translation_arr,
                gpu_scene_yaw_arr,
                gpu_scene_yaw_cos_arr,
                gpu_scene_yaw_sin_arr,
                gpu_positions, gpu_num_positions, gpu_env_scene, MAX_INDOOR_POS,
                gpu_episode_start_step,
                rng_gpu, step_i, frame_step_i, rng_offset, mode);
            check_status(device_gpu);
            if constexpr(MIXED_TRAINING){
                geometric_teacher_kernel<<<grid, block, 0, device_gpu.stream>>>(tag_device, gpu_params_arr, gpu_states_arr, data(state_observations), data(teacher_actions_gpu) + step_i * N_ENVIRONMENTS * ACTION_DIM);
            }
            evaluate_values(device_gpu, dataset_gpu, ppo_gpu.critic, critic_states_gae, critic_buffers_gae, rng_gpu, step_i);
            record_episode_start_kernel<<<grid, block, 0, device_gpu.stream>>>(gpu_episode_start_step, gpu_episode_start_step_per_row, step_i);
            check_status(device_gpu);

            l2f_visual_make_cameras_kernel<<<grid, block, 0, device_gpu.stream>>>(
                tag_device, gpu_params_arr, gpu_states_arr,
                gpu_cameras, gpu_cameras_open, gpu_prev_cameras,
                gpu_render_reset_arr,
                gpu_shutter_fraction_arr,
                frame_step_i, cam_aspect,
                gpu_scene_translation_arr, gpu_scene_yaw_cos_arr, gpu_scene_yaw_sin_arr);
            check_status(device_gpu);

            T* obs_ptr = data(dataset_gpu.all_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
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
                rlt::render_launch(device, renderer);
                const float* obs_src_ptr = active_scene_observation_ptrs[active_scene_i];
                if constexpr(BRIGHTNESS_RANDOMIZATION_RANGE > 0){
                    scatter_pixel_to_float_kernel<true><<<pf_grid, pf_block, 0, optix_stream>>>(obs_src_ptr, obs_ptr, gpu_brightness_scale_arr, base_env, n_envs_s, CAM_WIDTH * CAM_HEIGHT, OBSERVATION_DIM);
                } else {
                    scatter_pixel_to_float_kernel<false><<<pf_grid, pf_block, 0, optix_stream>>>(obs_src_ptr, obs_ptr, nullptr, base_env, n_envs_s, CAM_WIDTH * CAM_HEIGHT, OBSERVATION_DIM);
                }
                cudaEventRecord(render_scatter_done_events[active_scene_i], optix_stream);
            }
            for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                cudaStreamWaitEvent(device_gpu.stream, render_scatter_done_events[active_scene_i], 0);
            }
            {
                TI history_slot = frame_step_i % FRAME_STACK_HISTORY_LENGTH;
                T* history_slot_ptr = data(gpu_frame_stack_history) + (TI)(history_slot * N_ENVIRONMENTS) * OBSERVATION_DIM;
                init_level_cameras_kernel<<<(N_ENVIRONMENTS + 255) / 256, 256, 0, device_gpu.stream>>>(tag_device, gpu_params_arr, data(state_observations), data(level_cameras_gpu));
                rectify_images_kernel<<<(N_ENVIRONMENTS * CAM_HEIGHT * CAM_WIDTH + 255) / 256, 256, 0, device_gpu.stream>>>(tag_device, data(level_cameras_gpu), obs_ptr, history_slot_ptr);
            }

            l2f_visual_make_target_cameras_kernel<<<grid, block, 0, device_gpu.stream>>>(
                tag_device, gpu_params_arr, gpu_target_cameras,
                cam_aspect,
                gpu_target_frame_roll_arr,
                gpu_target_frame_pitch_arr,
                gpu_scene_translation_arr, gpu_scene_yaw_cos_arr, gpu_scene_yaw_sin_arr);
            check_status(device_gpu);

            T* target_obs_ptr = data(gpu_all_target_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
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
                rlt::render_launch(device, renderer);
                const float* obs_src_ptr = active_scene_observation_ptrs[active_scene_i];
                if constexpr(BRIGHTNESS_RANDOMIZATION_RANGE > 0 || TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE > 0){
                    scatter_pixel_to_float_kernel<true><<<pf_grid, pf_block, 0, optix_stream>>>(obs_src_ptr, target_obs_ptr, gpu_target_brightness_scale_arr, base_env, n_envs_s, CAM_WIDTH * CAM_HEIGHT, OBSERVATION_DIM);
                } else {
                    scatter_pixel_to_float_kernel<false><<<pf_grid, pf_block, 0, optix_stream>>>(obs_src_ptr, target_obs_ptr, nullptr, base_env, n_envs_s, CAM_WIDTH * CAM_HEIGHT, OBSERVATION_DIM);
                }
                cudaEventRecord(target_render_scatter_done_events[active_scene_i], optix_stream);
            }
            for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                cudaStreamWaitEvent(device_gpu.stream, target_render_scatter_done_events[active_scene_i], 0);
            }

            if(record_video && ffmpeg_pipe && frame_step_i % VIDEO_FRAME_STRIDE == 0){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(cpu_obs_video.data(), obs_ptr, N_ENVIRONMENTS * OBSERVATION_DIM * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(cpu_target_obs_video.data(), target_obs_ptr, N_ENVIRONMENTS * OBSERVATION_DIM * sizeof(float), cudaMemcpyDeviceToHost);
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
                if(std::fwrite(mosaic_frame.data(), 1, mosaic_frame.size(), ffmpeg_pipe) != mosaic_frame.size()){
                    throw std::runtime_error("Video frame write failed");
                }
            }

            {
                int total_elements = N_ENVIRONMENTS * COMBINED_OBS_DIM;
                build_frame_stacked_with_target_from_history_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                    data(gpu_frame_stack_history),
                    target_obs_ptr,
                    gpu_episode_start_step,
                    frame_step_i,
                    data(gpu_rollout_combined),
                    OBSERVATION_DIM, IMG_C, FRAME_STACK_N, FRAME_STACK_STRIDE, COMBINED_IMG_C, COMBINED_OBS_DIM, N_ENVIRONMENTS);
            }
            check_status(device_gpu);

            if constexpr(rlt::mode::is<MODE, hold_evaluation::Blind>){
                set_all(device_gpu, gpu_rollout_combined, 0);
            }
            {
                using ROLLOUT_IMG_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, IMG_H, IMG_W, COMBINED_IMG_C>;
                auto step_combined_reshaped = reshape_row_major(device_gpu, gpu_rollout_combined, ROLLOUT_IMG_SHAPE{});
                auto step_state_obs = view_range(device_gpu, gpu_all_state_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                using ROLLOUT_STATE_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, STATE_OBS_DIM>;
                auto step_state_reshaped = reshape_row_major(device_gpu, step_state_obs, ROLLOUT_STATE_SHAPE{});
                auto inputs = rlt::nn_models::parallel::pack_inputs(step_combined_reshaped, step_state_reshaped);
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
                cudaMemcpy(&reward_log_state, gpu_states_arr, sizeof(typename ENVIRONMENT::State), cudaMemcpyDeviceToHost);
            }
            {
                auto& last_layer_gpu = ppo_gpu.actor.head;
                auto log_std_gpu = matrix_view(device_gpu, last_layer_gpu.log_std.parameters);
                auto actions_mean_view = view(device_gpu, dataset_gpu.actions_mean, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
                auto actions_view = view(device_gpu, dataset_gpu.actions, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
                l2f_visual_epilogue_kernel<<<grid, block, 0, device_gpu.stream>>>(
                    tag_device, gpu_envs_arr, gpu_params_arr, gpu_states_arr,
                    gpu_truncated_arr, gpu_episode_step_arr, gpu_episode_return_arr,
                    gpu_episode_end_reason_arr,
                    actions_mean_view, actions_view, log_std_gpu,
                    dataset_gpu, MIXED_TRAINING ? data(teacher_actions_gpu) : nullptr, optimizer_updates,
                    rng_gpu, step_i, EVALUATION ? 0 : EPISODE_STEP_LIMIT, rng_offset, mode);
                check_status(device_gpu);
            }
            if constexpr(ON_POLICY_RUNNER_SPEC::COLLECT_NEXT_OBSERVATIONS){
                l2f_visual_final_priv_obs_kernel<<<grid, block, 0, device_gpu.stream>>>(tag_device, gpu_envs_arr, gpu_params_arr, gpu_states_arr, next_privileged_observations, rng_gpu, rng_offset);
                check_status(device_gpu);
                evaluate_bootstrap_values(device_gpu, dataset_gpu, next_privileged_observations, ppo_gpu.critic, critic_states_gae, critic_buffers_gae, rng_gpu, step_i);
            }
            if(log_reward_components_this_step && step_i == STEPS_PER_ENV - 1){
                cudaStreamSynchronize(device_gpu.stream);
                cudaMemcpy(&reward_log_next_state, gpu_states_arr, sizeof(typename ENVIRONMENT::State), cudaMemcpyDeviceToHost);
                cudaMemcpy(&reward_log_parameters, gpu_params_arr, sizeof(typename ENVIRONMENT::Parameters), cudaMemcpyDeviceToHost);
            }

            if(save_extrack_step){
                cudaStreamSynchronize(device_gpu.stream);
                std::vector<typename ENVIRONMENT::State> tmp_states(TRAJECTORY_NUM_ENVS);
                cudaMemcpy(tmp_states.data(), gpu_states_arr, TRAJECTORY_NUM_ENVS * sizeof(typename ENVIRONMENT::State), cudaMemcpyDeviceToHost);
                for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                    trajectory_states[step_i * TRAJECTORY_NUM_ENVS + env_i] = tmp_states[env_i];
                }
            }
        }

        if(!EVALUATION && ffmpeg_pipe && rollout_in_scene_set + 1 == ROLLOUTS_PER_SCENE_SET){
            close_video_pipe();
        }

        {
            auto final_obs_priv = view_range(device_gpu, dataset_gpu.all_observations_privileged, STEPS_PER_ENV * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            l2f_visual_final_priv_obs_kernel<<<grid, block, 0, device_gpu.stream>>>(
                tag_device, gpu_envs_arr, gpu_params_arr, gpu_states_arr, final_obs_priv, rng_gpu, rng_offset);
            check_status(device_gpu);
        }
        environment_step += N_ENVIRONMENTS * STEPS_PER_ENV;
        set_step(device, device.logger, environment_step);

        evaluate_rollout_values(device_gpu, dataset_gpu, ppo_gpu.critic, critic_buffers_gae, rng_gpu, PPO_SPEC::COLLECTION_MODE{});

        cudaDeviceSynchronize();
        cudaMemcpy(group_episode_steps[accumulation_i].data(), gpu_episode_step_arr, N_ENVIRONMENTS * sizeof(TI), cudaMemcpyDeviceToHost);
        cudaMemcpy(group_episode_returns[accumulation_i].data(), gpu_episode_return_arr, N_ENVIRONMENTS * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(group_episode_end_reasons[accumulation_i].data(), gpu_episode_end_reason_arr, N_ENVIRONMENTS * sizeof(T), cudaMemcpyDeviceToHost);
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
        if constexpr(EVALUATION){
            auto observations = matrix_view(device, dataset.all_observations_privileged);
            for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
                const TI time_step = ppo_step_i * STEPS_PER_ENV + step_i;
                if(time_step >= evaluation_steps){
                    break;
                }
                for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                    const TI row_i = step_i * N_ENVIRONMENTS + env_i;
                    if(first_failure_step[env_i] <= evaluation_steps){
                        continue;
                    }
                    if(time_step % 10 == 0){
                        evaluation_trace << static_cast<double>(time_step) / SIMULATION_FREQUENCY << ',' << env_i;
                        for(TI column = 0; column < 18; column++){
                            evaluation_trace << ',' << get(observations, row_i, column);
                        }
                        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                            evaluation_trace << ',' << get(dataset.actions, row_i, action_i);
                        }
                        evaluation_trace << ',' << get(dataset.terminated, row_i, 0) << '\n';
                    }
                    bool finite = true;
                    for(TI column = 0; column < 18; column++){
                        const T value = get(observations, row_i, column);
                        std::uint32_t bits;
                        static_assert(sizeof(value) == sizeof(bits));
                        std::memcpy(&bits, &value, sizeof(bits));
                        finite = finite && (bits & 0x7f800000u) != 0x7f800000u;
                    }
                    if(!finite || get(dataset.terminated, row_i, 0)){
                        first_failure_step[env_i] = time_step + 1;
                        continue;
                    }
                    if(time_step >= 5 * SIMULATION_FREQUENCY){
                        double position_squared = 0;
                        double speed_squared = 0;
                        for(TI axis = 0; axis < 3; axis++){
                            const double position = get(observations, row_i, axis);
                            const double velocity = get(observations, row_i, 12 + axis);
                            position_squared += position * position;
                            speed_squared += velocity * velocity;
                            position_sum[env_i][axis] += position;
                            if(time_step < 10 * SIMULATION_FREQUENCY){
                                initial_position_sum[env_i][axis] += position;
                            }
                            if(time_step + 5 * SIMULATION_FREQUENCY >= evaluation_steps){
                                final_position_sum[env_i][axis] += position;
                            }
                        }
                        initial_window_samples[env_i] += time_step < 10 * SIMULATION_FREQUENCY;
                        final_window_samples[env_i] += time_step + 5 * SIMULATION_FREQUENCY >= evaluation_steps;
                        position_squared_sum[env_i] += position_squared;
                        speed_squared_sum[env_i] += speed_squared;
                        position_max[env_i] = std::max(position_max[env_i], std::sqrt(position_squared));
                        evaluation_samples[env_i]++;
                    }
                }
            }
            TI survivors = 0;
            double total_position_squared = 0, speed_sum = 0;
            TI samples = 0;
            for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                survivors += first_failure_step[env_i] > evaluation_steps;
                total_position_squared += position_squared_sum[env_i];
                speed_sum += speed_squared_sum[env_i];
                samples += evaluation_samples[env_i];
            }
            const TI elapsed_steps = std::min((ppo_step_i + 1) * STEPS_PER_ENV, evaluation_steps);
            std::cout << "Evaluation " << static_cast<double>(elapsed_steps) / SIMULATION_FREQUENCY
                      << "s: survivors " << survivors << "/" << N_ENVIRONMENTS
                      << " position RMS " << (samples ? std::to_string(std::sqrt(total_position_squared / samples)) : "n/a")
                      << " m speed RMS " << (samples ? std::to_string(std::sqrt(speed_sum / samples)) : "n/a") << " m/s" << std::endl;
            if(elapsed_steps == evaluation_steps || survivors == 0){
                std::ofstream report(extrack_paths.seed / "evaluation.json");
                report << "{\"checkpoint\":" << nlohmann::json(argv[3])
                       << ",\"seed\":" << seed << ",\"requested_seconds\":" << static_cast<double>(evaluation_steps) / SIMULATION_FREQUENCY
                       << ",\"elapsed_seconds\":" << static_cast<double>(elapsed_steps) / SIMULATION_FREQUENCY
                       << ",\"blind\":" << (rlt::mode::is<MODE, hold_evaluation::Blind> ? "true" : "false")
                       << ",\"recovery\":" << (rlt::mode::is<MODE, hold_evaluation::Recovery> ? "true" : "false")
                       << ",\"survivors\":" << survivors << ",\"environments\":[";
                for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                    if(env_i > 0){
                        report << ',';
                    }
                    const TI samples_i = evaluation_samples[env_i];
                    double mean_squared = 0, drift_squared = 0;
                    std::array<double, 3> mean_position{}, position_drift{};
                    for(TI axis = 0; axis < 3; axis++){
                        if(samples_i > 0){
                            const double mean = position_sum[env_i][axis] / samples_i;
                            mean_squared += mean * mean;
                            mean_position[axis] = mean;
                        }
                        if(initial_window_samples[env_i] > 0 && final_window_samples[env_i] > 0){
                            const double drift = final_position_sum[env_i][axis] / final_window_samples[env_i] - initial_position_sum[env_i][axis] / initial_window_samples[env_i];
                            drift_squared += drift * drift;
                            position_drift[axis] = drift;
                        }
                    }
                    report << "{\"scene\":" << nlohmann::json(scene_paths[active_scene_indices[env_i / N_ENVIRONMENTS_PER_SCENE]])
                           << ",\"first_failure_seconds\":";
                    if(first_failure_step[env_i] > evaluation_steps){
                        report << "null";
                    }
                    else{
                        report << static_cast<double>(first_failure_step[env_i]) / SIMULATION_FREQUENCY;
                    }
                    report << ",\"samples_after_settling\":" << samples_i
                           << ",\"position_mean_m\":" << (samples_i ? nlohmann::json(mean_position) : nlohmann::json(nullptr))
                           << ",\"position_drift_xyz_m\":" << (initial_window_samples[env_i] && final_window_samples[env_i] ? nlohmann::json(position_drift) : nlohmann::json(nullptr))
                           << ",\"position_rms_m\":" << (samples_i ? std::to_string(std::sqrt(position_squared_sum[env_i] / samples_i)) : "null")
                           << ",\"position_max_m\":" << (samples_i ? std::to_string(position_max[env_i]) : "null")
                           << ",\"speed_rms_m_s\":" << (samples_i ? std::to_string(std::sqrt(speed_squared_sum[env_i] / samples_i)) : "null")
                           << ",\"position_bias_m\":" << (samples_i ? std::to_string(std::sqrt(mean_squared)) : "null")
                           << ",\"position_variation_rms_m\":" << (samples_i ? std::to_string(std::sqrt(std::max(0.0, position_squared_sum[env_i] / samples_i - mean_squared))) : "null")
                           << ",\"position_drift_m\":" << (initial_window_samples[env_i] && final_window_samples[env_i] ? std::to_string(std::sqrt(drift_squared)) : "null") << '}';
                }
                report << "]}\n";
                metra::log("l2f_visual_hold/survival_fraction", static_cast<double>(survivors) / N_ENVIRONMENTS);
                if(samples > 0){
                    metra::log("l2f_visual_hold/position_rms_m", std::sqrt(total_position_squared / samples));
                    metra::log("l2f_visual_hold/speed_rms_m_s", std::sqrt(speed_sum / samples));
                }
                break;
            }
            continue;
        }
        if(log_reward_components_this_step){
            static constexpr TI REWARD_LOG_POS = (STEPS_PER_ENV - 1) * N_ENVIRONMENTS;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                set(reward_log_action, 0, action_i, get(dataset.actions, REWARD_LOG_POS, action_i));
            }
            log_reward(device, envs[0].dynamics, reward_log_parameters.dynamics, reward_log_state, reward_log_action, reward_log_next_state, reward_log_rng);
        }

        {
            copy(device_gpu, device, gpu_episode_lengths_log, cpu_episode_lengths_log);
            copy(device_gpu, device, gpu_episode_returns_log, cpu_episode_returns_log);
            copy(device_gpu, device, gpu_episode_end_reasons_log, cpu_episode_end_reasons_log);
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
                T ep_len = get(cpu_episode_lengths_log, pos, 0);
                if(ep_len >= (T)0){
                    const TI subset = is_imitation_sample(device, MIXED_SPEC{}, pos) ? 0 : 1;
                    subset_length_sum[subset] += ep_len;
                    subset_episode_count[subset]++;
                    add_scalar(device, device.logger, "episode/length", ep_len, 100);
                    T ep_return = get(cpu_episode_returns_log, pos, 0);
                    add_scalar(device, device.logger, "episode/return", ep_return, 100);
                    length_sum += ep_len;
                    length_sq_sum += ep_len * ep_len;
                    return_sum += ep_return;
                    return_sq_sum += ep_return * ep_return;
                    TI ep_reason = static_cast<TI>(get(cpu_episode_end_reasons_log, pos, 0) + (T)0.5);
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
                }
            }
            rollout_reward_mean = reward_sum / static_cast<T>(STEPS_TOTAL);
            rollout_reward_std = rlt::math::sqrt(device.math, rlt::math::max(device.math, (T)0, reward_sq_sum / static_cast<T>(STEPS_TOTAL) - rollout_reward_mean * rollout_reward_mean));
            rollout_terminated_share = rollout_done_count > 0 ? static_cast<T>(rollout_terminated_count) / static_cast<T>(rollout_done_count) : (T)0;
            if constexpr(MIXED_TRAINING){
                for(TI subset = 0; subset < 2; subset++){
                    if(subset_episode_count[subset] > 0){
                        const T mean = subset_length_sum[subset] / subset_episode_count[subset];
                        const char* key = subset == 0 ? "mixed/il_episode_length" : "mixed/rl_episode_length";
                        add_scalar(device, device.logger, key, mean);
                        std::cout << "  " << key << ": " << mean << std::endl;
                    }
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
                auto& parameters_ref = env_parameters[0];
                std::string trajectories_json = trajectory_episodes_to_json(device, envs[0], parameters_ref, completed_episodes, simulation_dt);
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
        if constexpr(MIXED_TRAINING){
            copy(device_gpu, device, teacher_actions_gpu, teacher_actions);

        }

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

                {
                    int total_elements = BATCH_SIZE * COMBINED_OBS_DIM;
                    build_frame_stacked_with_target_from_dataset_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                        data(gpu_frame_stack_history),
                        data(gpu_all_target_observations),
                        gpu_episode_start_step_per_row,
                        data(gpu_combined_batch),
                        OBSERVATION_DIM, IMG_C, FRAME_STACK_N, FRAME_STACK_STRIDE, COMBINED_IMG_C, COMBINED_OBS_DIM,
                        frame_step_start, (int)batch_offset, (int)BATCH_SIZE, (int)N_ENVIRONMENTS);
                    check_status(device_gpu);
                }

                auto batch_state_obs = view_range(device_gpu, gpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                using TRAIN_IMG_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
                using TRAIN_STATE_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>;
                auto batch_combined_reshaped = reshape_row_major(device_gpu, gpu_combined_batch, TRAIN_IMG_SHAPE{});
                auto batch_state_reshaped = reshape_row_major(device_gpu, batch_state_obs, TRAIN_STATE_SHAPE{});
                auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(batch_combined_reshaped, batch_state_reshaped);

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
                if constexpr(MIXED_TRAINING){
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
                            metra::log(std::string("l2f_visual_training_mixed/") + key, static_cast<double>(value));
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
                auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(batch_combined_reshaped, batch_state_reshaped);

                    auto& trainable_image = get<0>(ppo_gpu.actor.pipelines);
                    auto& trainable_buffer = get<0>(actor_buffers.sub_buffers);
                    backward_trainable_tail<5>(device_gpu, trainable_image, gpu_d_action_reshaped, trainable_buffer);

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
                        T critic_loss;
                        if constexpr(MIXED_TRAINING){
                            critic_loss = 0;
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
                        }
                        else{
                            critic_loss = rlt::nn::loss_functions::mse::evaluate(device, cpu_critic_output, batch_target_values);
                            rlt::nn::loss_functions::mse::gradient(device, cpu_critic_output, batch_target_values, cpu_d_critic, (T)0.5 * GRADIENT_SCALE);
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
        if(update_ready){
            rlt::utils::assert_exit(device, accumulated_samples == UPDATE_SAMPLES, "Incorrect accumulated update size");
            {
                auto& image_branch = get<0>(ppo_gpu.actor.pipelines);
                zero_gradient(device_gpu, rlt::nn_models::sequential::layer<0>(image_branch));
                zero_gradient(device_gpu, rlt::nn_models::sequential::layer<1>(image_branch));
                zero_gradient(device_gpu, rlt::nn_models::sequential::layer<2>(image_branch));
                zero_gradient(device_gpu, rlt::nn_models::sequential::layer<3>(image_branch));
                zero_gradient(device_gpu, rlt::nn_models::sequential::layer<4>(image_branch));
                set_all(device_gpu, rlt::nn_models::sequential::layer<7>(image_branch).biases.gradient, 0);
                zero_gradient(device_gpu, rlt::nn_models::sequential::layer<9>(image_branch));
                zero_gradient(device_gpu, get<1>(ppo_gpu.actor.pipelines));
                zero_gradient(device_gpu, ppo_gpu.actor.head.input_layer);
                zero_gradient(device_gpu, ppo_gpu.actor.head.output_layer);
            }
            step(device_gpu, actor_optimizer_gpu, ppo_gpu.actor);
            step(device_gpu, critic_optimizer_gpu, ppo_gpu.critic);

            cudaDeviceSynchronize();
            optimizer_updates++;
        }

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
        if constexpr(MIXED_TRAINING){
            std::cout << "  mixed samples: " << imitation_update_samples << " IL + " << ppo_update_samples << " RL; imitation MSE: " << std::setprecision(6) << imitation_mse_sum / N_BATCHES << std::endl;
        }

        add_scalar(device, device.logger, "ppo/step", ppo_step_i);
        add_scalar(device, device.logger, "training/optimizer_updates", optimizer_updates);
        add_scalar(device, device.logger, "training/accumulated_samples", accumulated_samples);
        add_scalar(device, device.logger, "training/update_samples", UPDATE_SAMPLES);
        add_scalar(device, device.logger, "training/sample_uses", N_EPOCHS);
        const std::string metra_prefix = MIXED_TRAINING ? "l2f_visual_training_mixed/" : "l2f_visual_training_on_policy/";
        if(!MIXED_TRAINING || optimizer_updates % 20 == 0){
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

        const bool training_complete = ppo_step_i + 1 == run_steps || (TRAINING_TIME_LIMIT_SECONDS > 0 && training_elapsed.count() >= TRAINING_TIME_LIMIT_SECONDS);
        save_extrack_step = save_extrack_step || training_complete;
        if(save_extrack_step){
            {
                auto step_folder = get_step_folder(device, extrack_config, extrack_paths, environment_step);
                auto latest_folder = get_latest_folder(device, extrack_paths);
                std::filesystem::create_directories(step_folder);

                CHECKPOINT_ACTOR_TYPE eval_actor;
                malloc(device, eval_actor);
                copy(device_gpu, device, ppo_gpu.actor, eval_actor);

                char fov_buf[32];
                std::snprintf(fov_buf, sizeof(fov_buf), "%.6g", (double)envs[0].parameters.fov);
                std::string state_obs_string = rlt::string(device, envs[0].dynamics, ACTOR_STATE_OBS{});
                std::string image_obs_string = std::string("CameraRGBStackedWithTarget(") + fov_buf + ", "
                    + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ", "
                    + std::to_string(FRAME_STACK_STRIDE) + ", " + std::to_string(FRAME_STACK_N) + ")";
                std::string obs_string = image_obs_string + ", " + state_obs_string;
                std::string rendering_string = std::string("{\"anti_aliasing\": ") + (RENDER_ANTI_ALIASING_ACTIVE ? "true" : "false")
                    + ", \"anti_aliasing_grid_size\": " + std::to_string(RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : (TI)1)
                    + ", \"evaluation_default_anti_aliasing_grid_size\": " + std::to_string(RENDER_ENABLE_ANTI_ALIASING ? RENDER_ANTI_ALIASING_GRID_SIZE_EVALUATION : (TI)1)
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
                malloc(device, example_input_0_image);
                malloc(device, example_input_1_state);
                malloc(device, example_output);

                static constexpr TI EXAMPLE_ROW_OFFSET = STEPS_TOTAL - N_EXAMPLES;
                {
                    int total_elements = N_EXAMPLES * COMBINED_OBS_DIM;
                    build_frame_stacked_with_target_from_dataset_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                        data(gpu_frame_stack_history),
                        data(gpu_all_target_observations),
                        gpu_episode_start_step_per_row,
                        data(gpu_combined_batch),
                        OBSERVATION_DIM, IMG_C, FRAME_STACK_N, FRAME_STACK_STRIDE, COMBINED_IMG_C, COMBINED_OBS_DIM,
                        frame_step_start, (int)EXAMPLE_ROW_OFFSET, (int)N_EXAMPLES, (int)N_ENVIRONMENTS);
                    check_status(device_gpu);
                    cudaDeviceSynchronize();
                    auto src_combined = view_range(device_gpu, gpu_combined_batch, (TI)0, rlt::tensor::ViewSpec<0, N_EXAMPLES>{});
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
                    copy(device_gpu, device, ppo_gpu.actor, sized_eval_actor);
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

    close_video_pipe();

    std::cout << "Training finished at env step " << environment_step << std::endl;

    free(device, ppo);
    free(device, ppo_buffers);
    free(device, dataset);
    free(device, actor_optimizer);
    free(device, critic_optimizer);

    free(device_gpu, teacher_actions_gpu);
    free(device, teacher_actions);
    free(device, cpu_episode_lengths_log);
    free(device, cpu_episode_returns_log);
    free(device, cpu_episode_end_reasons_log);
    free(device, reward_log_action);
    free(device, reward_log_rng);

    free(device_gpu, ppo_gpu);
    free(device_gpu, actor_buffers);
    free(device_gpu, critic_buffers);
    free(device_gpu, critic_states_gae);
    free(device_gpu, critic_buffers_gae);
    if constexpr(ON_POLICY_RUNNER_SPEC::COLLECT_NEXT_OBSERVATIONS){
        free(device_gpu, next_privileged_observations);
    }
    free(device_gpu, dataset_gpu);
    free(device_gpu, rollout_actor_gpu);
    free(device_gpu, rollout_actor_buffers);
    free(device_gpu, actor_optimizer_gpu);
    free(device_gpu, critic_optimizer_gpu);
    free(device_gpu, gpu_all_target_observations);
    free(device_gpu, gpu_frame_stack_history);
    free(device_gpu, level_cameras_gpu);
    free(device_gpu, gpu_rollout_combined);
    free(device_gpu, gpu_combined_batch);

    free(device_gpu, gpu_all_state_observations);
    free(device_gpu, inertial_states_gpu);
    free(device_gpu, gpu_actions_eval);
    free(device_gpu, gpu_actions_train);
    free(device_gpu, gpu_d_action_train);
    free(device_gpu, gpu_critic_obs);
    free(device_gpu, gpu_d_critic_output);
    free(device_gpu, gpu_episode_lengths_log);
    free(device_gpu, gpu_episode_returns_log);
    free(device_gpu, gpu_episode_end_reasons_log);
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

    cudaFree(gpu_positions);
    cudaFree(gpu_num_positions);
    cudaFree(gpu_env_scene);
    cudaFree(gpu_envs_arr);
    cudaFree(gpu_params_arr);
    cudaFree(gpu_states_arr);
    cudaFree(gpu_truncated_arr);
    cudaFree(gpu_episode_step_arr);
    cudaFree(gpu_episode_return_arr);
    cudaFree(gpu_episode_end_reason_arr);
    cudaFree(gpu_brightness_scale_arr);
    cudaFree(gpu_target_brightness_scale_arr);
    cudaFree(gpu_target_frame_roll_arr);
    cudaFree(gpu_target_frame_pitch_arr);
    cudaFree(gpu_scene_translation_arr);
    cudaFree(gpu_scene_yaw_arr);
    cudaFree(gpu_scene_yaw_cos_arr);
    cudaFree(gpu_scene_yaw_sin_arr);
    cudaFree(gpu_episode_start_step);
    cudaFree(gpu_episode_start_step_per_row);
    cudaEventDestroy(cameras_ready_event);
    cudaEventDestroy(target_cameras_ready_event);
    for(TI scene_i = 0; scene_i < N_ACTIVE_SCENES; scene_i++){
        cudaEventDestroy(render_scatter_done_events[scene_i]);
        cudaEventDestroy(target_render_scatter_done_events[scene_i]);
    }
    free(device_gpu, rng_gpu);
    free(device, rng);
    for(auto& env: envs){
        free(device, env.dynamics);
    }
    for(auto* renderer: renderers){
        free(device, *renderer);
    }
    delete renderer_storage;
    free(device, *library);
    delete library;

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
    free(device, device.logger);
#endif

    return 0;
}

int main(int argc, char** argv){
    if constexpr(EVALUATION_BUILD){
        if(argc < 4){
            std::cerr << "Usage: " << argv[0] << " <scene_directory> <seed> <checkpoint_2examples.h5> [seconds=600] [blind|recovery|blind-recovery]" << std::endl;
            return 1;
        }
        if(std::string(argv[3]) == "teacher"){
            return run(argc, argv, rlt::Mode<hold_evaluation::Teacher<>>{});
        }
        if(argc > 5){
            if(std::string(argv[5]) == "blind-recovery"){
                return run(argc, argv, rlt::Mode<hold_evaluation::Blind<hold_evaluation::Recovery<>>>{});
            }
            if(std::string(argv[5]) == "blind"){
                return run(argc, argv, rlt::Mode<hold_evaluation::Blind<>>{});
            }
            if(std::string(argv[5]) == "recovery"){
                return run(argc, argv, rlt::Mode<hold_evaluation::Recovery<>>{});
            }
            std::cerr << "Unknown evaluation scenario: " << argv[5] << std::endl;
            return 1;
        }
        return run(argc, argv, rlt::Mode<rlt::mode::Evaluation<>>{});
    }
    else{
        if(argc > 3){
            std::cerr << "Use rl_environments_l2f_visual_evaluate_mixed_cuda for checkpoint evaluation" << std::endl;
            return 1;
        }
        return run(argc, argv, rlt::Mode<rlt::mode::l2f_visual::IndependentPose<>>{});
    }
}
