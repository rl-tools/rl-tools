#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include "on_policy_runner.h"
#include "../../../random/operations_generic_array.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename DATASET, typename RUNNER>
    RL_TOOLS_FUNCTION_PLACEMENT void record_episode_start(DEVICE& device, DATASET& dataset, RUNNER& runner, typename RUNNER::TI env_i){
        using T = typename DATASET::T;
        const rl::components::on_policy_runner::EpisodeEndReason reason = get(device, runner.completed_episode_reason, env_i);
        set(dataset.reset, env_i, 0, get(device, runner.reset, env_i) ? (T)1 : (T)0);
        set(device, dataset.episode_end_reason, reason == rl::components::on_policy_runner::EpisodeEndReason::FORCED ? reason : rl::components::on_policy_runner::EpisodeEndReason::NONE, 0, env_i);
        set(device, dataset.episode_length, reason == rl::components::on_policy_runner::EpisodeEndReason::FORCED ? get(device, runner.completed_episode_length, env_i) : 0, 0, env_i);
        set(device, dataset.episode_return, reason == rl::components::on_policy_runner::EpisodeEndReason::FORCED ? get(device, runner.completed_episode_return, env_i) : 0, 0, env_i);
        set(device, runner.completed_episode_reason, rl::components::on_policy_runner::EpisodeEndReason::NONE, env_i);
        set(device, runner.completed_episode_length, 0, env_i);
        set(device, runner.completed_episode_return, 0, env_i);
    }
    template <typename DEVICE, typename DATASET, typename RUNNER>
    RL_TOOLS_FUNCTION_PLACEMENT void record_transition(DEVICE& device, DATASET& dataset, RUNNER& runner, typename DATASET::T transition_reward, bool terminated, typename RUNNER::TI step_i, typename RUNNER::TI env_i){
        using T = typename DATASET::T;
        using TI = typename RUNNER::TI;
        const TI pos = step_i * RUNNER::SPEC::N_ENVIRONMENTS + env_i;
        const TI episode_step = get(device, runner.episode_step, env_i) + 1;
        const typename RUNNER::EPISODE_T reward = transition_reward;
        const typename RUNNER::EPISODE_T episode_return = get(device, runner.episode_return, env_i) + reward;
        const bool time_limit = runner.episode_step_limit > 0 && episode_step >= runner.episode_step_limit;
        const bool reset = terminated || time_limit;
        const rl::components::on_policy_runner::EpisodeEndReason reason = terminated ? rl::components::on_policy_runner::EpisodeEndReason::TERMINATED : time_limit ? rl::components::on_policy_runner::EpisodeEndReason::TIME_LIMIT : rl::components::on_policy_runner::EpisodeEndReason::NONE;
        set(dataset.rewards, pos, 0, (T)reward);
        set(dataset.terminated, pos, 0, terminated ? (T)1 : (T)0);
        set(dataset.truncated, pos, 0, reset ? (T)1 : (T)0);
        set(dataset.all_reset, pos + RUNNER::SPEC::N_ENVIRONMENTS, 0, reset ? (T)1 : (T)0);
        set(device, dataset.episode_end_reason, reason, step_i + 1, env_i);
        set(device, dataset.episode_length, reset ? episode_step : (TI)0, step_i + 1, env_i);
        set(device, dataset.episode_return, reset ? episode_return : (typename RUNNER::EPISODE_T)0, step_i + 1, env_i);
        set(device, runner.reset, reset, env_i);
        set(device, runner.completed_episode_reason, reason, env_i);
        set(device, runner.completed_episode_length, reset ? episode_step : (TI)0, env_i);
        set(device, runner.completed_episode_return, reset ? episode_return : (typename RUNNER::EPISODE_T)0, env_i);
        set(device, runner.episode_step, reset ? (TI)0 : episode_step, env_i);
        set(device, runner.episode_return, reset ? (typename RUNNER::EPISODE_T)0 : episode_return, env_i);
    }
    template <typename DEVICE, typename RUNNER>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_episode(DEVICE& device, RUNNER& runner, typename RUNNER::TI env_i){
        using TI = typename RUNNER::TI;
        using T = typename RUNNER::EPISODE_T;
        const TI episode_step = get(device, runner.episode_step, env_i);
        set(device, runner.reset, true, env_i);
        set(device, runner.completed_episode_reason, episode_step > 0 ? rl::components::on_policy_runner::EpisodeEndReason::FORCED : rl::components::on_policy_runner::EpisodeEndReason::NONE, env_i);
        set(device, runner.completed_episode_length, episode_step, env_i);
        set(device, runner.completed_episode_return, get(device, runner.episode_return, env_i), env_i);
        set(device, runner.episode_step, (TI)0, env_i);
        set(device, runner.episode_return, (T)0, env_i);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        malloc(device, dataset.all_observations);
        malloc(device, dataset.all_observations_privileged);
        malloc(device, dataset.episode_end_reason);
        malloc(device, dataset.episode_length);
        malloc(device, dataset.episode_return);
        malloc(device, dataset.scalar_data);
        using SCALAR_DATA_SPEC = typename decltype(dataset.scalar_data)::SPEC;
        using TI = typename SPEC::SPEC::TI;
        TI pos = 0;
        dataset.actions_mean     = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.actions_mean    )::ROWS, decltype(dataset.actions_mean    )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.actions_mean    )::COLS;
        dataset.actions          = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.actions         )::ROWS, decltype(dataset.actions         )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.actions         )::COLS;
        dataset.action_log_probs = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.action_log_probs)::ROWS, decltype(dataset.action_log_probs)::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.action_log_probs)::COLS;
        dataset.rewards          = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.rewards         )::ROWS, decltype(dataset.rewards         )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.rewards         )::COLS;
        dataset.terminated       = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.terminated      )::ROWS, decltype(dataset.terminated      )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.terminated      )::COLS;
        dataset.truncated        = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.truncated       )::ROWS, decltype(dataset.truncated       )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.truncated       )::COLS;
        dataset.all_reset        = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.all_reset       )::ROWS, decltype(dataset.all_reset       )::COLS>(device, dataset.scalar_data, 0, pos);
        dataset.reset            = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.reset           )::ROWS, decltype(dataset.reset           )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.reset           )::COLS;
        dataset.all_values       = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.all_values      )::ROWS, decltype(dataset.all_values      )::COLS>(device, dataset.scalar_data, 0, pos);
        dataset.values           = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.values          )::ROWS, decltype(dataset.values          )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.values          )::COLS;
        dataset.advantages       = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.advantages      )::ROWS, decltype(dataset.advantages      )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.advantages      )::COLS;
        dataset.target_values    = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.target_values   )::ROWS, decltype(dataset.target_values   )::COLS>(device, dataset.scalar_data, 0, pos);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        free(device, dataset.all_observations);
        free(device, dataset.all_observations_privileged);
        free(device, dataset.episode_end_reason);
        free(device, dataset.episode_length);
        free(device, dataset.episode_return);
        free(device, dataset.scalar_data);
        dataset.actions_mean               ._data = nullptr;
        dataset.actions                    ._data = nullptr;
        dataset.action_log_probs           ._data = nullptr;
        dataset.rewards                    ._data = nullptr;
        dataset.terminated                 ._data = nullptr;
        dataset.truncated                  ._data = nullptr;
        dataset.all_reset                  ._data = nullptr;
        dataset.reset                      ._data = nullptr;
        dataset.all_values                 ._data = nullptr;
        dataset.values                     ._data = nullptr;
        dataset.advantages                 ._data = nullptr;
        dataset.target_values              ._data = nullptr;
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        malloc(device, runner.policy_state);
        malloc(device, runner.env_parameters);
        malloc(device, runner.states);
        malloc(device, runner.episode_step);
        malloc(device, runner.reset);
        malloc(device, runner.episode_return);
        malloc(device, runner.completed_episode_length);
        malloc(device, runner.completed_episode_return);
        malloc(device, runner.completed_episode_reason);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        free(device, runner.policy_state);
        free(device, runner.env_parameters);
        free(device, runner.states);
        free(device, runner.episode_step);
        free(device, runner.reset);
        free(device, runner.episode_return);
        free(device, runner.completed_episode_length);
        free(device, runner.completed_episode_return);
        free(device, runner.completed_episode_reason);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        using T = typename rl::components::OnPolicyRunner<SPEC>::EPISODE_T;
        using TI = typename SPEC::TI;
        using END_REASON = rl::components::on_policy_runner::EpisodeEndReason;
        runner.step = 0;
        set_all(device, runner.episode_step, (TI)0);
        set_all(device, runner.reset, true);
        set_all(device, runner.episode_return, (T)0);
        set_all(device, runner.completed_episode_length, (TI)0);
        set_all(device, runner.completed_episode_return, (T)0);
        set_all(device, runner.completed_episode_reason, END_REASON::NONE);
        runner.episode_step_limit = SPEC::STEP_LIMIT;
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = false;
#endif
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer){
        malloc(device, buffer.next_states);
        malloc(device, buffer.actions);
        malloc(device, buffer.rewards);
        malloc(device, buffer.terminated);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer){
        free(device, buffer.next_states);
        free(device, buffer.actions);
        free(device, buffer.rewards);
        free(device, buffer.terminated);
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::SPEC::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC_1>& d1, rl::components::on_policy_runner::Dataset<SPEC_2>& d2){
        using T = typename SPEC_1::SPEC::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, d1.all_observations, d2.all_observations);
        acc += abs_diff(device, d1.all_observations_privileged, d2.all_observations_privileged);
        acc += abs_diff(device, d1.episode_length, d2.episode_length);
        acc += abs_diff(device, d1.episode_return, d2.episode_return);
        for(typename SPEC_1::TI step_i = 0; step_i <= SPEC_1::STEPS_PER_ENV; step_i++){
            for(typename SPEC_1::TI env_i = 0; env_i < SPEC_1::SPEC::N_ENVIRONMENTS; env_i++){
                acc += math::abs(device.math, (T)get(device, d1.episode_end_reason, step_i, env_i) - (T)get(device, d2.episode_end_reason, step_i, env_i));
            }
        }
        acc += abs_diff(device, d1.actions_mean, d2.actions_mean);
        acc += abs_diff(device, d1.actions, d2.actions);
        acc += abs_diff(device, d1.action_log_probs, d2.action_log_probs);
        acc += abs_diff(device, d1.rewards, d2.rewards);
        acc += abs_diff(device, d1.terminated, d2.terminated);
        acc += abs_diff(device, d1.truncated, d2.truncated);
        acc += abs_diff(device, d1.all_reset, d2.all_reset);
        acc += abs_diff(device, d1.all_values, d2.all_values);
        acc += abs_diff(device, d1.advantages, d2.advantages);
        acc += abs_diff(device, d1.target_values, d2.target_values);
        return acc;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, rl::components::OnPolicyRunner<SPEC_1>& r1, rl::components::OnPolicyRunner<SPEC_2>& r2){
        using T = typename SPEC_1::TYPE_POLICY::DEFAULT;
        using TI = typename DEVICE::index_t;
        T acc = 0;
        acc += math::abs(device.math, (T)r1.step - (T)r2.step);
        acc += abs_diff(device, r1.policy_state, r2.policy_state);
        for(TI env_i = 0; env_i < SPEC_1::N_ENVIRONMENTS; env_i++){
            TI episode_step_r1 = get(device, r1.episode_step, env_i);
            TI episode_step_r2 = get(device, r2.episode_step, env_i);
            acc += math::abs(device.math, (T)episode_step_r1 - (T)episode_step_r2);
            TI completed_length_r1 = get(device, r1.completed_episode_length, env_i);
            TI completed_length_r2 = get(device, r2.completed_episode_length, env_i);
            acc += math::abs(device.math, (T)completed_length_r1 - (T)completed_length_r2);
            acc += math::abs(device.math, (T)get(device, r1.completed_episode_reason, env_i) - (T)get(device, r2.completed_episode_reason, env_i));
        }
        acc += abs_diff(device, r1.episode_return, r2.episode_return);
        acc += abs_diff(device, r1.reset, r2.reset);
        acc += abs_diff(device, r1.completed_episode_return, r2.completed_episode_return);
        acc += math::abs(device.math, (T)r1.episode_step_limit - (T)r2.episode_step_limit);
        for(TI env_i = 0; env_i < SPEC_1::N_ENVIRONMENTS; env_i++){
            acc += abs_diff(device, get_ref(device, r1.states, env_i), get_ref(device, r2.states, env_i));
            acc += abs_diff(device, get_ref(device, r1.env_parameters, env_i), get_ref(device, r2.env_parameters, env_i));
        }
        return acc;
    }

    template <auto INSTANCES, typename RNG, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT RNG& instance_rng(RNG& rng, TI){
        return rng;
    }
    template <auto INSTANCES, typename RNG_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto& instance_rng(devices::generic::random::ArrayENGINE<RNG_SPEC>& rng, typename RNG_SPEC::TI instance_i){
        static_assert(RNG_SPEC::NUM_RNGS >= INSTANCES, "the runner needs one RNG state per environment instance");
        return get(rng.states, 0, instance_i);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename rl_tools::utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void record_episode_start(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner){
        using TI = typename SPEC::TI;
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            record_episode_start(device, dataset, runner, env_i);
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename rl_tools::utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void record_transition(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, const rl::components::on_policy_runner::Buffer<SPEC>& buffer, typename SPEC::TI step_i){
        using TI = typename SPEC::TI;
        rl_tools::utils::assert_exit(device, step_i < DATASET_SPEC::STEPS_PER_ENV, "on_policy_runner::epilogue: step index outside the dataset");
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            record_transition(device, dataset, runner, get(device, buffer.rewards, env_i), get(device, buffer.terminated, env_i), step_i, env_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename rl_tools::utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void reset_episode(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        using TI = typename SPEC::TI;
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            reset_episode(device, runner, env_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename MASK_SPEC, typename rl_tools::utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void reset_episode(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, const Tensor<MASK_SPEC>& mask){
        using TI = typename SPEC::TI;
        static_assert(get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::N_ENVIRONMENTS);
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            if(get(device, mask, env_i)){
                reset_episode(device, runner, env_i);
            }
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void summarize(DEVICE& device, const rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::EpisodeStatistics<typename rl::components::OnPolicyRunner<SPEC>::EPISODE_T, typename SPEC::TI>& statistics){
        using T = typename rl::components::OnPolicyRunner<SPEC>::EPISODE_T;
        using TI = typename SPEC::TI;
        using END_REASON = rl::components::on_policy_runner::EpisodeEndReason;
        statistics = {};
        for(TI step_i = 0; step_i <= DATASET_SPEC::STEPS_PER_ENV; step_i++){
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                const END_REASON reason = get(device, dataset.episode_end_reason, step_i, env_i);
                if(reason == END_REASON::NONE){
                    continue;
                }
                statistics.finished++;
                statistics.length_sum += (T)get(device, dataset.episode_length, step_i, env_i);
                statistics.return_sum += get(device, dataset.episode_return, step_i, env_i);
                statistics.terminated += reason == END_REASON::TERMINATED ? 1 : 0;
                statistics.time_limit += reason == END_REASON::TIME_LIMIT ? 1 : 0;
                statistics.forced += reason == END_REASON::FORCED ? 1 : 0;
            }
        }
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            const TI episode_step = get(device, runner.episode_step, env_i);
            if(episode_step > 0){
                statistics.in_progress++;
                statistics.in_progress_length_sum += (T)episode_step;
            }
        }
        statistics.mean_length = statistics.finished > 0 ? statistics.length_sum / (T)statistics.finished : (T)0;
        statistics.mean_return = statistics.finished > 0 ? statistics.return_sum / (T)statistics.finished : (T)0;
        statistics.terminated_share = statistics.finished > 0 ? (T)statistics.terminated / (T)statistics.finished : (T)0;
        statistics.mean_in_progress_length = statistics.in_progress > 0 ? statistics.in_progress_length_sum / (T)statistics.in_progress : (T)0;
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
    void observe_row(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, typename SPEC::TI row_i, RNG& rng){
        constexpr typename SPEC::TI N_ENVIRONMENTS = SPEC::N_ENVIRONMENTS;
        auto observations = view_range(device, dataset.all_observations, row_i * N_ENVIRONMENTS, tensor::ViewSpec<0, N_ENVIRONMENTS>{});
        observe(device, environment, runner.env_parameters, runner.states, typename SPEC::OBSERVATION{}, observations, rng);
        auto observations_privileged = view_range(device, dataset.all_observations_privileged, row_i * N_ENVIRONMENTS, tensor::ViewSpec<0, N_ENVIRONMENTS>{});
        if constexpr(SPEC::ASYMMETRIC_OBSERVATIONS){
            observe(device, environment, runner.env_parameters, runner.states, typename SPEC::OBSERVATION_PRIVILEGED{}, observations_privileged, rng);
        }
        else{
            copy(device, device, observations, observations_privileged);
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_actions_env(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, typename DATASET_SPEC::TI env_i, RNG& rng){
        using SPEC = typename DATASET_SPEC::SPEC;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        constexpr TI N_AGENTS = SPEC::N_AGENTS_PER_ENV;
        static_assert(ACTION_DIM % N_AGENTS == 0);
        constexpr TI PER_AGENT_ACTION_DIM = ACTION_DIM / N_AGENTS;
        static_assert(LOG_STD_SPEC::ROWS == 1);
        static_assert(LOG_STD_SPEC::COLS * N_AGENTS == ACTION_DIM);
        const TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
        T action_log_prob = 0;
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            const T action_mean = get(dataset.actions_mean, pos, action_i);
            const T action_log_std = get(log_std, 0, action_i % PER_AGENT_ACTION_DIM);
            const T action_std = math::exp(device.math, action_log_std);
            const T action = random::normal_distribution::sample(device.random, action_mean, action_std, rng);
            action_log_prob += random::normal_distribution::log_prob(device.random, action_mean, action_log_std, action);
            set(dataset.actions, pos, action_i, action);
            set(device, step_actions, action, env_i, action_i);
        }
        set(dataset.action_log_probs, pos, 0, action_log_prob);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
    void sample_actions(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, RNG& rng){
        using TI = typename DATASET_SPEC::TI;
        for(TI env_i = 0; env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS; env_i++){
            auto& rng_state = instance_rng<DATASET_SPEC::SPEC::N_ENVIRONMENTS>(rng, env_i);
            sample_actions_env(device, dataset, log_std, step_actions, step_i, env_i, rng_state);
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
    void prologue(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
        record_episode_start(device, dataset, runner);
        observe_row(device, dataset, runner, environment, 0, rng);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ACTOR, typename ACTOR_BUFFERS, typename RNG>
    void interlude(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& buffer, ACTOR& actor, ACTOR_BUFFERS& actor_buffers, RNG& rng, typename SPEC::TI step_i){
        using TI = typename SPEC::TI;
        constexpr TI N_ENVIRONMENTS = SPEC::N_ENVIRONMENTS;
        constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        Mode<mode::sequential::ResetMask<mode::Default<>, mode::sequential::ResetMaskSpecification<decltype(runner.reset)>>> mode_reset_mask;
        mode_reset_mask.mask = runner.reset;
        reset(device, actor, runner.policy_state, rng, mode_reset_mask);
        auto observations = view_range(device, dataset.all_observations, step_i * N_ENVIRONMENTS, tensor::ViewSpec<0, N_ENVIRONMENTS>{});
        using EVAL_INPUT_SHAPE = tensor::Prepend<typename SPEC::OBSERVATION::SHAPE, N_ENVIRONMENTS>;
        auto observations_reshaped = reshape_row_major(device, observations, EVAL_INPUT_SHAPE{});
        auto actions_mean = view(device, dataset.actions_mean, matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
        auto actions_mean_tensor = to_tensor(device, actions_mean);
        Mode<mode::Rollout<>> mode;
        evaluate_step(device, actor, observations_reshaped, runner.policy_state, actions_mean_tensor, actor_buffers, rng, mode);
        auto& last_layer = get_last_layer(actor);
        auto log_std = matrix_view(device, last_layer.log_std.parameters);
        sample_actions(device, dataset, log_std, buffer.actions, step_i, rng);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
    void epilogue(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& buffer, ENVIRONMENT& environment, RNG& rng, typename SPEC::TI step_i){
        step(device, environment, runner.env_parameters, runner.states, buffer.actions, buffer.next_states, rng);
        reward(device, environment, runner.env_parameters, runner.states, buffer.actions, buffer.next_states, buffer.rewards, rng);
        terminated(device, environment, runner.env_parameters, buffer.next_states, buffer.terminated, rng);
        copy(device, device, buffer.next_states, runner.states);
        record_transition(device, dataset, runner, buffer, step_i);
        sample_initial_parameters(device, environment, runner.env_parameters, runner.reset, rng);
        sample_initial_state(device, environment, runner.env_parameters, runner.states, runner.reset, rng);
        observe_row(device, dataset, runner, environment, step_i + 1, rng);
    }
    template <typename DEVICE, typename SPEC, typename ENVIRONMENT, typename RNG>
    void reset(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
        reset_episode(device, runner);
        sample_initial_parameters(device, environment, runner.env_parameters, runner.reset, rng);
        sample_initial_state(device, environment, runner.env_parameters, runner.states, runner.reset, rng);
    }
    template <typename DEVICE, typename SPEC, typename ENVIRONMENT, typename MASK_SPEC, typename RNG>
    void reset(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, const Tensor<MASK_SPEC>& mask, RNG& rng){
        reset_episode(device, runner, mask);
        sample_initial_parameters(device, environment, runner.env_parameters, mask, rng);
        sample_initial_state(device, environment, runner.env_parameters, runner.states, mask, rng);
    }
    template <typename DEVICE, typename SPEC, typename ENVIRONMENT, typename RNG>
    void init(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
        static_assert(rl_tools::utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
        init(device, runner);
        sample_initial_parameters(device, environment, runner.env_parameters, runner.reset, rng);
        sample_initial_state(device, environment, runner.env_parameters, runner.states, runner.reset, rng);
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename ACTOR, typename ACTOR_BUFFERS, typename RNG>
    void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& buffer, ENVIRONMENT& environment, ACTOR& actor, ACTOR_BUFFERS& actor_buffers, RNG& rng){
        static_assert(rl_tools::utils::typing::is_same_v<typename DATASET_SPEC::SPEC, SPEC>, "the dataset must be specified over the runner's specification");
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        using TI = typename SPEC::TI;
        if constexpr(SPEC::TRUNCATE_ON_EACH_ITERATION){
            reset(device, runner, environment, rng);
        }
        prologue(device, dataset, runner, environment, rng);
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            interlude(device, dataset, runner, buffer, actor, actor_buffers, rng, step_i);
            epilogue(device, dataset, runner, buffer, environment, rng, step_i);
        }
        runner.step += SPEC::N_ENVIRONMENTS * DATASET_SPEC::STEPS_PER_ENV;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
