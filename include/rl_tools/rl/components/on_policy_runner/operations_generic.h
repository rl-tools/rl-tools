#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include "on_policy_runner.h"
#include "../../../random/operations_generic_array.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::components::on_policy_runner::detail{
        template <typename DEVICE, typename STATE>
        RL_TOOLS_FUNCTION_PLACEMENT bool episode_due(DEVICE& device, const STATE& state, typename STATE::TI env_i){
            return get(device, state.truncated, env_i) || get(device, state.forced, env_i);
        }
        template <typename DEVICE, typename STATE>
        RL_TOOLS_FUNCTION_PLACEMENT void begin_episode_step(DEVICE& device, STATE& state, typename STATE::TI env_i, bool due){
            using T = typename STATE::EPISODE_T;
            using TI = typename STATE::TI;
            const TI episode_step = get(device, state.episode_step, env_i);
            const bool finished = due && episode_step > 0;
            set(device, state.finished, finished, env_i);
            set(device, state.finished_length, finished ? episode_step : (TI)0, env_i);
            set(device, state.finished_return, finished ? get(device, state.episode_return, env_i) : (T)0, env_i);
            set(device, state.finished_reason, finished ? get(device, state.end_reason, env_i) : EpisodeEndReason::NONE, env_i);
            set(device, state.reset, due, env_i);
            if(due){
                set(device, state.episode_step, (TI)0, env_i);
                set(device, state.episode_return, (T)0, env_i);
                set(device, state.end_reason, EpisodeEndReason::NONE, env_i);
                set(device, state.truncated, false, env_i);
                set(device, state.forced, false, env_i);
            }
        }
        template <typename DEVICE, typename STATE>
        RL_TOOLS_FUNCTION_PLACEMENT void end_episode_step(DEVICE& device, STATE& state, typename STATE::TI env_i, typename STATE::EPISODE_T reward){
            using TI = typename STATE::TI;
            const TI episode_step = get(device, state.episode_step, env_i) + 1;
            set(device, state.episode_step, episode_step, env_i);
            set(device, state.episode_return, get(device, state.episode_return, env_i) + reward, env_i);
            const bool terminated = get(device, state.terminated, env_i);
            const bool time_limit = state.episode_step_limit > 0 && episode_step >= state.episode_step_limit;
            const bool truncated = terminated || time_limit;
            set(device, state.truncated, truncated, env_i);
            if(truncated){
                set(device, state.end_reason, terminated ? EpisodeEndReason::TERMINATED : EpisodeEndReason::TIME_LIMIT, env_i);
            }
        }
        template <typename DEVICE, typename STATE>
        RL_TOOLS_FUNCTION_PLACEMENT void force_reset(DEVICE& device, STATE& state, typename STATE::TI env_i){
            set(device, state.forced, true, env_i);
            if(!get(device, state.truncated, env_i) && get(device, state.episode_step, env_i) > 0){
                set(device, state.end_reason, EpisodeEndReason::FORCED, env_i);
            }
        }
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        malloc(device, dataset.all_observations);
        malloc(device, dataset.all_observations_privileged);
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
        malloc(device, runner.terminated);
        malloc(device, runner.truncated);
        malloc(device, runner.reset);
        malloc(device, runner.forced);
        malloc(device, runner.episode_return);
        malloc(device, runner.end_reason);
        malloc(device, runner.finished);
        malloc(device, runner.finished_length);
        malloc(device, runner.finished_return);
        malloc(device, runner.finished_reason);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        free(device, runner.policy_state);
        free(device, runner.env_parameters);
        free(device, runner.states);
        free(device, runner.episode_step);
        free(device, runner.terminated);
        free(device, runner.truncated);
        free(device, runner.reset);
        free(device, runner.forced);
        free(device, runner.episode_return);
        free(device, runner.end_reason);
        free(device, runner.finished);
        free(device, runner.finished_length);
        free(device, runner.finished_return);
        free(device, runner.finished_reason);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        using T = typename rl::components::OnPolicyRunner<SPEC>::EPISODE_T;
        using TI = typename SPEC::TI;
        using END_REASON = rl::components::on_policy_runner::EpisodeEndReason;
        runner.step = 0;
        set_all(device, runner.episode_step, (TI)0);
        set_all(device, runner.terminated, false);
        set_all(device, runner.truncated, true);
        set_all(device, runner.reset, false);
        set_all(device, runner.forced, false);
        set_all(device, runner.episode_return, (T)0);
        set_all(device, runner.end_reason, END_REASON::NONE);
        set_all(device, runner.finished, false);
        set_all(device, runner.finished_length, (TI)0);
        set_all(device, runner.finished_return, (T)0);
        set_all(device, runner.finished_reason, END_REASON::NONE);
        runner.episode_step_limit = SPEC::STEP_LIMIT;
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = false;
#endif
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI STEPS>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::on_policy_runner::EpisodeLog<SPEC, STEPS>& log){
        malloc(device, log.finished);
        malloc(device, log.finished_length);
        malloc(device, log.finished_return);
        malloc(device, log.finished_reason);
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI STEPS>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::on_policy_runner::EpisodeLog<SPEC, STEPS>& log){
        free(device, log.finished);
        free(device, log.finished_length);
        free(device, log.finished_return);
        free(device, log.finished_reason);
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI STEPS>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::components::on_policy_runner::EpisodeLog<SPEC, STEPS>& log){
        using T = typename rl::components::on_policy_runner::EpisodeLog<SPEC, STEPS>::T;
        using TI = typename SPEC::TI;
        set_all(device, log.finished, false);
        set_all(device, log.finished_length, (TI)0);
        set_all(device, log.finished_return, (T)0);
        set_all(device, log.finished_reason, rl::components::on_policy_runner::EpisodeEndReason::NONE);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename SOURCE_SPEC::TI SOURCE_STEPS, typename TARGET_SPEC, typename TARGET_SPEC::TI TARGET_STEPS>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const rl::components::on_policy_runner::EpisodeLog<SOURCE_SPEC, SOURCE_STEPS>& source, rl::components::on_policy_runner::EpisodeLog<TARGET_SPEC, TARGET_STEPS>& target){
        copy(source_device, target_device, source.finished, target.finished);
        copy(source_device, target_device, source.finished_length, target.finished_length);
        copy(source_device, target_device, source.finished_return, target.finished_return);
        copy(source_device, target_device, source.finished_reason, target.finished_reason);
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI STEPS>
    RL_TOOLS_FUNCTION_PLACEMENT void record(DEVICE& device, rl::components::on_policy_runner::EpisodeLog<SPEC, STEPS>& log, const rl::components::OnPolicyRunner<SPEC>& runner, typename SPEC::TI step_i){
        utils::assert_exit(device, step_i < STEPS, "on_policy_runner::record: step index outside the episode log");
        auto finished = view(device, log.finished, step_i);
        auto finished_length = view(device, log.finished_length, step_i);
        auto finished_return = view(device, log.finished_return, step_i);
        auto finished_reason = view(device, log.finished_reason, step_i);
        copy(device, device, runner.finished, finished);
        copy(device, device, runner.finished_length, finished_length);
        copy(device, device, runner.finished_return, finished_return);
        copy(device, device, runner.finished_reason, finished_reason);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer){
        malloc(device, buffer.next_states);
        malloc(device, buffer.actions);
        malloc(device, buffer.rewards);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer){
        free(device, buffer.next_states);
        free(device, buffer.actions);
        free(device, buffer.rewards);
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::SPEC::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC_1>& d1, rl::components::on_policy_runner::Dataset<SPEC_2>& d2){
        using T = typename SPEC_1::SPEC::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, d1.all_observations, d2.all_observations);
        acc += abs_diff(device, d1.all_observations_privileged, d2.all_observations_privileged);
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
            TI finished_length_r1 = get(device, r1.finished_length, env_i);
            TI finished_length_r2 = get(device, r2.finished_length, env_i);
            acc += math::abs(device.math, (T)finished_length_r1 - (T)finished_length_r2);
            acc += math::abs(device.math, (T)get(device, r1.end_reason, env_i) - (T)get(device, r2.end_reason, env_i));
            acc += math::abs(device.math, (T)get(device, r1.finished_reason, env_i) - (T)get(device, r2.finished_reason, env_i));
        }
        acc += abs_diff(device, r1.episode_return, r2.episode_return);
        acc += abs_diff(device, r1.terminated, r2.terminated);
        acc += abs_diff(device, r1.truncated, r2.truncated);
        acc += abs_diff(device, r1.reset, r2.reset);
        acc += abs_diff(device, r1.forced, r2.forced);
        acc += abs_diff(device, r1.finished, r2.finished);
        acc += abs_diff(device, r1.finished_return, r2.finished_return);
        acc += math::abs(device.math, (T)r1.episode_step_limit - (T)r2.episode_step_limit);
        for(TI env_i = 0; env_i < SPEC_1::N_ENVIRONMENTS; env_i++){
            acc += abs_diff(device, get_ref(device, r1.states, env_i), get_ref(device, r2.states, env_i));
            acc += abs_diff(device, get_ref(device, r1.env_parameters, env_i), get_ref(device, r2.env_parameters, env_i));
        }
        return acc;
    }

    // batched ingest of a step's flags: the row of step_i and the reset column of step_i + 1
    namespace rl::components::on_policy_runner{
        namespace detail{
            template <auto INSTANCES, typename RNG, typename TI>
            RL_TOOLS_FUNCTION_PLACEMENT RNG& instance_rng(RNG& rng, TI){
                return rng;
            }
            template <auto INSTANCES, typename RNG_SPEC>
            RL_TOOLS_FUNCTION_PLACEMENT auto& instance_rng(devices::generic::random::ArrayENGINE<RNG_SPEC>& rng, typename RNG_SPEC::TI instance_i){
                static_assert(RNG_SPEC::NUM_RNGS >= INSTANCES, "the runner needs one RNG state per environment instance");
                return get(rng.states, 0, instance_i);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename REWARD_SPEC, typename TERMINATED_SPEC, typename TRUNCATED_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void record_step_env(DEVICE& device, Dataset<DATASET_SPEC>& dataset, typename DATASET_SPEC::TI step_i, const Tensor<REWARD_SPEC>& rewards, const Tensor<TERMINATED_SPEC>& terminated, const Tensor<TRUNCATED_SPEC>& truncated, typename DATASET_SPEC::TI env_i){
            using T = typename Dataset<DATASET_SPEC>::T;
            using TI = typename DATASET_SPEC::TI;
            constexpr TI N_ENVIRONMENTS = DATASET_SPEC::SPEC::N_ENVIRONMENTS;
            const TI pos = step_i * N_ENVIRONMENTS + env_i;
            const bool truncated_flag = get(device, truncated, env_i);
            set(dataset.rewards, pos, 0, (T)get(device, rewards, env_i));
            set(dataset.terminated, pos, 0, get(device, terminated, env_i) ? (T)1 : (T)0);
            set(dataset.truncated, pos, 0, truncated_flag ? (T)1 : (T)0);
            set(dataset.all_reset, pos + N_ENVIRONMENTS, 0, truncated_flag ? (T)1 : (T)0);
        }
        template <typename DEVICE, typename DATASET_SPEC, typename RESET_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void record_reset_env(DEVICE& device, Dataset<DATASET_SPEC>& dataset, const Tensor<RESET_SPEC>& reset, typename DATASET_SPEC::TI env_i){
            using T = typename Dataset<DATASET_SPEC>::T;
            set(dataset.reset, env_i, 0, get(device, reset, env_i) ? (T)1 : (T)0);
        }
        template <typename DATASET_SPEC, typename FLAG_SPEC>
        constexpr bool check_batched_flags(){
            static_assert(get<0>(typename FLAG_SPEC::SHAPE{}) == DATASET_SPEC::SPEC::N_ENVIRONMENTS, "the flag tensors must cover all environments");
            return true;
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename REWARD_SPEC, typename TERMINATED_SPEC, typename TRUNCATED_SPEC>
    void record_step(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, typename DATASET_SPEC::TI step_i, const Tensor<REWARD_SPEC>& rewards, const Tensor<TERMINATED_SPEC>& terminated, const Tensor<TRUNCATED_SPEC>& truncated){
        using TI = typename DATASET_SPEC::TI;
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, REWARD_SPEC>());
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, TERMINATED_SPEC>());
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, TRUNCATED_SPEC>());
        utils::assert_exit(device, step_i < DATASET_SPEC::STEPS_PER_ENV, "on_policy_runner::record_step: step index outside the dataset");
        for(TI env_i = 0; env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS; env_i++){
            rl::components::on_policy_runner::record_step_env(device, dataset, step_i, rewards, terminated, truncated, env_i);
        }
    }
    // the reset column of step 0: the mask applied at the start of the rollout
    template <typename DEVICE, typename DATASET_SPEC, typename RESET_SPEC>
    void record_reset(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const Tensor<RESET_SPEC>& reset){
        using TI = typename DATASET_SPEC::TI;
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, RESET_SPEC>());
        for(TI env_i = 0; env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS; env_i++){
            rl::components::on_policy_runner::record_reset_env(device, dataset, reset, env_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename ENVIRONMENT, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void begin_step(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
        bool any_due = false;
        if constexpr(SPEC::SYNCHRONIZED){
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                any_due = any_due || rl::components::on_policy_runner::detail::episode_due(device, runner, env_i);
            }
        }
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            const bool due = SPEC::SYNCHRONIZED ? any_due : rl::components::on_policy_runner::detail::episode_due(device, runner, env_i);
            rl::components::on_policy_runner::detail::begin_episode_step(device, runner, env_i, due);
        }
        sample_initial_parameters(device, environment, runner.env_parameters, runner.reset, rng);
        sample_initial_state(device, environment, runner.env_parameters, runner.states, runner.reset, rng);
    }
    template <typename DEVICE, typename SPEC, typename ENVIRONMENT, typename REWARD_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void end_step(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, const Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
        terminated(device, environment, runner.env_parameters, runner.states, runner.terminated, rng);
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            rl::components::on_policy_runner::detail::end_episode_step(device, runner, env_i, get(device, rewards, env_i));
        }
    }
    template <typename DEVICE, typename SPEC, typename ENVIRONMENT, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void end_step(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
        using T = typename rl::components::OnPolicyRunner<SPEC>::EPISODE_T;
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
        terminated(device, environment, runner.env_parameters, runner.states, runner.terminated, rng);
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            rl::components::on_policy_runner::detail::end_episode_step(device, runner, env_i, (T)0);
        }
    }
    template <typename DEVICE, typename SPEC, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void force_reset(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        using TI = typename SPEC::TI;
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            rl::components::on_policy_runner::detail::force_reset(device, runner, env_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename MASK_SPEC, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void force_reset(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, const Tensor<MASK_SPEC>& mask){
        using TI = typename SPEC::TI;
        static_assert(get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::N_ENVIRONMENTS);
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            if(get(device, mask, env_i)){
                rl::components::on_policy_runner::detail::force_reset(device, runner, env_i);
            }
        }
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI STEPS, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void summarize(DEVICE& device, const rl::components::on_policy_runner::EpisodeLog<SPEC, STEPS>& log, const rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::EpisodeStatistics<typename rl::components::OnPolicyRunner<SPEC>::EPISODE_T, typename SPEC::TI>& statistics){
        using T = typename rl::components::OnPolicyRunner<SPEC>::EPISODE_T;
        using TI = typename SPEC::TI;
        using END_REASON = rl::components::on_policy_runner::EpisodeEndReason;
        statistics = {};
        for(TI step_i = 0; step_i < STEPS; step_i++){
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                if(!get(device, log.finished, step_i, env_i)){
                    continue;
                }
                statistics.finished++;
                statistics.length_sum += (T)get(device, log.finished_length, step_i, env_i);
                statistics.return_sum += get(device, log.finished_return, step_i, env_i);
                const END_REASON reason = get(device, log.finished_reason, step_i, env_i);
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
    namespace rl::components::on_policy_runner{
        template <typename DEVICE, typename SPEC, typename ENVIRONMENT>
        void render(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment){
            rl_tools::render(device, environment, runner.env_parameters, runner.states, runner.reset);
        }
        template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
        void observe_row(DEVICE& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, typename SPEC::TI row_i, RNG& rng){
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
        RL_TOOLS_FUNCTION_PLACEMENT void sample_actions_env(DEVICE& device, Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, typename DATASET_SPEC::TI env_i, RNG& rng){
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
        void sample_actions(DEVICE& device, Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, RNG& rng){
            using TI = typename DATASET_SPEC::TI;
            for(TI env_i = 0; env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS; env_i++){
                auto& rng_state = detail::instance_rng<DATASET_SPEC::SPEC::N_ENVIRONMENTS>(rng, env_i);
                sample_actions_env(device, dataset, log_std, step_actions, step_i, env_i, rng_state);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
        void prologue(DEVICE& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
            record_reset(device, dataset, runner.reset);
            observe_row(device, dataset, runner, environment, 0, rng);
        }
        template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ACTOR, typename ACTOR_BUFFERS, typename RNG>
        void interlude(DEVICE& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, Buffer<SPEC>& buffer, ACTOR& actor, ACTOR_BUFFERS& actor_buffers, RNG& rng, typename SPEC::TI step_i){
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
        void epilogue(DEVICE& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, Buffer<SPEC>& buffer, ENVIRONMENT& environment, RNG& rng, typename SPEC::TI step_i){
            step(device, environment, runner.env_parameters, runner.states, buffer.actions, buffer.next_states, rng);
            reward(device, environment, runner.env_parameters, runner.states, buffer.actions, buffer.next_states, buffer.rewards, rng);
            copy(device, device, buffer.next_states, runner.states);
            end_step(device, runner, environment, buffer.rewards, rng);
            record_step(device, dataset, step_i, buffer.rewards, runner.terminated, runner.truncated);
            begin_step(device, runner, environment, rng);
            render(device, runner, environment);
            observe_row(device, dataset, runner, environment, step_i + 1, rng);
        }
        template <typename DEVICE, typename SPEC, typename ENVIRONMENT, typename RNG>
        void reset(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
            begin_step(device, runner, environment, rng);
            render(device, runner, environment);
        }
    }
    template <typename DEVICE, typename SPEC, typename ENVIRONMENT, typename RNG>
    void init(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
        static_assert(rl_tools::utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
        init(device, runner);
        rl::components::on_policy_runner::reset(device, runner, environment, rng);
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
            force_reset(device, runner);
            rl::components::on_policy_runner::reset(device, runner, environment, rng);
        }
        rl::components::on_policy_runner::prologue(device, dataset, runner, environment, rng);
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            rl::components::on_policy_runner::interlude(device, dataset, runner, buffer, actor, actor_buffers, rng, step_i);
            rl::components::on_policy_runner::epilogue(device, dataset, runner, buffer, environment, rng, step_i);
        }
        runner.step += SPEC::N_ENVIRONMENTS * DATASET_SPEC::STEPS_PER_ENV;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
