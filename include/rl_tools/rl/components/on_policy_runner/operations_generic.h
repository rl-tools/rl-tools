#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include "on_policy_runner.h"
#include "operations_generic_per_env.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        malloc(device, dataset.data);
        using DATA_SPEC = typename decltype(dataset.data)::SPEC;
        using TI = typename SPEC::SPEC::TI;
        TI pos = 0;
        dataset.all_observations_privileged = view<DEVICE, DATA_SPEC, decltype(dataset.all_observations_privileged)::ROWS, decltype(dataset.all_observations_privileged)::COLS>(device, dataset.data, 0, pos); pos += (SPEC::ASYMMETRIC_OBSERVATIONS ? decltype(dataset.all_observations_privileged)::COLS : 0);
        dataset.observations                = view<DEVICE, DATA_SPEC, decltype(dataset.observations               )::ROWS, decltype(dataset.observations               )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.observations           )::COLS;
        dataset.actions_mean                = view<DEVICE, DATA_SPEC, decltype(dataset.actions_mean               )::ROWS, decltype(dataset.actions_mean               )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.actions_mean           )::COLS;
        dataset.actions                     = view<DEVICE, DATA_SPEC, decltype(dataset.actions                    )::ROWS, decltype(dataset.actions                    )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.actions                )::COLS;
        dataset.action_log_probs            = view<DEVICE, DATA_SPEC, decltype(dataset.action_log_probs           )::ROWS, decltype(dataset.action_log_probs           )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.action_log_probs       )::COLS;
        dataset.rewards                     = view<DEVICE, DATA_SPEC, decltype(dataset.rewards                    )::ROWS, decltype(dataset.rewards                    )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.rewards                )::COLS;
        dataset.terminated                  = view<DEVICE, DATA_SPEC, decltype(dataset.terminated                 )::ROWS, decltype(dataset.terminated                 )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.terminated             )::COLS;
        dataset.truncated                   = view<DEVICE, DATA_SPEC, decltype(dataset.truncated                  )::ROWS, decltype(dataset.truncated                  )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.truncated              )::COLS;
        dataset.all_reset                   = view<DEVICE, DATA_SPEC, decltype(dataset.all_reset                  )::ROWS, decltype(dataset.all_reset                  )::COLS>(device, dataset.data, 0, pos);
        dataset.reset                       = view<DEVICE, DATA_SPEC, decltype(dataset.reset                      )::ROWS, decltype(dataset.reset                      )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.reset                  )::COLS;
        dataset.all_values                  = view<DEVICE, DATA_SPEC, decltype(dataset.all_values                 )::ROWS, decltype(dataset.all_values                 )::COLS>(device, dataset.data, 0, pos);
        dataset.values                      = view<DEVICE, DATA_SPEC, decltype(dataset.values                     )::ROWS, decltype(dataset.values                     )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.values                 )::COLS;
        dataset.advantages                  = view<DEVICE, DATA_SPEC, decltype(dataset.advantages                 )::ROWS, decltype(dataset.advantages                 )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.advantages             )::COLS;
        dataset.target_values               = view<DEVICE, DATA_SPEC, decltype(dataset.target_values              )::ROWS, decltype(dataset.target_values              )::COLS>(device, dataset.data, 0, pos);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        free(device, dataset.data);
        dataset.all_observations_privileged._data = nullptr;
        dataset.observations               ._data = nullptr;
        dataset.actions_mean               ._data = nullptr;
        dataset.actions                    ._data = nullptr;
        dataset.action_log_probs           ._data = nullptr;
        dataset.rewards                    ._data = nullptr;
        dataset.terminated                 ._data = nullptr;
        dataset.truncated                  ._data = nullptr;
        dataset.all_values                 ._data = nullptr;
        dataset.values                     ._data = nullptr;
        dataset.advantages                 ._data = nullptr;
        dataset.target_values              ._data = nullptr;
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        malloc(device, runner.environments);
        malloc(device, runner.env_parameters);
        malloc(device, runner.states);
        malloc(device, runner.policy_state);
        malloc(device, runner.episode_step);
        malloc(device, runner.episode_return);
        malloc(device, runner.truncated);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        free(device, runner.environments);
        free(device, runner.env_parameters);
        free(device, runner.states);
        free(device, runner.policy_state);
        free(device, runner.episode_step);
        free(device, runner.episode_return);
        free(device, runner.truncated);
    }
    template <typename DEVICE, typename SPEC, typename ENV_SPEC, typename PARAM_SPEC, typename ACTOR, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, Tensor<ENV_SPEC> environments, Tensor<PARAM_SPEC> parameters, ACTOR& actor, RNG& rng){
        using TI = typename SPEC::TI;
        set_all(device, runner.episode_step, 0);
        set_all(device, runner.episode_return, 0);
        set_all(device, runner.truncated, true);
        for(TI env_i=0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            set(runner.environments, 0, env_i, get_ref(device, environments, env_i));
            set(runner.env_parameters, 0, env_i, get_ref(device, parameters, env_i));
        }
        reset(device, actor, runner.policy_state, rng);
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
    }
    namespace rl::components::on_policy_runner{
        template <typename DEVICE, typename OBSERVATIONS_PRIVILEGED_SPEC, typename OBSERVATIONS_SPEC, typename SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
        RL_TOOLS_FUNCTION_PLACEMENT void prologue(DEVICE& device, Matrix<OBSERVATIONS_PRIVILEGED_SPEC>& observations_privileged, Matrix<OBSERVATIONS_SPEC>& observations, rl::components::OnPolicyRunner<SPEC>& runner, RNG& rng, typename DEVICE::index_t step_i){
            static_assert(OBSERVATIONS_SPEC::ROWS == SPEC::N_ENVIRONMENTS);
            static_assert(OBSERVATIONS_SPEC::COLS == SPEC::ENVIRONMENT::Observation::DIM);
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                per_env::prologue(device, observations_privileged, observations, runner, rng, env_i);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename ACTIONS_MEAN_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
        RL_TOOLS_FUNCTION_PLACEMENT void epilogue(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, Matrix<ACTIONS_MEAN_SPEC>& actions_mean, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, RNG& rng, typename DEVICE::index_t step_i){
            using SPEC = typename DATASET_SPEC::SPEC;
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                per_env::epilogue(device, dataset, runner, actions_mean, actions, action_log_std, rng, pos, env_i);
            }
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename ACTOR, typename ACTOR_BUFFER, typename RNG> // todo: make this not PPO but general policy with output distribution
    RL_TOOLS_FUNCTION_PLACEMENT void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, ACTOR& actor, ACTOR_BUFFER& policy_eval_buffers, RNG& rng){
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        using SPEC = typename DATASET_SPEC::SPEC;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        if constexpr(SPEC::TRUNCATE_ON_EACH_ITERATION){
            set_all(device, runner.truncated, true);
        }
        for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
            set(dataset.reset, env_i, 0, get(runner.truncated, 0, env_i));
        }
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            auto actions_mean            = view(device, dataset.actions_mean               , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>()                , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto actions                 = view(device, dataset.actions                    , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>()                , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations_privileged = view(device, dataset.all_observations_privileged, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ObservationPrivileged::DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations            = view(device, dataset.observations               , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::Observation::DIM>()          , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto truncated_view = view(device, runner.truncated);
            Mode<mode::sequential::ResetMask<mode::Default<>, mode::sequential::ResetMaskSpecification<decltype(truncated_view)>>> mode_reset_mask;
            mode_reset_mask.mask = truncated_view;
            reset(device, actor, runner.policy_state, rng, mode_reset_mask); // it is important that this happens before prologue because prologue resets the truncated flags on the runner
            rl::components::on_policy_runner::prologue(device, observations_privileged, observations, runner, rng, step_i);
            auto observations_tensor = to_tensor(device, observations);
            auto actions_mean_tensor = to_tensor(device, actions_mean);
            Mode<mode::Rollout<>> mode;
            evaluate_step(device, actor, observations_tensor, runner.policy_state, actions_mean_tensor, policy_eval_buffers, rng, mode);
            auto& last_layer = get_last_layer(actor);
            auto log_std = matrix_view(device, last_layer.log_std.parameters);
            rl::components::on_policy_runner::epilogue(device, dataset, runner, actions_mean, actions, log_std, rng, step_i);
        }
        // final observation
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            auto& env = get(runner.environments, 0, env_i);
            auto& state = get(runner.states, 0, env_i);
            auto& parameters = get(runner.env_parameters, 0, env_i);
            TI row_i = DATASET_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i;
            auto observation = row(device, dataset.all_observations_privileged, row_i);
            observe(device, env, parameters, state, typename DATASET_SPEC::SPEC::ENVIRONMENT::ObservationPrivileged{}, observation, rng);
        }
        runner.step += SPEC::N_ENVIRONMENTS * DATASET_SPEC::STEPS_PER_ENV;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::SPEC::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC_1>& d1, rl::components::on_policy_runner::Dataset<SPEC_2>& d2){
        using T = typename SPEC_1::SPEC::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, d1.all_observations_privileged, d2.all_observations_privileged);
        acc += abs_diff(device, d1.observations, d2.observations);
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
            TI episode_step_r1 = get(r1.episode_step, 0, env_i);
            TI episode_step_r2 = get(r2.episode_step, 0, env_i);
            acc += math::abs(device.math, (T)episode_step_r1 - (T)episode_step_r2);
        }
        acc += abs_diff(device, r1.episode_return, r2.episode_return);
        acc += abs_diff(device, r1.truncated, r2.truncated);
        for(TI env_i = 0; env_i < SPEC_1::N_ENVIRONMENTS; env_i++){
            acc += abs_diff(device, get(r1.states, 0, env_i), get(r2.states, 0, env_i));
            acc += abs_diff(device, get(r1.env_parameters, 0, env_i), get(r2.env_parameters, 0, env_i));
        }
        return acc;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
