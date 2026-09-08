#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_COMMON_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_COMMON_H

#include "on_policy_runner.h"
#include "../../environments/batch/operations_generic_common.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename DATASET, typename RUNNER>
    RL_TOOLS_FUNCTION_PLACEMENT void record_transition(DEVICE& device, DATASET& dataset, RUNNER& runner, typename DATASET::T reward, bool terminated, typename RUNNER::TI step_i, typename RUNNER::TI env_i){
        using TI = typename RUNNER::TI;
        const TI pos = step_i * RUNNER::SPEC::N_ENVIRONMENTS + env_i;
        const TI episode_step = get(device, runner.episode_step, env_i) + 1;
        const bool reset = terminated || (RUNNER::SPEC::STEP_LIMIT > 0 && episode_step >= RUNNER::SPEC::STEP_LIMIT);
        set(dataset.rewards, pos, 0, reward);
        set(dataset.terminated, pos, 0, terminated);
        set(dataset.truncated, pos, 0, reset);
        set(dataset.all_reset, pos + RUNNER::SPEC::N_ENVIRONMENTS, 0, reset);
        set(device, runner.reset, reset, env_i);
        set(device, runner.episode_step, reset ? (TI)0 : episode_step, env_i);
    }
    template <typename DEVICE, typename RUNNER>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_episode(DEVICE& device, RUNNER& runner, typename RUNNER::TI env_i){
        set(device, runner.reset, true, env_i);
        set(device, runner.episode_step, (typename RUNNER::TI)0, env_i);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_actions_env(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, typename DATASET_SPEC::TI env_i, RNG& rng){
        using SPEC = typename DATASET_SPEC::SPEC;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        constexpr TI ACTION_DIM = SPEC::BATCH_ENVIRONMENT::ACTION_DIM;
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
    template <typename DEVICE, typename DATASET_SPEC, typename ENVIRONMENT, typename PARAMETERS, typename STATE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe_instance(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, ENVIRONMENT& environment, PARAMETERS& parameters, STATE& state, typename DATASET_SPEC::TI row_i, typename DATASET_SPEC::TI env_i, RNG& rng){
        using SPEC = typename DATASET_SPEC::SPEC;
        const auto pos = row_i * SPEC::N_ENVIRONMENTS + env_i;
        observe_instance(device, environment, parameters, state, typename SPEC::OBSERVATION{}, dataset.all_observations, pos, rng);
        if constexpr(SPEC::ASYMMETRIC_OBSERVATIONS){
            observe_instance(device, environment, parameters, state, typename SPEC::OBSERVATION_PRIVILEGED{}, dataset.all_observations_privileged, pos, rng);
        }
        else{
            auto observation = view(device, dataset.all_observations, pos);
            auto observation_privileged = view(device, dataset.all_observations_privileged, pos);
            copy(device, device, observation, observation_privileged);
        }
    }
    template <typename DEVICE, typename RUNNER, typename ENV_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_instance(DEVICE& device, RUNNER& runner, Tensor<ENV_SPEC>& environments, typename RUNNER::TI env_i, RNG& rng){
        sample_initial_parameters_instance(device, environments, runner.env_parameters, env_i, rng);
        sample_initial_state_instance(device, environments, runner.env_parameters, runner.states, env_i, rng);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
