#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include "on_policy_runner.h"
#include "operations_generic_per_env.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        malloc(device, dataset.data);
        using DATA_SPEC = typename decltype(dataset.data)::SPEC;
        using TI = typename SPEC::SPEC::TI;
        TI pos = 0;
        dataset.all_observations            = view<DEVICE, DATA_SPEC, decltype(dataset.all_observations           )::ROWS, decltype(dataset.all_observations           )::COLS>(device, dataset.data, 0, pos);
        dataset.observations                = view<DEVICE, DATA_SPEC, decltype(dataset.observations               )::ROWS, decltype(dataset.observations               )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.observations           )::COLS;
        dataset.all_observations_normalized = view<DEVICE, DATA_SPEC, decltype(dataset.all_observations_normalized)::ROWS, decltype(dataset.all_observations_normalized)::COLS>(device, dataset.data, 0, pos);
        dataset.observations_normalized     = view<DEVICE, DATA_SPEC, decltype(dataset.observations_normalized    )::ROWS, decltype(dataset.observations_normalized    )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.observations_normalized)::COLS;
        dataset.actions_mean                = view<DEVICE, DATA_SPEC, decltype(dataset.actions_mean               )::ROWS, decltype(dataset.actions_mean               )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.actions_mean           )::COLS;
        dataset.actions                     = view<DEVICE, DATA_SPEC, decltype(dataset.actions                    )::ROWS, decltype(dataset.actions                    )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.actions                )::COLS;
        dataset.action_log_probs            = view<DEVICE, DATA_SPEC, decltype(dataset.action_log_probs           )::ROWS, decltype(dataset.action_log_probs           )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.action_log_probs       )::COLS;
        dataset.rewards                     = view<DEVICE, DATA_SPEC, decltype(dataset.rewards                    )::ROWS, decltype(dataset.rewards                    )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.rewards                )::COLS;
        dataset.terminated                  = view<DEVICE, DATA_SPEC, decltype(dataset.terminated                 )::ROWS, decltype(dataset.terminated                 )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.terminated             )::COLS;
        dataset.truncated                   = view<DEVICE, DATA_SPEC, decltype(dataset.truncated                  )::ROWS, decltype(dataset.truncated                  )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.truncated              )::COLS;
        dataset.all_values                  = view<DEVICE, DATA_SPEC, decltype(dataset.all_values                 )::ROWS, decltype(dataset.all_values                 )::COLS>(device, dataset.data, 0, pos);
        dataset.values                      = view<DEVICE, DATA_SPEC, decltype(dataset.values                     )::ROWS, decltype(dataset.values                     )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.values                 )::COLS;
        dataset.advantages                  = view<DEVICE, DATA_SPEC, decltype(dataset.advantages                 )::ROWS, decltype(dataset.advantages                 )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.advantages             )::COLS;
        dataset.target_values               = view<DEVICE, DATA_SPEC, decltype(dataset.target_values              )::ROWS, decltype(dataset.target_values              )::COLS>(device, dataset.data, 0, pos);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        free(device, dataset.data);
        dataset.all_observations           ._data = nullptr;
        dataset.observations               ._data = nullptr;
        dataset.all_observations_normalized._data = nullptr;
        dataset.observations_normalized    ._data = nullptr;
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
    void malloc(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        malloc(device, runner.environments);
        malloc(device, runner.states);
        malloc(device, runner.episode_step);
        malloc(device, runner.episode_return);
        malloc(device, runner.truncated);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        free(device, runner.environments);
        free(device, runner.states);
        free(device, runner.episode_step);
        free(device, runner.episode_return);
        free(device, runner.truncated);
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void init(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, typename SPEC::ENVIRONMENT environments[SPEC::N_ENVIRONMENTS], RNG& rng){
        using TI = typename SPEC::TI;
        set_all(device, runner.episode_step, 0);
        set_all(device, runner.episode_return, 0);
        set_all(device, runner.truncated, true);
        for(TI env_i=0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            set(runner.environments, 0, env_i, environments[env_i]);
        }
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
    }
    namespace rl::components::on_policy_runner{
        template <typename DEVICE, typename OBSERVATIONS_SPEC, typename OBSERVATIONS_NORMALIZED_SPEC, typename SPEC, typename OBSERVATIONS_MEAN_SPEC, typename OBSERVATIONS_STD_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
        void prologue(DEVICE& device, Matrix<OBSERVATIONS_SPEC>& observations, Matrix<OBSERVATIONS_NORMALIZED_SPEC>& observations_normalized, rl::components::OnPolicyRunner<SPEC>& runner, Matrix<OBSERVATIONS_MEAN_SPEC>& observations_mean, Matrix<OBSERVATIONS_STD_SPEC>& observations_std, RNG& rng, typename DEVICE::index_t step_i){
            static_assert(OBSERVATIONS_SPEC::ROWS == SPEC::N_ENVIRONMENTS);
            static_assert(OBSERVATIONS_SPEC::COLS == SPEC::ENVIRONMENT::OBSERVATION_DIM);
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                per_env::prologue(device, observations, observations_normalized, runner, observations_mean, observations_std, rng, env_i);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename ACTIONS_MEAN_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
        void epilogue(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, Matrix<ACTIONS_MEAN_SPEC>& actions_mean, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, RNG& rng, typename DEVICE::index_t step_i){
            using SPEC = typename DATASET_SPEC::SPEC;
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                per_env::epilogue(device, dataset, runner, actions_mean, actions, action_log_std, rng, pos, env_i);
            }
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename ACTOR, typename OBSERVATION_MEAN_SPEC, typename OBSERVATION_STD_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
    void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, ACTOR& actor, typename ACTOR::template Buffer<DATASET_SPEC::SPEC::N_ENVIRONMENTS>& policy_eval_buffers, Matrix<OBSERVATION_MEAN_SPEC>& observation_mean, Matrix<OBSERVATION_STD_SPEC>& observation_std, RNG& rng){
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        using SPEC = typename DATASET_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            auto actions_mean            = view(device, dataset.actions_mean           , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>()     , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto actions                 = view(device, dataset.actions                , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>()     , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations            = view(device, dataset.observations           , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations_normalized = view(device, dataset.observations_normalized, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);
            rl::components::on_policy_runner::prologue(device, observations, observations_normalized, runner, observation_mean, observation_std, rng, step_i);
            evaluate(device, actor, observations_normalized, actions_mean, policy_eval_buffers);
            rl::components::on_policy_runner::epilogue(device, dataset, runner, actions_mean, actions, actor.log_std.parameters, rng, step_i);
        }
        // final observation
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            auto& env = get(runner.environments, 0, env_i);
            auto& state = get(runner.states, 0, env_i);
            TI row_i = DATASET_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i;
            auto observation = row(device, dataset.all_observations, row_i);
            observe(device, env, state, observation, rng);
            auto observation_normalized = row(device, dataset.all_observations_normalized, row_i);
            normalize(device, observation_mean, observation_std, observation, observation_normalized);
        }
        runner.step += SPEC::N_ENVIRONMENTS * DATASET_SPEC::STEPS_PER_ENV;
    }
    template <typename DEVICE, typename DATASET_SPEC, typename ACTOR, typename RNG> // todo: make this not PPO but general policy with output distribution
    void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, ACTOR& actor, typename ACTOR::template Buffer<DATASET_SPEC::SPEC::N_ENVIRONMENTS>& policy_eval_buffers, RNG& rng){
        using T = typename DATASET_SPEC::SPEC::T;
        using TI = typename DEVICE::index_t;
        using ENVIRONMENT = typename DATASET_SPEC::SPEC::ENVIRONMENT;
        MatrixDynamic<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_mean;
        MatrixDynamic<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_std;
        malloc(device, observation_mean);
        malloc(device, observation_std);
        set_all(device, observation_mean, 0);
        set_all(device, observation_std, 1);
        collect(device, dataset, runner, actor, policy_eval_buffers, observation_mean, observation_std, rng);
        free(device, observation_mean);
        free(device, observation_std);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
