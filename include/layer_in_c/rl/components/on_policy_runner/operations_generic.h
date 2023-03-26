#ifndef LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include "on_policy_runner.h"
#include "operations_generic_per_env.h"

namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        malloc(device, dataset.data);
        using DATA_SPEC = typename decltype(dataset.data)::SPEC;
        using TI = typename SPEC::SPEC::TI;
        TI pos = 0;
        dataset.all_observations = view<DEVICE, DATA_SPEC, decltype(dataset.all_observations)::ROWS, decltype(dataset.all_observations)::COLS>(device, dataset.data, 0, pos);
        dataset.observations     = view<DEVICE, DATA_SPEC, decltype(dataset.observations    )::ROWS, decltype(dataset.observations    )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.observations    )::COLS;
        dataset.actions          = view<DEVICE, DATA_SPEC, decltype(dataset.actions         )::ROWS, decltype(dataset.actions         )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.actions         )::COLS;
        dataset.action_log_probs = view<DEVICE, DATA_SPEC, decltype(dataset.action_log_probs)::ROWS, decltype(dataset.action_log_probs)::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.action_log_probs)::COLS;
        dataset.rewards          = view<DEVICE, DATA_SPEC, decltype(dataset.rewards         )::ROWS, decltype(dataset.rewards         )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.rewards         )::COLS;
        dataset.terminated       = view<DEVICE, DATA_SPEC, decltype(dataset.terminated      )::ROWS, decltype(dataset.terminated      )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.terminated      )::COLS;
        dataset.truncated        = view<DEVICE, DATA_SPEC, decltype(dataset.truncated       )::ROWS, decltype(dataset.truncated       )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.truncated       )::COLS;
        dataset.all_values       = view<DEVICE, DATA_SPEC, decltype(dataset.all_values      )::ROWS, decltype(dataset.all_values      )::COLS>(device, dataset.data, 0, pos);
        dataset.values           = view<DEVICE, DATA_SPEC, decltype(dataset.values          )::ROWS, decltype(dataset.values          )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.values          )::COLS;
        dataset.advantages       = view<DEVICE, DATA_SPEC, decltype(dataset.advantages      )::ROWS, decltype(dataset.advantages      )::COLS>(device, dataset.data, 0, pos); pos += decltype(dataset.advantages      )::COLS;
        dataset.target_values    = view<DEVICE, DATA_SPEC, decltype(dataset.target_values   )::ROWS, decltype(dataset.target_values   )::COLS>(device, dataset.data, 0, pos);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        free(device, dataset.data);
        dataset.all_observations._data = nullptr;
        dataset.observations    ._data = nullptr;
        dataset.actions         ._data = nullptr;
        dataset.action_log_probs._data = nullptr;
        dataset.rewards         ._data = nullptr;
        dataset.terminated      ._data = nullptr;
        dataset.truncated       ._data = nullptr;
        dataset.all_values      ._data = nullptr;
        dataset.values          ._data = nullptr;
        dataset.advantages      ._data = nullptr;
        dataset.target_values   ._data = nullptr;
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
#ifdef LAYER_IN_C_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
    }
    namespace rl::components::on_policy_runner{
        template <typename DEVICE, typename OBSERVATIONS_SPEC, typename SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
        void prologue(DEVICE& device, Matrix<OBSERVATIONS_SPEC>& observations, rl::components::OnPolicyRunner<SPEC>& runner, RNG& rng, typename DEVICE::index_t step_i){
            static_assert(OBSERVATIONS_SPEC::ROWS == SPEC::N_ENVIRONMENTS);
            static_assert(OBSERVATIONS_SPEC::COLS == SPEC::ENVIRONMENT::OBSERVATION_DIM);
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                per_env::prologue(device, observations, runner, rng, env_i);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
        void epilogue(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, RNG& rng, typename DEVICE::index_t step_i){
            using SPEC = typename DATASET_SPEC::SPEC;
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                per_env::epilogue(device, dataset, runner, actions, action_log_std, rng, pos, env_i);
            }
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename ACTOR, typename RNG> // todo: make this not PPO but general policy with output distribution
    void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, ACTOR& actor, typename ACTOR::template Buffers<DATASET_SPEC::SPEC::N_ENVIRONMENTS>& policy_eval_buffers, RNG& rng){
#ifdef LAYER_IN_C_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        using SPEC = typename DATASET_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            auto actions = view(device, dataset.actions, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations = view(device, dataset.observations, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);

            rl::components::on_policy_runner::prologue(device, observations, runner, rng, step_i);
            evaluate(device, actor, observations, actions, policy_eval_buffers);
            rl::components::on_policy_runner::epilogue(device, dataset, runner, actions, actor.action_log_std.parameters, rng, step_i);
        }
        // final observation
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            auto& env = get(runner.environments, 0, env_i);
            auto& state = get(runner.states, 0, env_i);
            TI pos = DATASET_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i;
            auto observation = view(device, dataset.all_observations, matrix::ViewSpec<1, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), pos, 0);
            observe(device, env, state, observation);
        }
        runner.step += SPEC::N_ENVIRONMENTS * DATASET_SPEC::STEPS_PER_ENV;
    }
}
#endif
