#ifndef LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include "on_policy_runner.h"
#include "operations_generic_per_env.h"

namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer){
        malloc(device, buffer.data);
        using BUFFER = rl::components::on_policy_runner::Buffer<SPEC>;
        using DATA_SPEC = typename decltype(buffer.data)::SPEC;
        using TI = typename SPEC::SPEC::TI;
        TI pos = 0;
        buffer.all_observations = view<DEVICE, DATA_SPEC, decltype(buffer.all_observations)::ROWS, decltype(buffer.all_observations)::COLS>(device, buffer.data, 0, pos);
        buffer.observations     = view<DEVICE, DATA_SPEC, decltype(buffer.observations    )::ROWS, decltype(buffer.observations    )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.observations    )::COLS;
        buffer.actions          = view<DEVICE, DATA_SPEC, decltype(buffer.actions         )::ROWS, decltype(buffer.actions         )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.actions         )::COLS;
        buffer.action_log_probs = view<DEVICE, DATA_SPEC, decltype(buffer.action_log_probs)::ROWS, decltype(buffer.action_log_probs)::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.action_log_probs)::COLS;
        buffer.rewards          = view<DEVICE, DATA_SPEC, decltype(buffer.rewards         )::ROWS, decltype(buffer.rewards         )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.rewards         )::COLS;
        buffer.terminated       = view<DEVICE, DATA_SPEC, decltype(buffer.terminated      )::ROWS, decltype(buffer.terminated      )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.terminated      )::COLS;
        buffer.truncated        = view<DEVICE, DATA_SPEC, decltype(buffer.truncated       )::ROWS, decltype(buffer.truncated       )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.truncated       )::COLS;
        buffer.all_values       = view<DEVICE, DATA_SPEC, decltype(buffer.all_values      )::ROWS, decltype(buffer.all_values      )::COLS>(device, buffer.data, 0, pos);
        buffer.values           = view<DEVICE, DATA_SPEC, decltype(buffer.values          )::ROWS, decltype(buffer.values          )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.values          )::COLS;
        buffer.advantages       = view<DEVICE, DATA_SPEC, decltype(buffer.advantages      )::ROWS, decltype(buffer.advantages      )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.advantages      )::COLS;
        buffer.target_values    = view<DEVICE, DATA_SPEC, decltype(buffer.target_values   )::ROWS, decltype(buffer.target_values   )::COLS>(device, buffer.data, 0, pos);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer){
        free(device, buffer.data);
        buffer.all_observations._data = nullptr;
        buffer.observations    ._data = nullptr;
        buffer.actions         ._data = nullptr;
        buffer.action_log_probs._data = nullptr;
        buffer.rewards         ._data = nullptr;
        buffer.terminated      ._data = nullptr;
        buffer.truncated       ._data = nullptr;
        buffer.all_values      ._data = nullptr;
        buffer.values          ._data = nullptr;
        buffer.advantages      ._data = nullptr;
        buffer.target_values   ._data = nullptr;
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
        template <typename DEVICE, typename BUFFER_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
        void prologue(DEVICE& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, RNG& rng, typename DEVICE::index_t step_i){
            using SPEC = typename BUFFER_SPEC::SPEC;
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                per_env::prologue(device, buffer, runner, rng, pos, env_i);
            }
        }
        template <typename DEVICE, typename BUFFER_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
        void epilogue(DEVICE& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, RNG& rng, typename DEVICE::index_t step_i){
            using SPEC = typename BUFFER_SPEC::SPEC;
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                per_env::epilogue(device, buffer, runner, actions, action_log_std, rng, pos, env_i);
            }
        }
    }
    template <typename DEVICE, typename BUFFER_SPEC, typename ACTOR, typename RNG> // todo: make this not PPO but general policy with output distribution
    void collect(DEVICE& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, ACTOR& actor, typename ACTOR::template Buffers<BUFFER_SPEC::SPEC::N_ENVIRONMENTS>& policy_eval_buffers, RNG& rng){
#ifdef LAYER_IN_C_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        using SPEC = typename BUFFER_SPEC::SPEC;
        using BUFFER = rl::components::on_policy_runner::Buffer<SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI step_i = 0; step_i < BUFFER_SPEC::STEPS_PER_ENV; step_i++){
            auto actions = view(device, buffer.actions, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations = view(device, buffer.observations, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);

            rl::components::on_policy_runner::prologue(device, buffer, runner, rng, step_i);
            evaluate(device, actor, observations, actions, policy_eval_buffers);
            rl::components::on_policy_runner::epilogue(device, buffer, runner, actions, actor.action_log_std.parameters, rng, step_i);
        }
        // final observation
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            auto& env = get(runner.environments, 0, env_i);
            auto& state = get(runner.states, 0, env_i);
            TI pos = BUFFER_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i;
            auto observation = view<DEVICE, typename decltype(buffer.all_observations)::SPEC, 1, SPEC::ENVIRONMENT::OBSERVATION_DIM>(device, buffer.all_observations, pos, 0);
            observe(device, env, state, observation);
        }
        runner.step += SPEC::N_ENVIRONMENTS * BUFFER_SPEC::STEPS_PER_ENV;
    }
}
#endif
