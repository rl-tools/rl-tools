#ifndef LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include "on_policy_runner.h"

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
    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, typename SPEC::ENVIRONMENT environments[SPEC::N_ENVIRONMENTS]){
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
    template <typename DEVICE, typename BUFFER_SPEC, typename PPO, typename RNG> // todo: make this not PPO but general policy with output distribution
    void collect(DEVICE& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, PPO& ppo, typename PPO::SPEC::ACTOR_NETWORK_TYPE::template Buffers<BUFFER_SPEC::SPEC::N_ENVIRONMENTS>& policy_eval_buffers, RNG& rng){
#ifdef LAYER_IN_C_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        using SPEC = typename BUFFER_SPEC::SPEC;
        using BUFFER = rl::components::on_policy_runner::Buffer<SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI step_i = 0; step_i < BUFFER_SPEC::STEPS_PER_ENV; step_i++){
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                auto& env = get(runner.environments, 0, env_i);
                auto& state = get(runner.states, 0, env_i);
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                if(get(runner.truncated, 0, env_i)){
                    std::cout << "episode return: " << get(runner.episode_return, 0, env_i) << " in " << get(runner.episode_step, 0, env_i) << " steps" << std::endl;
                    set(runner.truncated, 0, env_i, false);
                    set(runner.episode_step, 0, env_i, 0);
                    set(runner.episode_return, 0, env_i, 0);
                    sample_initial_state(device, env, state, rng);
//                    initial_state(device, env, state);
                }
                auto observation = view<DEVICE, typename decltype(buffer.observations)::SPEC, 1, SPEC::ENVIRONMENT::OBSERVATION_DIM>(device, buffer.observations, pos, 0);
                observe(device, env, state, observation);

            }
            auto actions = view(device, buffer.actions, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations = view(device, buffer.observations, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);

            evaluate(device, ppo.actor, observations, actions);

            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;

                T action_log_prob = 0;
                for(TI action_i = 0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++) {
                    T action_mu = get(actions, env_i, action_i);
                    T action_std = math::exp(typename DEVICE::SPEC::MATH(), get(ppo.actor_log_std, 0, action_i));
                    T action_noisy = random::normal_distribution(typename DEVICE::SPEC::RANDOM(), action_mu, action_std, rng);
                    T action_by_action_std = (action_noisy-action_mu) / action_std;
                    action_log_prob += -0.5 * action_by_action_std * action_by_action_std - math::log(typename DEVICE::SPEC::MATH(), action_std) - 0.5 * math::log(typename DEVICE::SPEC::MATH(), 2 * math::PI<T>);
                    set(actions, env_i, action_i, action_noisy);
                }
                set(buffer.action_log_probs, pos, 0, action_log_prob);
                auto& env = get(runner.environments, 0, env_i);
                auto& state = get(runner.states, 0, env_i);
                typename SPEC::ENVIRONMENT::State next_state;
                auto action = row(device, actions, env_i);
                step(device, env, state, action, next_state);
                bool terminated_flag = terminated(device, env, next_state, rng);
                set(buffer.terminated, pos, 0, terminated_flag);
                T reward_value = reward(device, env, state, action, next_state);
                increment(runner.episode_return, 0, env_i, reward_value);
                set(buffer.rewards, pos, 0, reward_value);
                increment(runner.episode_step, 0, env_i, 1);
                bool truncated = terminated_flag || (SPEC::STEP_LIMIT > 0 && get(runner.episode_step, 0, env_i) >= SPEC::STEP_LIMIT);
                set(buffer.truncated, pos, 0, truncated);
                set(runner.truncated, 0, env_i, truncated);
                state = next_state;
            }

        }
        // final observation
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            auto& env = get(runner.environments, 0, env_i);
            auto& state = get(runner.states, 0, env_i);
            TI pos = BUFFER_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i;
            auto observation = view<DEVICE, typename decltype(buffer.all_observations)::SPEC, 1, SPEC::ENVIRONMENT::OBSERVATION_DIM>(device, buffer.all_observations, pos, 0);
            observe(device, env, state, observation);
        }
    }
}
#endif
