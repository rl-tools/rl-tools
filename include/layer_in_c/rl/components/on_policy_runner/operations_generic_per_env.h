#ifndef LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_PER_ENV_H
#define LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_PER_ENV_H

namespace layer_in_c::rl::components::on_policy_runner::per_env{
    template <typename DEVICE, typename OBSERVATIONS_SPEC, typename SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
    void prologue(DEVICE& device, Matrix<OBSERVATIONS_SPEC>& observations, rl::components::OnPolicyRunner<SPEC>& runner, RNG& rng, typename DEVICE::index_t env_i){
        static_assert(OBSERVATIONS_SPEC::ROWS == SPEC::N_ENVIRONMENTS);
        static_assert(OBSERVATIONS_SPEC::COLS == SPEC::ENVIRONMENT::OBSERVATION_DIM);
        auto& env = get(runner.environments, 0, env_i);
        auto& state = get(runner.states, 0, env_i);
        if(get(runner.truncated, 0, env_i)){
            add_scalar(device, device.logger, "episode/length", get(runner.episode_step, 0, env_i));
            add_scalar(device, device.logger, "episode/return", get(runner.episode_return, 0, env_i));
//                    std::cout << "epsidoe return: " << get(runner.episode_return, 0, env_i) << std::endl;
            set(runner.truncated, 0, env_i, false);
            set(runner.episode_step, 0, env_i, 0);
            set(runner.episode_return, 0, env_i, 0);
            sample_initial_state(device, env, state, rng);
        }
        auto observation = view(device, observations, matrix::ViewSpec<1, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
        observe(device, env, state, observation);
    }
    template <typename DEVICE, typename BUFFER_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
    void epilogue(DEVICE& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, RNG& rng, typename DEVICE::index_t pos, typename DEVICE::index_t env_i){
        using SPEC = typename BUFFER_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        T action_log_prob = 0;
        for(TI action_i = 0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++) {
            T action_mu = get(actions, env_i, action_i);
//                    std::stringstream topic;
//                    topic << "action/" << action_i;
//                    add_scalar(device, device.logger, topic.str(), action_mu);
            T action_std = math::exp(typename DEVICE::SPEC::MATH(), get(action_log_std, 0, action_i));
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

#endif
