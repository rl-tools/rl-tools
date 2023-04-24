#ifndef BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_PER_ENV_H
#define BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_PER_ENV_H
#include "off_policy_runner.h"
namespace backprop_tools::rl::components::off_policy_runner{
    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void prologue_per_env(DEVICE& device, rl::components::OffPolicyRunner<SPEC>* runner, RNG &rng, typename DEVICE::index_t env_i) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        // if the episode is done (step limit activated for STEP_LIMIT > 0) or if the step is the first step for this runner, reset the environment
        using RUNNER = rl::components::OffPolicyRunner<SPEC>;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        auto& env = runner->envs[env_i];
        auto& state = get(runner->states, 0, env_i);
        static_assert(!SPEC::COLLECT_EPISODE_STATS || SPEC::EPISODE_STATS_BUFFER_SIZE > 1);
        if (get(runner->truncated, 0, env_i)){
            if(SPEC::COLLECT_EPISODE_STATS){
                // todo: the first episode is always zero steps and zero return because the initialization is done by setting truncated to true
                auto& episode_stats = runner->episode_stats[env_i];
                TI next_episode_i = episode_stats.next_episode_i;
                if(next_episode_i > 0){
                    TI episode_i = next_episode_i - 1;
                    set(episode_stats.returns, episode_i, 0, get(runner->episode_return, 0, env_i));
                    set(episode_stats.steps  , episode_i, 0, get(runner->episode_step  , 0, env_i));
                    episode_i = (episode_i + 1) % SPEC::EPISODE_STATS_BUFFER_SIZE;
                    next_episode_i = episode_i + 1;
                }
                else{
                    next_episode_i = 1;
                }
                episode_stats.next_episode_i = next_episode_i;
            }
            sample_initial_state(device, env, state, rng);
            set(runner->episode_step, 0, env_i, 0);
            set(runner->episode_return, 0, env_i, 0);
        }
        auto observation = view<DEVICE, typename decltype(runner->buffers.observations)::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM>(device, runner->buffers.observations, env_i, 0);
        observe(device, env, state, observation);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void epilogue_per_env(DEVICE& device, rl::components::OffPolicyRunner<SPEC>* runner, RNG &rng, typename DEVICE::index_t env_i) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        using PARAMETERS = typename SPEC::PARAMETERS;
        auto observation = view<DEVICE, typename decltype(runner->buffers.observations)::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM>(device, runner->buffers.observations, env_i, 0);
        auto next_observation = view<DEVICE, typename decltype(runner->buffers.observations)::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM>(device, runner->buffers.next_observations, env_i, 0);
        auto action = view<DEVICE, typename decltype(runner->buffers.actions)::SPEC, 1, ENVIRONMENT::ACTION_DIM>(device, runner->buffers.actions, env_i, 0);
        auto& env = runner->envs[env_i];
        auto& state = get(runner->states, 0, env_i);
        typename ENVIRONMENT::State next_state;

        for (typename DEVICE::index_t i = 0; i < ENVIRONMENT::ACTION_DIM; i++) {
            T action_noisy = get(action, 0, i) + random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T) 0, PARAMETERS::EXPLORATION_NOISE, rng);
            set(action, 0, i, math::clamp<T>(typename DEVICE::SPEC::MATH(), action_noisy, -1, 1));
        }
        step(device, env, state, action, next_state);

        T reward_value = reward(device, env, state, action, next_state);

        observe(device, env, next_state, next_observation);

        bool terminated_flag = terminated(device, env, next_state, rng);
        increment(runner->episode_step, 0, env_i, 1);
        increment(runner->episode_return, 0, env_i, reward_value);
        auto episode_step_i = get(runner->episode_step, 0, env_i);
        bool truncated = terminated_flag || episode_step_i == SPEC::STEP_LIMIT;
        set(runner->truncated, 0, env_i, truncated);
        add(device, runner->replay_buffers[env_i], observation, action, reward_value, next_observation, terminated_flag, truncated);

        // state progression needs to come after the addition to the replay buffer because "observation" can point to the memory of runner_state.state (in the case of REQUIRES_OBSERVATION=false)
        state = next_state;
    }

}

#endif
