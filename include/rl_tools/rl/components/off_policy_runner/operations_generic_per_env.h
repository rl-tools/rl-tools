#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_PER_ENV_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_PER_ENV_H
#include "off_policy_runner.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components::off_policy_runner{
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void prologue_per_env(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng, typename DEVICE::index_t env_i) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        // if the episode is done (step limit activated for STEP_LIMIT > 0) or if the step is the first step for this runner, reset the environment
        using RUNNER = rl::components::OffPolicyRunner<SPEC>;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        auto& env = runner.envs[env_i];
        auto& state = get(runner.states, 0, env_i);
        static_assert(!SPEC::COLLECT_EPISODE_STATS || SPEC::EPISODE_STATS_BUFFER_SIZE > 1);
        if (get(runner.truncated, 0, env_i)){
            if constexpr(SPEC::COLLECT_EPISODE_STATS){
                // todo: the first episode is always zero steps and zero return because the initialization is done by setting truncated to true
                auto& episode_stats = runner.episode_stats[env_i];
                TI next_episode_i = episode_stats.next_episode_i;
                if(next_episode_i > 0){
                    TI episode_i = next_episode_i - 1;
                    set(episode_stats.returns, episode_i, 0, get(runner.episode_return, 0, env_i));
                    set(episode_stats.steps  , episode_i, 0, get(runner.episode_step  , 0, env_i));
                    episode_i = (episode_i + 1) % SPEC::EPISODE_STATS_BUFFER_SIZE;
                    next_episode_i = episode_i + 1;
                }
                else{
                    next_episode_i = 1;
                }
                episode_stats.next_episode_i = next_episode_i;
            }
            sample_initial_state(device, env, state, rng);
            set(runner.episode_step, 0, env_i, 0);
            set(runner.episode_return, 0, env_i, 0);
        }
        auto observation            = view<DEVICE, typename decltype(runner.buffers.observations           )::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM           >(device, runner.buffers.observations           , env_i, 0);
        auto observation_privileged = view<DEVICE, typename decltype(runner.buffers.observations_privileged)::SPEC, 1, SPEC::OBSERVATION_DIM_PRIVILEGED>(device, runner.buffers.observations_privileged, env_i, 0);
        observe(device, env, state, observation, rng);
        if constexpr(SPEC::ASYMMETRIC_OBSERVATIONS){
            observe_privileged(device, env, state, observation_privileged, rng);
        }
    }
    template<typename DEVICE, typename SPEC, typename POLICY, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT auto process_and_get_action(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, const POLICY& policy, RNG &rng, typename DEVICE::index_t env_i) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        static constexpr bool STOCHASTIC_POLICY = POLICY::OUTPUT_DIM == ENVIRONMENT::ACTION_DIM * 2;
        if constexpr(STOCHASTIC_POLICY){
            for (TI i = 0; i < ENVIRONMENT::ACTION_DIM; i++){
                T log_std = get(runner.buffers.actions, env_i, ENVIRONMENT::ACTION_DIM+i);
                T log_std_clip = math::clamp<T>(device.math, log_std, -20, 2); // todo: absorb this into the policy
                T std = math::exp(typename DEVICE::SPEC::MATH{}, log_std_clip);
                T mu = get(runner.buffers.actions, env_i, i);
                T action_noisy = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), mu, std, rng);
                static_assert(SPEC::ACTION_CLAMPING_TANH);
                if constexpr(SPEC::ACTION_CLAMPING_TANH){
                    set(runner.buffers.actions, env_i, i, math::tanh<T>(device.math, action_noisy));
                }
                else{
                    set(runner.buffers.actions, env_i, i, math::clamp<T>(device.math, action_noisy, -1, 1));
                }
            }
            return view(device, runner.buffers.actions, matrix::ViewSpec<1, SPEC::ENVIRONMENT::ACTION_DIM>{}, env_i, 0);
        }
        else{
            for (TI i = 0; i < ENVIRONMENT::ACTION_DIM; i++){
                T action_noisy = get(runner.buffers.actions, env_i, i) + random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T) 0, runner.parameters.exploration_noise, rng);
                set(runner.buffers.actions, env_i, i, math::clamp<T>(device.math, action_noisy, -1, 1));
            }
            return view(device, runner.buffers.actions, matrix::ViewSpec<1, SPEC::ENVIRONMENT::ACTION_DIM>{}, env_i, 0);
//            return row(device, runner.buffers.actions, env_i);
        }

    }
    template<typename DEVICE, typename SPEC, typename POLICY, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void epilogue_per_env(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, const POLICY& policy, RNG &rng, typename DEVICE::index_t env_i) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        auto observation                 = view<DEVICE, typename decltype(runner.buffers.observations           )::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM           >(device, runner.buffers.observations           , env_i, 0);
        auto observation_privileged      = view<DEVICE, typename decltype(runner.buffers.observations_privileged)::SPEC, 1, SPEC::OBSERVATION_DIM_PRIVILEGED>(device, runner.buffers.observations_privileged, env_i, 0);
        auto next_observation            = view<DEVICE, typename decltype(runner.buffers.observations           )::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM           >(device, runner.buffers.next_observations           , env_i, 0);
        auto next_observation_privileged = view<DEVICE, typename decltype(runner.buffers.observations_privileged)::SPEC, 1, SPEC::OBSERVATION_DIM_PRIVILEGED>(device, runner.buffers.next_observations_privileged, env_i, 0);
//        auto action_raw = view<DEVICE, typename decltype(runner.buffers.actions)::SPEC, 1, ENVIRONMENT::ACTION_DIM>(device, runner.buffers.actions, env_i, 0);
        auto& env = runner.envs[env_i];
        auto& state = get(runner.states, 0, env_i);
        typename ENVIRONMENT::State next_state;

        auto action = process_and_get_action(device, runner, policy, rng, env_i);

        step(device, env, state, action, next_state, rng);

        T reward_value = reward(device, env, state, action, next_state, rng);

        observe(device, env, next_state, next_observation, rng);
        if constexpr(SPEC::ASYMMETRIC_OBSERVATIONS) {
            observe_privileged(device, env, next_state, next_observation_privileged, rng);
        }

        bool terminated_flag = terminated(device, env, next_state, rng);
        increment(runner.episode_step, 0, env_i, 1);
        increment(runner.episode_return, 0, env_i, reward_value);
        auto episode_step_i = get(runner.episode_step, 0, env_i);
        bool truncated = terminated_flag || episode_step_i == SPEC::STEP_LIMIT;
        set(runner.truncated, 0, env_i, truncated);
        add(device, runner.replay_buffers[env_i], state, observation, observation_privileged, action, reward_value, next_state, next_observation, next_observation_privileged, terminated_flag, truncated);

        // state progression needs to come after the addition to the replay buffer because "observation" can point to the memory of runner_state.state (in the case of REQUIRES_OBSERVATION=false)
        state = next_state;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
