#ifndef LAYER_IN_C_RL_ALGORITHMS_TD3_OFF_POLICY_RUNNER
#define LAYER_IN_C_RL_ALGORITHMS_TD3_OFF_POLICY_RUNNER
#include "replay_buffer.h"

namespace lic = layer_in_c;

namespace layer_in_c::rl::algorithms::td3 {
    template<typename T, int T_CAPACITY, int T_STEP_LIMIT>
    struct DefaultOffPolicyRunnerParameters {
        static constexpr uint32_t CAPACITY = T_CAPACITY;
        static constexpr uint32_t STEP_LIMIT = T_STEP_LIMIT;
        static constexpr T EXPLORATION_NOISE = 0.1;
    };

    template<typename T, typename ENVIRONMENT, typename PARAMETERS>
    struct OffPolicyRunner {
        ReplayBuffer<T, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, PARAMETERS::CAPACITY> replay_buffer;
        T state[ENVIRONMENT::STATE_DIM];
        uint32_t episode_step = 0;
        T episode_return = 0;
    };


    template<typename T, typename ENVIRONMENT, typename POLICY, typename PARAMETERS, typename RNG>
    void step(OffPolicyRunner<T, ENVIRONMENT, PARAMETERS> &runner, POLICY &policy, RNG &rng) {
        // if the episode is done (step limit activated for STEP_LIMIT > 0) or if the step is the first step for this runner, reset the environment
        if ((PARAMETERS::STEP_LIMIT > 0 && runner.episode_step == PARAMETERS::STEP_LIMIT) ||
            (runner.replay_buffer.position == 0 && !runner.replay_buffer.full)) {
            // first step
            lic::sample_initial_state(ENVIRONMENT(), runner.state, rng);
            runner.episode_step = 0;
            runner.episode_return = 0;
        }
        // todo: increase efficiency by removing the double observation of each state
        T observation[ENVIRONMENT::OBSERVATION_DIM];
        observe(ENVIRONMENT(), runner.state, observation);
        T next_state[ENVIRONMENT::STATE_DIM];
        T action[ENVIRONMENT::ACTION_DIM];
        lic::evaluate(policy, observation, action);
        std::normal_distribution<T> exploration_noise_distribution(0, PARAMETERS::EXPLORATION_NOISE);
        for (int i = 0; i < ENVIRONMENT::ACTION_DIM; i++) {
            action[i] += exploration_noise_distribution(rng);
            action[i] = std::clamp<T>(action[i], -1, 1);
        }
        T reward = lic::step(ENVIRONMENT(), runner.state, action, next_state);
        memcpy(runner.state, next_state, sizeof(T) * ENVIRONMENT::STATE_DIM);
        T next_observation[ENVIRONMENT::OBSERVATION_DIM];
        lic::observe(ENVIRONMENT(), next_state, next_observation);
        bool terminated = false;
        runner.episode_step += 1;
        runner.episode_return += reward;
        bool truncated = runner.episode_step == PARAMETERS::STEP_LIMIT;
        if (truncated || terminated) {
            std::cout << "Episode return: " << runner.episode_return << std::endl;
        }
        // todo: add truncation / termination handling (stemming from the environment)
        add(runner.replay_buffer, observation, action, reward, next_observation, terminated, truncated);
    }
    template<typename ENVIRONMENT, typename POLICY, typename RNG, int STEP_LIMIT>
    typename POLICY::T evaluate(POLICY &policy, RNG &rng) {
        typedef typename POLICY::T T;
        T state[ENVIRONMENT::STATE_DIM];
        lic::sample_initial_state(ENVIRONMENT(), state, rng);
        T episode_return = 0;
        for (int i = 0; i < STEP_LIMIT; i++) {
            T observation[ENVIRONMENT::OBSERVATION_DIM];
            lic::observe(ENVIRONMENT(), state, observation);
            T action[ENVIRONMENT::ACTION_DIM];
            lic::evaluate(policy, observation, action);
            T next_state[ENVIRONMENT::STATE_DIM];
            T reward = lic::step(ENVIRONMENT(), state, action, next_state);
            memcpy(state, next_state, sizeof(T) * ENVIRONMENT::STATE_DIM);
            episode_return += reward;
            bool terminated = lic::terminated(ENVIRONMENT(), state);
            if (terminated) {
                break;
            }
        }
    }
}
#endif