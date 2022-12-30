#include <layer_in_c/utils/generic/math.h>
#include "off_policy_runner.h"

#include <layer_in_c/rl/components/replay_buffer/operations_cpu.h>

#include <random>
#include <iostream>

namespace layer_in_c{
    template<typename SPEC, typename POLICY, typename RNG>
    void step(rl::components::OffPolicyRunner<devices::CPU, SPEC> &runner, POLICY &policy, RNG &rng) {
        using T = typename SPEC::T;
        // if the episode is done (step limit activated for STEP_LIMIT > 0) or if the step is the first step for this runner, reset the environment
        typedef typename SPEC::ENVIRONMENT ENVIRONMENT;
        typedef typename SPEC::PARAMETERS PARAMETERS;
        if ((SPEC::STEP_LIMIT > 0 && runner.episode_step == SPEC::STEP_LIMIT) ||
            (runner.replay_buffer.position == 0 && !runner.replay_buffer.full)) {
            // first step
            sample_initial_state(ENVIRONMENT(), runner.state, rng);
            runner.episode_step = 0;
            runner.episode_return = 0;
        }
        // todo: increase efficiency by removing the double observation of each state
        T observation[ENVIRONMENT::OBSERVATION_DIM];
        observe(ENVIRONMENT(), runner.state, observation);
        typename ENVIRONMENT::State next_state;
        T action[ENVIRONMENT::ACTION_DIM];
        evaluate(policy, observation, action);
        std::normal_distribution<T> exploration_noise_distribution(0, PARAMETERS::EXPLORATION_NOISE);
        for (int i = 0; i < ENVIRONMENT::ACTION_DIM; i++) {
            action[i] += exploration_noise_distribution(rng);
            action[i] = lic::utils::math::clamp<T>(action[i], -1, 1);
        }
        step(ENVIRONMENT(), runner.state, action, next_state);
        T reward_value = reward(ENVIRONMENT(), runner.state, action, next_state);
        runner.state = next_state;
        T next_observation[ENVIRONMENT::OBSERVATION_DIM];
        observe(ENVIRONMENT(), next_state, next_observation);
        bool terminated_flag = terminated(ENVIRONMENT(), next_state);
        runner.episode_step += 1;
        runner.episode_return += reward_value;
        bool truncated = runner.episode_step == SPEC::STEP_LIMIT;
        if (truncated || terminated_flag) {
            std::cout << "Episode return: " << runner.episode_return << std::endl;
        }
        // todo: add truncation / termination handling (stemming from the environment)
        add(runner.replay_buffer, observation, action, reward_value, next_observation, terminated_flag, truncated);
    }
}
