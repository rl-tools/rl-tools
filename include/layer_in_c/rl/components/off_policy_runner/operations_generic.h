#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include <layer_in_c/math/operations_generic.h>
#include "off_policy_runner.h"

#include <layer_in_c/rl/components/replay_buffer/operations_generic.h>

namespace layer_in_c{
    template<typename DEVICE, typename SPEC, typename POLICY, typename RNG>
    void step(rl::components::OffPolicyRunner<DEVICE, SPEC> &runner, POLICY &policy, RNG &rng) {
        static_assert(POLICY::INPUT_DIM == SPEC::ENVIRONMENT::OBSERVATION_DIM, "The policy's input dimension must match the environment's observation dimension.");
        static_assert(POLICY::OUTPUT_DIM == SPEC::ENVIRONMENT::ACTION_DIM, "The policy's output dimension must match the environment's action dimension.");
        using T = typename SPEC::T;
        // if the episode is done (step limit activated for STEP_LIMIT > 0) or if the step is the first step for this runner, reset the environment
        typedef typename SPEC::ENVIRONMENT ENVIRONMENT;
        typedef typename SPEC::PARAMETERS PARAMETERS;
        if ((SPEC::STEP_LIMIT > 0 && runner.episode_step == SPEC::STEP_LIMIT) ||
            (runner.replay_buffer.position == 0 && !runner.replay_buffer.full)) {
            // first step
            sample_initial_state(runner.env, runner.state, rng);
            runner.episode_step = 0;
            runner.episode_return = 0;
        }
        // todo: increase efficiency by removing the double observation of each state

        T observation_mem[ENVIRONMENT::OBSERVATION_DIM];
        T* observation;
        if constexpr(ENVIRONMENT::REQUIRES_OBSERVATION){
            observation = observation_mem;
            observe(runner.env, runner.state, observation);
        }
        else{
            static_assert(sizeof(runner.state.state)/sizeof(runner.state.state[0]) == SPEC::ENVIRONMENT::OBSERVATION_DIM, "The environments state dimension must match the environment's observation dimension.");
            observation = runner.state.state;
        }

        typename ENVIRONMENT::State next_state;
        T action[ENVIRONMENT::ACTION_DIM];
        evaluate(policy, observation, action);
        for(typename DEVICE::index_t i = 0; i < ENVIRONMENT::ACTION_DIM; i++) {
            action[i] += utils::random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, PARAMETERS::EXPLORATION_NOISE, rng);
            action[i] = lic::math::clamp<T>(action[i], -1, 1);
        }
        step(runner.env, runner.state, action, next_state);
        T reward_value = reward(runner.env, runner.state, action, next_state);
        runner.state = next_state;

        T next_observation_mem[ENVIRONMENT::OBSERVATION_DIM];
        T* next_observation;
        if constexpr(ENVIRONMENT::REQUIRES_OBSERVATION){
            next_observation = next_observation_mem;
            observe(runner.env, next_state, next_observation);
        }
        else{
            next_observation = next_state.state;
        }

        bool terminated_flag = terminated(runner.env, next_state);
        runner.episode_step += 1;
        runner.episode_return += reward_value;
        bool truncated = runner.episode_step == SPEC::STEP_LIMIT;
        if (truncated || terminated_flag) {
            logging::text(runner.device.logger, "Episode return: ", runner.episode_return);
        }
        // todo: add truncation / termination handling (stemming from the environment)
        add(runner.replay_buffer, observation, action, reward_value, next_observation, terminated_flag, truncated);
    }
}

#endif
