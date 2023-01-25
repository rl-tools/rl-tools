#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include <layer_in_c/math/operations_generic.h>
#include "off_policy_runner.h"

#include <layer_in_c/rl/components/replay_buffer/operations_generic.h>

namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner) {
        malloc(device, runner.replay_buffer);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner) {
        free(device, runner.replay_buffer);
    }
    template<typename DEVICE, typename SPEC, typename POLICY, typename RNG>
    void step(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner, POLICY &policy, RNG &rng) {
        static_assert(POLICY::INPUT_DIM == SPEC::ENVIRONMENT::OBSERVATION_DIM, "The policy's input dimension must match the environment's observation dimension.");
        static_assert(POLICY::OUTPUT_DIM == SPEC::ENVIRONMENT::ACTION_DIM, "The policy's output dimension must match the environment's action dimension.");
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        // if the episode is done (step limit activated for STEP_LIMIT > 0) or if the step is the first step for this runner, reset the environment
        typedef typename SPEC::ENVIRONMENT ENVIRONMENT;
        typedef typename SPEC::PARAMETERS PARAMETERS;
        if (runner.truncated) {
            // first step
            sample_initial_state(device, runner.env, runner.state, rng);
            runner.episode_step = 0;
            runner.episode_return = 0;
        }
        // todo: increase efficiency by removing the double observation of each state

        T observation_mem[ENVIRONMENT::OBSERVATION_DIM];
        T* observation;
        if constexpr(ENVIRONMENT::REQUIRES_OBSERVATION){
            observation = observation_mem;
            observe(device, runner.env, runner.state, observation);
        }
        else{
            static_assert(sizeof(runner.state.state)/sizeof(runner.state.state[0]) == SPEC::ENVIRONMENT::OBSERVATION_DIM, "The environments state dimension must match the environment's observation dimension.");
            observation = runner.state.state;
        }

        typename ENVIRONMENT::State next_state;
        T action[ENVIRONMENT::ACTION_DIM];
        Matrix<MatrixSpecification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action_m = {action};
        Matrix<MatrixSpecification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_m = {observation};
        evaluate(device, policy, observation_m, action_m);
        for(typename DEVICE::index_t i = 0; i < ENVIRONMENT::ACTION_DIM; i++) {
            action[i] += random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, PARAMETERS::EXPLORATION_NOISE, rng);
            action[i] = lic::math::clamp<T>(action[i], -1, 1);
        }
        step(device, runner.env, runner.state, action, next_state);
        T reward_value = reward(device, runner.env, runner.state, action, next_state);
//        if constexpr(DEVICE::DEBUG::PRINT_REWARD){
//            std::cout << "reward: " << reward_value << std::endl;
//        }
        runner.state = next_state;

        T next_observation_mem[ENVIRONMENT::OBSERVATION_DIM];
        T* next_observation;
        if constexpr(ENVIRONMENT::REQUIRES_OBSERVATION){
            next_observation = next_observation_mem;
            observe(device, runner.env, next_state, next_observation);
        }
        else{
            next_observation = next_state.state;
        }

        bool terminated_flag = terminated(device, runner.env, next_state);
        runner.episode_step += 1;
        runner.episode_return += reward_value;
        runner.truncated = terminated_flag || runner.episode_step == SPEC::STEP_LIMIT;
        if (runner.truncated) {
//            logging::text(device.logger, "Episode return: ", runner.episode_return);
//            logging::text(device.logger, "Episode steps: ", runner.episode_step);
        }
        // todo: add truncation / termination handling (stemming from the environment)
        add(device, runner.replay_buffer, observation, action, reward_value, next_observation, terminated_flag, runner.truncated);
    }
}

#endif
