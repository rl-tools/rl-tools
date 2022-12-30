#ifndef LAYER_IN_C_RL_UTILS_EVALUATION_H
#define LAYER_IN_C_RL_UTILS_EVALUATION_H
/*
 * This file relies on the environments methods hence it should be included after the operations of the environments that it will be used with
 */

#include "layer_in_c/rl/environments/environments.h"
#include <iostream>

namespace layer_in_c {
    template<typename ENVIRONMENT, typename POLICY, int STEP_LIMIT>
    typename POLICY::T evaluate(const rl::environments::Environment env, POLICY &policy, const typename ENVIRONMENT::State initial_state) {
        typedef typename POLICY::T T;
        typename ENVIRONMENT::State state;
        state = initial_state;
        T episode_return = 0;
        for (int i = 0; i < STEP_LIMIT; i++) {
            T observation[ENVIRONMENT::OBSERVATION_DIM];
            observe(ENVIRONMENT(), state, observation);
            T action[ENVIRONMENT::ACTION_DIM];
            evaluate(policy, observation, action);
            T action_clipped[ENVIRONMENT::ACTION_DIM];
            for(int action_i=0; action_i<ENVIRONMENT::ACTION_DIM; action_i++){
                action_clipped[action_i] = std::clamp<T>(action[action_i], -1, 1);
            }
            typename ENVIRONMENT::State next_state;
            step(ENVIRONMENT(), state, action_clipped, next_state);
            T r = reward(ENVIRONMENT(), state, action_clipped, next_state);
            state = next_state;
            episode_return += r;
            bool terminated_flag = terminated(ENVIRONMENT(), state);
            if (terminated_flag) {
                break;
            }
        }
        return episode_return;
    }
    template<typename ENVIRONMENT, typename POLICY, typename RNG, int STEP_LIMIT, bool DETERMINISTIC>
    typename POLICY::T evaluate(const rl::environments::Environment env, POLICY &policy, uint32_t N, RNG &rng) {
        typedef typename POLICY::T T;
        static_assert(ENVIRONMENT::OBSERVATION_DIM == POLICY::INPUT_DIM, "Observation and policy input dimensions must match");
        static_assert(ENVIRONMENT::ACTION_DIM == POLICY::OUTPUT_DIM, "Action and policy output dimensions must match");
        T episode_returns[N];
        for (int i = 0; i < N; i++) {
            typename ENVIRONMENT::State initial_state;
            if(DETERMINISTIC) {
                initial_state.theta = -M_PI;
                initial_state.theta_dot = 0;
            }
            else{
                sample_initial_state(ENVIRONMENT(), initial_state, rng);
            }
            episode_returns[i] = evaluate<ENVIRONMENT, POLICY, STEP_LIMIT>(env, policy, initial_state);
        }
        T mean = 0;
        for (int i = 0; i < N; i++) {
            mean += episode_returns[i];
        }
        mean /= N;
        T variance = 0;
        for (int i = 0; i < N; i++) {
            variance += (episode_returns[i] - mean) * (episode_returns[i] - mean);
        }
        variance /= N;
        T standard_deviation = std::sqrt(variance);
        std::cout << "Mean: " << mean << ", standard deviation: " << standard_deviation << std::endl;
        return mean;
    }


}

#endif
