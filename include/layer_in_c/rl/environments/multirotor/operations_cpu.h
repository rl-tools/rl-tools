#ifndef LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_CPU_H
#define LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_CPU_H

#include "operations_generic.h"

#include <random>
namespace layer_in_c{
    template<typename SPEC, typename RNG>
    static void sample_initial_state(const rl::environments::Multirotor<devices::CPU, SPEC>& env, typename rl::environments::multirotor::State<typename SPEC::T>& state, RNG& rng){
        using T = typename SPEC::T;
        for(size_t i = 0; i < 3; i++){
            state.state[i] = std::uniform_real_distribution<T>(-env.parameters.init.max_position, env.parameters.init.max_position)(rng);
        }
        // https://web.archive.org/web/20181126051029/http://planning.cs.uiuc.edu/node198.html
        T u[3];
        for(size_t i = 0; i < 3; i++){
            u[i] = std::uniform_real_distribution<T>(0, 1)(rng);
        }
        state.state[3+0] = std::sqrt(1-u[0]) * std::sin(2*M_PI*u[1]);
        state.state[3+1] = std::sqrt(1-u[0]) * std::cos(2*M_PI*u[1]);
        state.state[3+2] = std::sqrt(u[0]) * std::sin(2*M_PI*u[2]);
        state.state[3+3] = std::sqrt(u[0]) * std::cos(2*M_PI*u[2]);
        for(size_t i = 0; i < 3; i++){
            state.state[7+i] = std::uniform_real_distribution<T>(-env.parameters.init.max_linear_velocity, env.parameters.init.max_linear_velocity)(rng);
        }
        for(size_t i = 0; i < 3; i++){
            state.state[10+i] = std::uniform_real_distribution<T>(-env.parameters.init.max_angular_velocity, env.parameters.init.max_angular_velocity)(rng);
        }
    }
}
#endif
