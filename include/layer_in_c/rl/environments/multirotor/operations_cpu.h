#ifndef LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_CPU_H
#define LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_CPU_H

#include "operations_generic.h"

#include <random>
namespace layer_in_c{
    template<typename DEV_SPEC, typename SPEC, typename RNG>
    static void sample_initial_state(devices::CPU<DEV_SPEC>& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::State& state, RNG& rng){
        using T = typename SPEC::T;
        using index_t = typename devices::CPU<DEV_SPEC>::index_t;
        for(index_t i = 0; i < 3; i++){
            state.state[i] = std::uniform_real_distribution<T>(-env.parameters.mdp.init.max_position, env.parameters.mdp.init.max_position)(rng);
        }
        // https://web.archive.org/web/20181126051029/http://planning.cs.uiuc.edu/node198.html
        if(env.parameters.mdp.init.max_angle > 0 && std::uniform_real_distribution<T>(0, 1)(rng) > env.parameters.mdp.init.guidance){
            T u[3];
            for(typename devices::CPU<DEV_SPEC>::index_t i = 0; i < 3; i++){
                u[i] = std::uniform_real_distribution<T>(0, 1)(rng);
            }
            state.state[3+0] = math::sqrt(typename DEV_SPEC::MATH(), 1-u[0]) * math::sin(typename DEV_SPEC::MATH(), 2*M_PI*u[1]);
            state.state[3+1] = math::sqrt(typename DEV_SPEC::MATH(), 1-u[0]) * math::cos(typename DEV_SPEC::MATH(), 2*M_PI*u[1]);
            state.state[3+2] = math::sqrt(typename DEV_SPEC::MATH(), u[0]) * math::sin(typename DEV_SPEC::MATH(), 2*M_PI*u[2]);
            state.state[3+3] = math::sqrt(typename DEV_SPEC::MATH(), u[0]) * math::cos(typename DEV_SPEC::MATH(), 2*M_PI*u[2]);
        }
        else{
            state.state[3+0] = 1;
            state.state[3+1] = 0;
            state.state[3+2] = 0;
            state.state[3+3] = 0;
        }
        for(index_t i = 0; i < 3; i++){
            state.state[7+i] = std::uniform_real_distribution<T>(-env.parameters.mdp.init.max_linear_velocity, env.parameters.mdp.init.max_linear_velocity)(rng);
        }
        for(index_t i = 0; i < 3; i++){
            state.state[10+i] = std::uniform_real_distribution<T>(-env.parameters.mdp.init.max_angular_velocity, env.parameters.mdp.init.max_angular_velocity)(rng);
        }
    }
}
#endif
