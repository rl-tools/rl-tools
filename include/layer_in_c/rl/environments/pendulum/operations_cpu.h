#ifndef LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_OPERATIONS_CPU_H
#define LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_OPERATIONS_CPU_H

#include <random>

#include "pendulum.h"
#include "operations_generic.h"

namespace layer_in_c{
    template<typename SPEC, typename RNG>
    static void sample_initial_state(const rl::environments::Pendulum::CPU<SPEC>& env, typename rl::environments::Pendulum::CPU<SPEC>::State& state, RNG& rng){
        state.theta     = std::uniform_real_distribution<typename SPEC::T>(SPEC::PARAMETERS::initial_state_min_angle, SPEC::PARAMETERS::initial_state_max_angle)(rng);
        state.theta_dot = std::uniform_real_distribution<typename SPEC::T>(SPEC::PARAMETERS::initial_state_min_speed, SPEC::PARAMETERS::initial_state_max_speed)(rng);
    }
}



#endif
