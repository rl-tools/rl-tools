#ifndef LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_CPU_H
#define LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_CPU_H

#include "operations_generic.h"
namespace layer_in_c{
    template<typename SPEC, typename RNG>
    static void sample_initial_state(const rl::environments::Multirotor<devices::CPU, SPEC>& env, typename rl::environments::multirotor::State<typename SPEC::T>& state, RNG& rng){
//        state.theta     = std::uniform_real_distribution<typename SPEC::T>(SPEC::PARAMETERS::initial_state_min_angle, SPEC::PARAMETERS::initial_state_max_angle)(rng);
//        state.theta_dot = std::uniform_real_distribution<typename SPEC::T>(SPEC::PARAMETERS::initial_state_min_speed, SPEC::PARAMETERS::initial_state_max_speed)(rng);
    }
}
#endif
