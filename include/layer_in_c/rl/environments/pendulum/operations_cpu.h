#ifndef LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_OPERATIONS_CPU_H
#define LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_OPERATIONS_CPU_H

#include <random>

#include "pendulum.h"
#include "operations_generic.h"

namespace layer_in_c{
    template<typename SPEC>
    static typename SPEC::T step(const rl::environments::Pendulum<devices::CPU, SPEC>& env, const rl::environments::pendulum::State<typename SPEC::T>& state, const typename SPEC::T action[1], rl::environments::pendulum::State<typename SPEC::T>& next_state) {
        return step<SPEC>((rl::environments::Pendulum<devices::Generic, SPEC>&)env, state, action, next_state);
    }
    template<typename SPEC>
    static typename SPEC::T reward(const rl::environments::Pendulum<devices::CPU, SPEC>& env, const rl::environments::pendulum::State<typename SPEC::T>& state, const typename SPEC::T action[1], const rl::environments::pendulum::State<typename SPEC::T>& next_state){
        return reward((rl::environments::Pendulum<devices::Generic, SPEC>&)env, state, action, next_state);
    }

    template<typename SPEC>
    static void observe(const rl::environments::Pendulum<devices::CPU, SPEC>& env, const rl::environments::pendulum::State<typename SPEC::T>& state, typename SPEC::T observation[3]){
        observe((rl::environments::Pendulum<devices::Generic, SPEC>&)env, state, observation);
    }
    template<typename SPEC>
    static bool terminated(const rl::environments::Pendulum<devices::CPU, SPEC>& env, const typename rl::environments::pendulum::State<typename SPEC::T> state){
        return terminated((rl::environments::Pendulum<devices::Generic, SPEC>&)env, state);
    }
}
// Specializations
namespace layer_in_c{
    template<typename SPEC, typename RNG>
    static void sample_initial_state(const rl::environments::Pendulum<devices::CPU, SPEC>& env, rl::environments::pendulum::State<typename SPEC::T>& state, RNG& rng){
        state.theta     = std::uniform_real_distribution<typename SPEC::T>(SPEC::PARAMETERS::initial_state_min_angle, SPEC::PARAMETERS::initial_state_max_angle)(rng);
        state.theta_dot = std::uniform_real_distribution<typename SPEC::T>(SPEC::PARAMETERS::initial_state_min_speed, SPEC::PARAMETERS::initial_state_max_speed)(rng);
    }
}



#endif
