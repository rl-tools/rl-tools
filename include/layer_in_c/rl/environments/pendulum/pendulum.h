#ifndef LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_PENDULUM_H
#define LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_PENDULUM_H

#include <layer_in_c/math/operations_generic.h>

namespace layer_in_c::rl::environments::pendulum {
    template <typename T>
    struct DefaultParameters {
        constexpr static T g = 10;
        constexpr static T max_speed = 8;
        constexpr static T max_torque = 2;
        constexpr static T dt = 0.05;
        constexpr static T m = 1;
        constexpr static T l = 1;
        constexpr static T initial_state_min_angle = -math::PI<T>;
        constexpr static T initial_state_max_angle = math::PI<T>;
        constexpr static T initial_state_min_speed = -1;
        constexpr static T initial_state_max_speed = 1;
    };
    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T>>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    template <typename T>
    struct State{
        T theta;
        T theta_dot;
    };

}

namespace layer_in_c::rl::environments{
    template <typename T_SPEC>
    struct Pendulum{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using State = pendulum::State<typename SPEC::T>;
        using PARAMETERS = typename SPEC::PARAMETERS;
        static constexpr TI OBSERVATION_DIM = 3;
        static constexpr TI ACTION_DIM = 1;
    };
}







#endif
