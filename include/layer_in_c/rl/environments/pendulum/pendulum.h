#ifndef LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_PENDULUM_H
#define LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_PENDULUM_H

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/devices.h>

#include <layer_in_c/utils/generic/math.h>

namespace layer_in_c::rl::environments::pendulum {
    template <typename T>
    struct DefaultParameters {
        constexpr static T g = 10;
        constexpr static T max_speed = 8;
        constexpr static T max_torque = 2;
        constexpr static T dt = 0.05;
        constexpr static T m = 1;
        constexpr static T l = 1;
        constexpr static T initial_state_min_angle = -M_PI;
        constexpr static T initial_state_max_angle = M_PI;
        constexpr static T initial_state_min_speed = -1;
        constexpr static T initial_state_max_speed = 1;
    };
    template <typename T_T, typename T_PARAMETERS>
    struct Spec{
        typedef T_T T;
        typedef T_PARAMETERS PARAMETERS;
    };

    template <typename T>
    struct State{
        T theta;
        T theta_dot;
    };

}

namespace layer_in_c::rl::environments::Pendulum{
    // todo: if P0293R0 is ever accepted the template should be on the Pendulum container: Pendulum<SPEC>::CPU
    template <typename SPEC>
    struct Generic: devices::Generic{
        using State = pendulum::State<typename SPEC::T>;
        using PARAMETERS = typename SPEC::PARAMETERS;
        static constexpr size_t OBSERVATION_DIM = 3;
        static constexpr size_t ACTION_DIM = 1;
    };
    template <typename SPEC>
    struct CPU: Generic<SPEC>, devices::CPU{};

}







#endif
