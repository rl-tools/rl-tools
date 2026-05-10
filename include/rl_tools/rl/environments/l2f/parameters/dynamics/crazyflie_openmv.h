#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DYNAMICS_CRAZYFLIE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DYNAMICS_CRAZYFLIE_H
#include "../../multirotor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters::dynamics{
    template<typename T, typename TI> // Crazyflie is a quadrotor
    constexpr Dynamics<T, TI, 4> crazyflie = {
            // Rotor positions
            {
                    {
                            0.028,
                            -0.028,
                            0
                    },
                    {
                            -0.028,
                            -0.028,
                            0
                    },
                    {
                            -0.028,
                            0.028,
                            0
                    },
                    {
                            0.028,
                            0.028,
                            0
                    },
            },
            // Rotor thrust directions
            {
                    {0, 0, 1},
                    {0, 0, 1},
                    {0, 0, 1},
                    {0, 0, 1},
            },
            // Rotor torque directions
            {
                    {0, 0, -1},
                    {0, 0, +1},
                    {0, 0, -1},
                    {0, 0, +1},
            },
            // thrust constants
            {
                    {0.01244825, 0.05290432, 0.15135338},
                    {0.01244825, 0.05290432, 0.15135338},
                    {0.01244825, 0.05290432, 0.15135338},
                    {0.01244825, 0.05290432, 0.15135338}
            },
            // torque constant
            {4.665e-3, 4.665e-3, 4.665e-3, 4.665e-3},
            // T, RPM time constant
            { // rising
                    0.061,
                    0.061,
                    0.061,
                    0.061
            },
            { // falling
                    0.061,
                    0.061,
                    0.061,
                    0.061
            },
            // mass vehicle
            0.0477,
            // gravity
            {0, 0, -9.81},
            // J
            {
                    {
                            1.543e-5,
                            0.0,
                            0.0
                    },
                    {
                            0.0,
                            1.946e-5,
                            0.0
                    },
                    {
                            0.0,
                            0.0,
                            3.1957e-5
                    }
            },
            // J_inv
            {
                    {
                            64808,
                            0.0,
                            0.0
                    },
                    {
                            0.0,
                            51387,
                            0.0
                    },
                    {
                            0.0,
                            0.0,
                            31292
                    }
            },
            // hovering throttle (julia): sqrt((mass * 9.81/4 - thrust_curve[1])/thrust_curve[3]),
//            "hovering_throttle": 14475.809152959684,
            0.8310686061945857, // "hovering_throttle_relative"
            // action limit
            {0, 1},
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif