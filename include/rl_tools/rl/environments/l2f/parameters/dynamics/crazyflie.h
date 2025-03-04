#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DYNAMICS_CRAZYFLIE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DYNAMICS_CRAZYFLIE_H
#include "../../multirotor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters::dynamics{
    template<typename SPEC, typename = rl_tools::utils::typing::enable_if_t<SPEC::N == 4>> // Crazyflie is a quadrotor
    constexpr typename ParametersBase<SPEC>::Dynamics crazy_flie = {
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
                    // {0.0213, -0.0112, 0.1201},
                    // {0.0213, -0.0112, 0.1201},
                    // {0.0213, -0.0112, 0.1201},
                    // {0.0213, -0.0112, 0.1201}
                    {0, 0, 0.1302},
                    {0, 0, 0.1302},
                    {0, 0, 0.1302},
                    {0, 0, 0.1302}
            },
            // torque constant
            {4.665e-3, 4.665e-3, 4.665e-3, 4.665e-3},
            // T, RPM time constant
            {0.072, 0.072, 0.072, 0.072},
            // mass vehicle
            0.027,
            // gravity
            {0, 0, -9.81},
            // J
            {
                    {
                            1.5e-5,
                            0.0,
                            0.0
                    },
                    {
                            0.0,
                            1.5e-5,
                            0.0
                    },
                    {
                            0.0,
                            0.0,
                            3.00e-5
                    }
            },
            // J_inv
            {
                    {
                            66666,
                            0.0,
                            0.0
                    },
                    {
                            0.0,
                            66666,
                            0.0
                    },
                    {
                            0.0,
                            0.0,
                            33333
                    }
            },
            // hovering throttle (julia): sqrt((mass * 9.81/4 - thrust_curve[1])/thrust_curve[3]),
//            "hovering_throttle": 14475.809152959684,
            0.6670265023020774, // "hovering_throttle_relative"
            // action limit
            {0, 1},
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif