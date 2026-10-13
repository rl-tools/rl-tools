#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DYNAMICS_CRAZYFLIE_BRUSHLESS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DYNAMICS_CRAZYFLIE_BRUSHLESS_H
#include "../../multirotor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters::dynamics{
    template<typename T, typename TI>
    constexpr Dynamics<T, TI, 4> crazyflie_brushless = {
            // Rotor positions
            {
                    {
                            0.03614938792820793,
                            -0.03548396793665085,
                            0
                    },
                    {
                            -0.0354058568601683,
                            -0.03517695689374581,
                            0
                    },
                    {
                            -0.03532426825872072,
                            0.03445540917862461,
                            0
                    },
                    {
                            0.0345807371906811,
                            0.03620551565177205,
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
                    {0, 0, 0.363092625},
                    {0, 0, 0.363092625},
                    {0, 0, 0.363092625},
                    {0, 0, 0.363092625},
            },
            // torque constant
            {0.00583125, 0.00583125, 0.00583125, 0.00583125},
            // T, RPM time constant
            {0.033, 0.033, 0.033, 0.033},
            {0.033, 0.033, 0.033, 0.033},
            // mass vehicle
            0.0423,
            // gravity
            {0, 0, -9.81},
            // J
            {
                    {
                            2.640865595841125e-05,
                            0.0,
                            0.0
                    },
                    {
                            0.0,
                            2.645856677608145e-05,
                            0.0
                    },
                    {
                            0.0,
                            0.0,
                            4.842637602479532e-05
                    }
            },
            // J_inv
            {
                    {
                            1/2.640865595841125e-05,
                            0.0,
                            0.0
                    },
                    {
                            0.0,
                            1/2.645856677608145e-05,
                            0.0
                    },
                    {
                            0.0,
                            0.0,
                            1/4.842637602479532e-05
                    }
            },
            // hovering throttle: 1/sqrt(3.5)
            0.5345224838248487694,
            // action limit
            {0, 1},
            10000, // max_rpm: visual convention shared with the web demo, the notebooks and the paper figures; replace with the measured full-scale rotor speed
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
