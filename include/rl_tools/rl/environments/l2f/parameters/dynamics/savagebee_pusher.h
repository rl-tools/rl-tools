#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DYNAMICS_SAVAGEBEE_PUSHER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DYNAMICS_SAVAGEBEE_PUSHER_H
#include "../../multirotor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters::dynamics{
    template<typename T, typename TI>
    constexpr Dynamics<T, TI, 4> savagebee_pusher = {
            // Rotor positions
            {
                    {
                            0.03421002761730008,
                            -0.04382408245097331,
                            0
                    },
                    {
                            -0.03421002262878919,
                            -0.05475936513475744,
                            0
                    },
                    {
                            -0.03421002577942765,
                            0.05475936592241706,
                            0
                    },
                    {
                            0.03421002079091676,
                            0.0438240816633137,
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
                    {0, 0, 1.33391475},
                    {0, 0, 1.33391475},
                    {0, 0, 1.33391475},
                    {0, 0, 1.33391475},
            },
            // torque constant
            {0.0069975, 0.0069975, 0.0069975, 0.0069975},
            // T, RPM time constant
            {0.033, 0.033, 0.033, 0.033},
            {0.033, 0.033, 0.033, 0.033},
            // mass vehicle
            0.1554,
            // gravity
            {0, 0, -9.81},
            // J
            {
                    {
                            0.0001911085222263557,
                            0.0,
                            0.0
                    },
                    {
                            0.0,
                            9.093431124475021e-05,
                            0.0
                    },
                    {
                            0.0,
                            0.0,
                            0.000258351235459533
                    }
            },
            // J_inv
            {
                    {
                            1/0.0001911085222263557,
                            0.0,
                            0.0
                    },
                    {
                            0.0,
                            1/9.093431124475021e-05,
                            0.0
                    },
                    {
                            0.0,
                            0.0,
                            1/0.000258351235459533
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
