#ifndef LEARNING_TO_FLY_SIMULATOR_PARAMETERS_DYNAMICS_CRAZY_FLIE_H
#define LEARNING_TO_FLY_SIMULATOR_PARAMETERS_DYNAMICS_CRAZY_FLIE_H
#include "../../multirotor.h"

namespace rl_tools::rl::environments::multirotor::parameters::dynamics{
    template<typename SPEC, typename = utils::typing::enable_if_t<SPEC::N == 4>> // Crazyflie is a quadrotor
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
                    0,
                    0,
                    3.16e-10
            },
            // torque constant
//            0.025126582278481014,
            0.005964552,
            // mass vehicle
            0.027,
            // gravity
            {0, 0, -9.81},
            // J
            {
                    {
                            3.85e-6,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            3.85e-6,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            5.9675e-6
                    }
            },
            // J_inv
            {
                    {
                            259740.2597402597,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            259740.2597402597,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            167574.36112274823

                    }
            },
            // T, RPM time constant
            0.15,
            // hovering throttle (julia): sqrt((mass * 9.81/4 - thrust_curve[1])/thrust_curve[3]),
//            "hovering_throttle": 14475.809152959684,
            0.6670265023020774, // "hovering_throttle_relative"
            // action limit
            {0, 21702},
    };
}

#endif