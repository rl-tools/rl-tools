#ifndef LEARNING_TO_FLY_SIMULATOR_PARAMETERS_DYNAMICS_X500_REAL_H
#define LEARNING_TO_FLY_SIMULATOR_PARAMETERS_DYNAMICS_X500_REAL_H
#include "../../multirotor.h"

namespace rl_tools::rl::environments::multirotor::parameters::dynamics{
    namespace x500{
        template<typename SPEC, typename = utils::typing::enable_if_t<SPEC::N == 4>> // This is a quadrotor
        constexpr typename ParametersBase<SPEC>::Dynamics real = {
            // Rotor positions
            {
                {
                    +0.176776695296636,
                    -0.176776695296636,
                    0
                },
                {
                    -0.176776695296636,
                    +0.176776695296636,
                    0
                },
                {
                    +0.176776695296636,
                    +0.176776695296636,
                    0
                },
                {
                    -0.176776695296636,
                    -0.176776695296636,
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
                {0, 0, -1},
                {0, 0, +1},
                {0, 0, +1},
            },
            // thrust constants
            {
                1.425,
                0,
                15.85
            },
            // torque constant
//            0.11697849233439939,
            0.2,
            // mass vehicle
            2.000,
            // gravity
            {0, 0, -9.81},
            // J
            {
                {
                    0.0619,
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    0.0694,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000,
                    0.11104000000000001
                }
            },
            // J_inv
            {
                {
                    16.155088852988694,
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    14.40922190201729,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000,
                    9.005763688760807
                }
            },
            // T, RPM time constant
            0.03,
            // hovering throttle (julia): sqrt((mass * 9.81/4 - thrust_curve[1])/thrust_curve[3]),
            0.4685705492468035,
            // action limit
            {0.3, 0.7},
        };
    }
}

#endif