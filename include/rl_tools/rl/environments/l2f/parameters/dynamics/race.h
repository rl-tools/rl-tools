#ifndef LEARNING_TO_FLY_SIMULATOR_PARAMETERS_DYNAMICS_RACE_H
#define LEARNING_TO_FLY_SIMULATOR_PARAMETERS_DYNAMICS_RACE_H
#include "../../multirotor.h"

namespace rl_tools::rl::environments::multirotor::parameters::dynamics{
    template<typename SPEC, typename = utils::typing::enable_if_t<SPEC::N == 4>> // This is a quadrotor
    constexpr typename ParametersBase<SPEC>::Dynamics race = {
            // Rotor positions
            {
                    {
                            0.0775,
                            -0.0981,
                            0

                    },
                    {
                            -0.0775,
                            0.0981,
                            0

                    },
                    {
                            0.0775,
                            0.0981,
                            0

                    },
                    {
                            -0.0775,
                            -0.0981,
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
                    0,
                    0,
                    143
            },
            // torque constant
//            0.025126582278481014,
            0.005964552,
            // mass vehicle
            1.000,
            // gravity
            {0, 0, -9.81},
            // J
            {
                    {
                            0.0008393451220413083,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0014118914879009208,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            0.002189767656999172
                    }
            },
            // J_inv
            {
                    {
                            1191.4050296354555,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            708.2697279283936,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            456.66945385903927
                    }
            },
            // T, RPM time constant
            0.04,
            // hovering throttle (julia): sqrt((mass * 9.81/4 - thrust_curve[1])/thrust_curve[3]),
            0.13095934350152208,
            // action limit
            {0.1, 0.2},
    };
}

#endif