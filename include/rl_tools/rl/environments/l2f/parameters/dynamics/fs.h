#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DYNAMICS_FS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DYNAMICS_FS_H
#include "../../multirotor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters::dynamics{
    namespace fs{
        template<typename SPEC, typename = rl_tools::utils::typing::enable_if_t<SPEC::N == 4>> // This is a quadrotor
        constexpr typename ParametersBase<SPEC>::Dynamics base = {
            // Rotor positions
//                array([[ 0.20895 , -0.240666,  0.      ],
//                [-0.20895 ,  0.240666,  0.      ],
//                [ 0.20895 ,  0.240666,  0.      ],
//                [-0.20895 , -0.240666,  0.      ]])
            {
                {
                    +0.20895,
                    -0.240666,
                    0
                },
                {
                    -0.20895,
                    +0.240666,
                    0
                },
                {
                    +0.20895,
                    +0.240666,
                    0
                },
                {
                    -0.20895,
                    -0.240666,
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
                2.55643, // alternative: 0
                0,
                28.06559 // alternative: 40.74310806823814
            },
            // torque constant
            0.01492,
            // mass vehicle
            3.35000,
            // gravity
            {0, 0, -9.81},
            // J
            {
                {
                    0.11361,
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    0.12552,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000,
                    0.22822
                }
            },
            // J_inv
            {
                {
                    8.802042073761113,
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    7.966857871255577,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000,
                    4.3817369205152925
                }
            },
            // T, RPM time constant
            0.06532,
            // hovering throttle (julia): sqrt((mass * 9.81/4 - thrust_curve[1])/thrust_curve[3]),
            0.44905530730868937,
            // action limit
            {0.3, 0.7}, // about [0.5thrust2weight, 2.5thrust2weight]
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
