#include "../../../../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_DYANMICS_CRAZY_FLIE_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_DYANMICS_CRAZY_FLIE_H

#include "../../multirotor.h"

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::rl::environments::multirotor::parameters::dynamics{
    template<typename T, typename TI, typename REWARD_FUNCTION>
    constexpr typename ParametersBase <T, TI, TI(4), REWARD_FUNCTION>::Dynamics crazy_flie_old = {
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
            0.005964552,
            // mass vehicle
            0.027,
            // gravity
            {0, 0, -9.81},
            // J
            {
                    {
                            7.7e-6,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            7.7e-6,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            1.1935e-5
                    }
            },
            // J_inv
            {
                    {
                            1.2987e5,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            1.2987e5,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            5.16796e5
                    }
            },
            // T, RPM time constant
            0.15,
            // action limit
            {0, 21702},
    };
    template<typename T, typename TI, typename REWARD_FUNCTION>
    constexpr typename ParametersBase <T, TI, TI(4), REWARD_FUNCTION>::Dynamics crazy_flie_old_reduced_inertia = {
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
            // action limit
            {0, 21702},
    };
    template<typename T, typename TI, typename REWARD_FUNCTION>
    constexpr typename ParametersBase <T, TI, TI(4), REWARD_FUNCTION>::Dynamics crazy_flie_low_inertia = {
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
                            7.7e-6,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            7.7e-6,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            1.1935e-5
                    }
            },
            // J_inv
            {
                    {
                            1.2987e5,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            1.2987e5,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            83787.2
                    }
            },
            // action limit
            0.01, // T, RPM time constant
            {0, 21702.1},
    };
    template<typename T, typename TI, typename REWARD_FUNCTION>
    constexpr typename ParametersBase <T, TI, TI(4), REWARD_FUNCTION>::Dynamics crazy_flie_medium_inertia = {
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
                            1.6e-5,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            1.6e-5,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            2.9e-5
                    }
            },
            // J_inv
            {
                    {
                            62500.0,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            62500.0,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            34482.8
                    }
            },
            // action limit
            0.03, // T, RPM time constant
            {0, 21702.1},
    };
    template<typename T, typename TI, typename REWARD_FUNCTION>
    constexpr typename ParametersBase <T, TI, TI(4), REWARD_FUNCTION>::Dynamics crazy_flie_high_inertia = {
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
                            7.7e-5,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            7.7e-5,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            1.1935e-4
                    }
            },
            // J_inv
            {
                    {
                            1.2987e4,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            1.2987e4,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            8378.72
                    }
            },
            // action limit
            0.01, // T, RPM time constant
            {0, 21702.1},
    };
    template<typename T, typename TI, typename REWARD_FUNCTION>
    constexpr typename ParametersBase <T, TI, TI(4), REWARD_FUNCTION>::Dynamics crazy_flie_very_high_inertia = {
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
                            3.85e-4,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            3.85e-4,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            5.95e-4
                    }
            },
            // J_inv
            {
                    {
                            2597,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            2597,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            1680
                    }
            },
            // action limit
            0.01, // T, RPM time constant
            {0, 21702.1},
    };

}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
/*
Some calculations

 Low inertia:
 J = [7.7e-6 0 0; 0 7.7e-6 0; 0 0 1.1935e-5]; J_inv = inv(J)
 thrust_curve = [0, 0, 3.16e-10]
 max_rpm =  21702.1;
 mass = 0.027;

 Medium  inertia:
 J = [16e-6 0 0; 0 16e-6 0; 0 0 29e-6]; J_inv = inv(J)
 thrust_curve = [0, 0, 3.16e-10]
 max_rpm =  21702.1;
 mass = 0.027;

 High inertia:
 J = [7.7e-5 0 0; 0 7.7e-5 0; 0 0 1.1935e-4]; J_inv = inv(J)
 thrust_curve = [0, 0, 3.16e-10]
 max_rpm =  21702.1;
 mass = 0.027;

 Very High inertia:
 J = [3.85e-4 0 0; 0 3.85e-4 0; 0 0 5.95e-4]; J_inv = inv(J)
 thrust_curve = [0, 0, 3.16e-10]
 max_rpm =  21702.1;
 mass = 0.027;


 using LinearAlgebra
 rotor_1_pos = [0.028, -0.028, 0];
 rotor_2_pos = [-0.028, -0.028, 0];
 rotor_3_pos = [-0.028, 0.028, 0];
 rotor_4_pos = [0.028, 0.028, 0];
 max_thrust_magnitude = thrust_curve[1] + thrust_curve[2] * max_rpm + thrust_curve[3] * max_rpm^2;
 max_thrust_vector = [0, 0, max_thrust_magnitude];
 max_torque = cross(rotor_3_pos, max_thrust_vector) + cross(rotor_4_pos, max_thrust_vector);
 max_angular_acceleration = J_inv * max_torque
 thrust_to_weight = max_thrust_magnitude * 4 / mass / 9.81
 hovering_rpm = sqrt(mass * 9.81 / 4 / thrust_curve[3])
# min_rpm = sqrt(mass * 9.81 / 2 / 4 / thrust_curve[3])
 min_rpm = 0
 hovering_level = (hovering_rpm - min_rpm) / (max_rpm - min_rpm) * 2 - 1;
 */
/*
 * first principles (box inertia)
using LinearAlgebra
a = norm([0.028, 0.028, 0])
b = norm([0.028, -0.028, 0])
c = b /15
m = 0.027
Ixx = (1/12)m*(b^2 + c^2)
Iyy = (1/12)m*(a^2 + c^2)
Izz = (1/12)m*(a^2 + b^2)
 */


#endif