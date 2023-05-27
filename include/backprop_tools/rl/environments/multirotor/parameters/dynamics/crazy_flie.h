
#include "../../multirotor.h"

namespace backprop_tools::rl::environments::multirotor::parameters::dynamics{
    template<typename T, typename TI, typename REWARD_FUNCTION>
    constexpr typename Parameters <T, TI, TI(4), REWARD_FUNCTION>::Dynamics crazy_flie = {
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

}
/*
Some calculations

 Bitcraze:
 J = [7.7e-6 0 0; 0 7.7e-6 0; 0 0 1.1935e-5]; J_inv = inv(J)
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
 hovering_level = hovering_rpm / max_rpm * 2 - 1;
 */

