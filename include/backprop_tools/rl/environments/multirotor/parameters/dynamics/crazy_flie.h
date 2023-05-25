
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
                    5.48e-4,
                    1.03e-6,
                    2.13e-11
            },
            // torque constant
            0.005964552, // 7.9379e-6/3.25e-4 (this is relative to the resulting thrust from the thrust curve)
            // mass vehicle
            0.027,
            // gravity
            {0, 0, -9.81},
            // J
            {
                    {
                            0.000016572,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.000016572,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            0.000029262
                    }
            },
            // J_inv
            {
                    {
                            60342.7,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            60342.7,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            34174.0
                    }
            },
            // action limit
            {0, 65535},
    };

}
/*
Some calculations

 Bitcraze:
 J = [0.000016572 0 0; 0 0.000016572 0; 0 0 0.000029262]; J_inv = inv(J)
 thrust_curve = [5.48e-4, 1.03e-6, 2.13e-11]
 max_rpm = 65535;


 using LinearAlgebra
 rotor_1_pos = [0.028, -0.028, 0];
 rotor_2_pos = [-0.028, -0.028, 0];
 rotor_3_pos = [-0.028, 0.028, 0];
 rotor_4_pos = [0.028, 0.028, 0];
 max_thrust_magnitude = thrust_curve[1] + thrust_curve[2] * max_rpm + thrust_curve[3] * max_rpm^2;
 max_thrust_vector = [0, 0, max_thrust_magnitude];
 max_torque = cross(rotor_3_pos, max_thrust_vector) + cross(rotor_4_pos, max_thrust_vector);
 max_angular_acceleration = J_inv * max_torque
 */

