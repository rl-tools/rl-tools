#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_ACTUATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_ACTUATION_H

#include "../multirotor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    template<typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T rotor_rpm_from_command(DEVICE&, const PARAMETERS& parameters, T command){
        return command * parameters.dynamics.max_rpm / parameters.dynamics.action_limit.max;
    }
    template<typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T rotor_angular_velocity_from_command(DEVICE& device, const PARAMETERS& parameters, T command){
        return rotor_rpm_from_command(device, parameters, command) * ((T)2 * math::PI<T> / (T)60);
    }
    template<typename DEVICE, typename PARAMETERS, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT T rotor_thrust_from_command(DEVICE&, const PARAMETERS& parameters, TI rotor_i, T command){
        return parameters.dynamics.rotor_thrust_coefficients[rotor_i][0] + parameters.dynamics.rotor_thrust_coefficients[rotor_i][1] * command + parameters.dynamics.rotor_thrust_coefficients[rotor_i][2] * command * command;
    }
    template<typename DEVICE, typename PARAMETERS, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT T rotor_command_from_thrust(DEVICE& device, const PARAMETERS& parameters, TI rotor_i, T thrust){
        T min_command = parameters.dynamics.action_limit.min;
        T max_command = parameters.dynamics.action_limit.max;
        T min_thrust = rotor_thrust_from_command(device, parameters, rotor_i, min_command);
        T max_thrust = rotor_thrust_from_command(device, parameters, rotor_i, max_command);
        rl_tools::utils::assert_exit(device, min_thrust <= max_thrust, "min_thrust > max_thrust");
        thrust = math::clamp(device.math, thrust, min_thrust, max_thrust);
        T c0 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][0];
        T c1 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][1];
        T c2 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][2];
        T d_min = c1 + (T)2 * c2 * min_command;
        T d_max = c1 + (T)2 * c2 * max_command;
        rl_tools::utils::assert_exit(device, d_min >= (T)0 && d_max >= (T)0, "rotor thrust curve is not monotone increasing over action range");
        T command = min_command;
        T eps = (T)1e-12;
        if(math::abs(device.math, c2) > eps){
            T discriminant = c1 * c1 - (T)4 * c2 * (c0 - thrust);
            T sqrt_discriminant = math::sqrt(device.math, math::max(device.math, discriminant, (T)0));
            T root_a = (-c1 + sqrt_discriminant) / ((T)2 * c2);
            T root_b = (-c1 - sqrt_discriminant) / ((T)2 * c2);
            bool root_a_valid = root_a >= min_command && root_a <= max_command;
            command = root_a_valid ? root_a : root_b;
        }
        else if(math::abs(device.math, c1) > eps){
            command = (thrust - c0) / c1;
        }
        return math::clamp(device.math, command, min_command, max_command);
    }
    template<typename DEVICE, typename PARAMETERS>
    RL_TOOLS_FUNCTION_PLACEMENT typename PARAMETERS::T gravity_norm(DEVICE& device, const PARAMETERS& parameters){
        const auto& gravity = parameters.dynamics.gravity;
        return math::sqrt(device.math, gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]);
    }
    template<typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T ideal_rates_collective_acceleration(DEVICE& device, const PARAMETERS& parameters, T normalized_collective){
        return (normalized_collective + (T)1) / (T)2 * parameters.ideal_rates.thrust_to_weight * gravity_norm(device, parameters);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
