#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_HELPER_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_HELPER_GENERIC_H

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T>
    void compare_parameters(DEVICE& device, const T& nominal, const T& perturbed){
        auto percentage_change = [](auto nominal_value, auto perturbed_value){
            return ((perturbed_value - nominal_value) / nominal_value) * 100.0;
        };
        log(device, device.logger, "Mass: ", nominal.mass, " -> ", perturbed.mass, " (", percentage_change(nominal.mass, perturbed.mass), "%)");
        for (int i = 0; i < 3; ++i) {
            log(device, device.logger, "J[", i, "][", i, "]: ", nominal.J[i][i], " -> ", perturbed.J[i][i], " (", percentage_change(nominal.J[i][i], perturbed.J[i][i]), "%)");
        }
        for (int i = 0; i < 3; ++i) {
            log(device, device.logger, "Rotor thrust coefficient[", i, "]: ", nominal.rotor_thrust_coefficients[i], " -> ", perturbed.rotor_thrust_coefficients[i], " (", percentage_change(nominal.rotor_thrust_coefficients[i], perturbed.rotor_thrust_coefficients[i]), "%)");
        }
        log(device, device.logger, "Rotor torque constant: ", nominal.rotor_torque_constant, " -> ", perturbed.rotor_torque_constant, " (", percentage_change(nominal.rotor_torque_constant, perturbed.rotor_torque_constant), "%)");
        log(device, device.logger, "Motor time constant: ", nominal.motor_time_constant, " -> ", perturbed.motor_time_constant, " (", percentage_change(nominal.motor_time_constant, perturbed.motor_time_constant), "%)");
        log(device, device.logger, "Hovering throttle relative: ", nominal.hovering_throttle_relative, " -> ", perturbed.hovering_throttle_relative, " (", percentage_change(nominal.hovering_throttle_relative, perturbed.hovering_throttle_relative), "%)");
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif