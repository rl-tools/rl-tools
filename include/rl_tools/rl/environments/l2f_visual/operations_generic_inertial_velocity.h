#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_GENERIC_INERTIAL_VELOCITY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_GENERIC_INERTIAL_VELOCITY_H

#include "inertial_velocity.h"
#include "../l2f/quaternion_helper.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE&, rl::environments::l2f_visual::inertial_velocity::State<SPEC>& state, const typename SPEC::T gyro[3]){
        for(typename SPEC::TI axis = 0; axis < 3; axis++){
            state.velocity[axis] = 0;
            state.previous_gyro[axis] = gyro[axis];
        }
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void step(DEVICE& device, rl::environments::l2f_visual::inertial_velocity::State<SPEC>& state, const typename SPEC::T world_z_body[3], const typename SPEC::T specific_force[3], const typename SPEC::T gyro[3], typename SPEC::T dt){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        T rotation[3];
        T angle_squared = 0;
        for(TI axis = 0; axis < 3; axis++){
            rotation[axis] = -(state.previous_gyro[axis] + gyro[axis]) * dt / static_cast<T>(2);
            angle_squared += rotation[axis] * rotation[axis];
        }
        const T angle = math::sqrt(device.math, angle_squared);
        const T half_angle = angle / static_cast<T>(2);
        const T scale = angle > static_cast<T>(1e-6) ? math::sin(device.math, half_angle) / angle : static_cast<T>(0.5) - angle_squared / static_cast<T>(48);
        T quaternion[4] = {math::cos(device.math, half_angle), rotation[0] * scale, rotation[1] * scale, rotation[2] * scale};
        T rotated_velocity[3];
        using rl::environments::l2f::rotate_vector_by_quaternion;
        rotate_vector_by_quaternion<DEVICE, T>(quaternion, state.velocity, rotated_velocity);
        const T decay = SPEC::TIME_CONSTANT > 0 ? math::exp(device.math, -dt / SPEC::TIME_CONSTANT) : static_cast<T>(1);
        for(TI axis = 0; axis < 3; axis++){
            state.velocity[axis] = decay * rotated_velocity[axis] + dt * (specific_force[axis] - SPEC::GRAVITY * world_z_body[axis]);
            state.previous_gyro[axis] = gyro[axis];
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
