#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RAPTOR_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RAPTOR_OPERATIONS_GENERIC_H

#include "raptor.h"
#include "../../l2f/quaternion_helper.h"
#include "../../../../nn/layers/dense/operations_generic.h"
#include "../../../../nn/layers/gru/operations_generic.h"
#include "../../../../nn_models/sequential/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::environments::hyperdrone::Raptor<SPEC>& policy) {
        malloc(device, policy.model);
        malloc(device, policy.state);
        malloc(device, policy.buffer);
        malloc(device, policy.observations);
        malloc(device, policy.actions);
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::environments::hyperdrone::Raptor<SPEC>& policy) {
        free(device, policy.model);
        free(device, policy.state);
        free(device, policy.buffer);
        free(device, policy.observations);
        free(device, policy.actions);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    void reset(DEVICE& device, rl::environments::hyperdrone::Raptor<SPEC>& policy, RNG& rng) {
        reset(device, policy.model, policy.state, rng);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    void evaluate_step(DEVICE& device, rl::environments::hyperdrone::Raptor<SPEC>& policy, RNG& rng) {
        Mode<nn::layers::gru::NoAutoResetMode<mode::Default<>>> mode;
        evaluate_step(device, policy.model, policy.observations, policy.state, policy.actions, policy.buffer, rng, mode);
    }

    template <typename DEVICE, typename SPEC, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, rl::environments::hyperdrone::Raptor<SPEC>& policy, const STATE& state, const rl::environments::hyperdrone::raptor::Target<typename SPEC::T>& target, const typename SPEC::T previous_action[4], typename SPEC::TI instance_i) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using namespace rl::environments::l2f;
        T inverse_yaw[4] = {math::cos(device.math, target.yaw / (T)2), 0, 0, -math::sin(device.math, target.yaw / (T)2)};
        T position_error[3];
        T velocity_error[3];
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            position_error[axis_i] = state.position[axis_i] - target.position[axis_i];
            velocity_error[axis_i] = state.linear_velocity[axis_i] - target.linear_velocity[axis_i];
        }
        T rotated_position[3];
        T rotated_velocity[3];
        T orientation[4];
        T rotation[3][3];
        rotate_vector_by_quaternion<DEVICE, T>(inverse_yaw, position_error, rotated_position);
        rotate_vector_by_quaternion<DEVICE, T>(inverse_yaw, velocity_error, rotated_velocity);
        quaternion_multiplication<DEVICE, T>(inverse_yaw, state.orientation, orientation);
        quaternion_to_rotation_matrix<DEVICE, T>(orientation, rotation);
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            set(device, policy.observations, math::clamp(device.math, rotated_position[axis_i], -SPEC::POSITION_CLIP, SPEC::POSITION_CLIP), instance_i, axis_i);
            set(device, policy.observations, math::clamp(device.math, rotated_velocity[axis_i], -SPEC::VELOCITY_CLIP, SPEC::VELOCITY_CLIP), instance_i, 12 + axis_i);
            set(device, policy.observations, state.angular_velocity[axis_i], instance_i, 15 + axis_i);
            for (TI column_i = 0; column_i < 3; column_i++) {
                set(device, policy.observations, rotation[axis_i][column_i], instance_i, 3 + 3 * axis_i + column_i);
            }
        }
        for (TI action_i = 0; action_i < 4; action_i++) {
            set(device, policy.observations, previous_action[action_i], instance_i, 18 + action_i);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
