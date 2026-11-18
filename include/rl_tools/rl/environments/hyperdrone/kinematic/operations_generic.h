#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_KINEMATIC_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_KINEMATIC_OPERATIONS_GENERIC_H

#include "kinematic.h"
#include "../../l2f/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::environments::hyperdrone {
        // commands at rest are zero, while the l2f initial state fills the action history from the hover rotor speed
        template <typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _clear_action_history(l2f::StateRotorsHistory<STATE_SPEC>* state) {
            using STATE = l2f::StateRotorsHistory<STATE_SPEC>;
            for (typename STATE::TI step_i = 0; step_i < STATE::HISTORY_LENGTH; step_i++) {
                for (typename STATE::TI action_i = 0; action_i < STATE::ACTION_DIM; action_i++) {
                    state->action_history[step_i][action_i] = 0;
                }
            }
        }
        RL_TOOLS_FUNCTION_PLACEMENT inline void _clear_action_history(void*) {}
    }
    // level and at rest: the position within max_position per axis of mdp.init.position, the yaw within max_angle of the twist of mdp.init.orientation about z (the orientation itself when level); guided starts take the centre
    template <typename DEVICE, typename SPEC, typename LIMITS, typename PARAMETERS, typename STATE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_state(DEVICE& device, rl::environments::hyperdrone::KinematicMultirotor<SPEC, LIMITS>& env, PARAMETERS& parameters, STATE& state, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        const auto& init = parameters.mdp.init;
        initial_state(device, env, parameters, state);
        rl::environments::hyperdrone::_clear_action_history(&state);
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            state.position[axis_i] = init.position[axis_i];
        }
        T yaw_offset = 0;
        if (random::uniform_real_distribution(device.random, (T)0, (T)1, rng) >= init.guidance) {
            for (TI axis_i = 0; axis_i < 3; axis_i++) {
                state.position[axis_i] += random::uniform_real_distribution(device.random, -init.max_position, init.max_position, rng);
            }
            yaw_offset = random::uniform_real_distribution(device.random, -init.max_angle, init.max_angle, rng);
        }
        const T twist_norm = math::sqrt(device.math, init.orientation[0] * init.orientation[0] + init.orientation[3] * init.orientation[3]);
        const T half_cos = math::cos(device.math, yaw_offset / 2);
        const T half_sin = math::sin(device.math, yaw_offset / 2);
        state.orientation[0] = (half_cos * init.orientation[0] - half_sin * init.orientation[3]) / twist_norm;
        state.orientation[1] = 0;
        state.orientation[2] = 0;
        state.orientation[3] = (half_sin * init.orientation[0] + half_cos * init.orientation[3]) / twist_norm;
    }
    template <typename DEVICE, typename SPEC, typename LIMITS, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::hyperdrone::KinematicMultirotor<SPEC, LIMITS>& env, PARAMETERS& parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, STATE& next_state, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 4, "the kinematic action is [linear_velocity (3), yaw_rate]");
        const T dt = parameters.integration.dt;
        T linear_velocity[3];
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            linear_velocity[axis_i] = math::clamp(device.math, (T)get(action, 0, axis_i), (T)-1, (T)1) * LIMITS::LINEAR_VELOCITY;
        }
        const T yaw_rate = math::clamp(device.math, (T)get(action, 0, 3), (T)-1, (T)1) * LIMITS::YAW_RATE;
        const T twist_norm = math::sqrt(device.math, state.orientation[0] * state.orientation[0] + state.orientation[3] * state.orientation[3]);
        const T w = state.orientation[0] / twist_norm;
        const T z = state.orientation[3] / twist_norm;
        const T half_angle = yaw_rate * dt / 2;
        const T half_cos = math::cos(device.math, half_angle);
        const T half_sin = math::sin(device.math, half_angle);
        const T sinc = math::abs(device.math, half_angle) < (T)1e-4 ? (T)1 - half_angle * half_angle / 6 : half_sin / half_angle;
        // constant twist over the interval: the planar displacement is the arc's chord, i.e. the velocity rotated to the mid-interval heading and shortened by sinc of the half turn
        const T yaw_cos = w * w - z * z;
        const T yaw_sin = 2 * w * z;
        const T chord_cos = yaw_cos * half_cos - yaw_sin * half_sin;
        const T chord_sin = yaw_sin * half_cos + yaw_cos * half_sin;
        const T next_yaw_cos = chord_cos * half_cos - chord_sin * half_sin;
        const T next_yaw_sin = chord_sin * half_cos + chord_cos * half_sin;
        next_state = state;
        next_state.position[0] = state.position[0] + dt * sinc * (chord_cos * linear_velocity[0] - chord_sin * linear_velocity[1]);
        next_state.position[1] = state.position[1] + dt * sinc * (chord_sin * linear_velocity[0] + chord_cos * linear_velocity[1]);
        next_state.position[2] = state.position[2] + dt * linear_velocity[2];
        next_state.orientation[0] = half_cos * w - half_sin * z;
        next_state.orientation[1] = 0;
        next_state.orientation[2] = 0;
        next_state.orientation[3] = half_sin * w + half_cos * z;
        next_state.linear_velocity[0] = next_yaw_cos * linear_velocity[0] - next_yaw_sin * linear_velocity[1];
        next_state.linear_velocity[1] = next_yaw_sin * linear_velocity[0] + next_yaw_cos * linear_velocity[1];
        next_state.linear_velocity[2] = linear_velocity[2];
        next_state.angular_velocity[0] = 0;
        next_state.angular_velocity[1] = 0;
        next_state.angular_velocity[2] = yaw_rate;
        post_integration(device, env, parameters, state, action, next_state, rng);
        return dt;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
