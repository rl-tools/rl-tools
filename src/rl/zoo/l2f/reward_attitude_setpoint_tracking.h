#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_REWARD_ATTITUDE_SETPOINT_TRACKING_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_REWARD_ATTITUDE_SETPOINT_TRACKING_H

#include "setpoint_attitude.h"

#include <rl_tools/rl/environments/l2f/multirotor.h>
#include <rl_tools/rl/environments/l2f/quaternion_helper.h>
#include <rl_tools/utils/generic/typing.h>

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters::reward_functions{
    template<typename T>
    struct AttitudeSetpointTrackingSquared{
        bool non_negative;
        T scale;
        T constant;
        T tilt;
        T yaw_rate;
        T angular_velocity_xy;
        T thrust_g;
        T d_action;
        T action_saturation;
        struct Components{
            T tilt_cost;
            T yaw_rate_cost;
            T angular_velocity_xy_cost;
            T thrust_g_cost;
            T actual_thrust_g;
            T d_action_cost;
            T action_saturation_cost;
            T weighted_cost;
            T scaled_weighted_cost;
            T reward;
        };
    };

    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reward_components(DEVICE& device, const Multirotor<SPEC>& env, const PARAMETERS& parameters, const AttitudeSetpointTrackingSquared<T>& reward_parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, const STATE& next_state, typename AttitudeSetpointTrackingSquared<T>::Components& components, RNG& rng){
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        T target_world_z_body[3];
        roll_pitch_to_world_z_body(device, state.target_roll, state.target_pitch, target_world_z_body);
        T current_world_z_body[3];
        quaternion_to_world_z_body<DEVICE, T>(next_state.orientation, current_world_z_body);
        T tilt_diff_sq = 0;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            T diff = current_world_z_body[dim_i] - target_world_z_body[dim_i];
            tilt_diff_sq += diff * diff;
        }
        components.tilt_cost = math::sqrt(device.math, tilt_diff_sq);

        components.yaw_rate_cost = math::abs(device.math, next_state.angular_velocity[2] - state.target_yaw_rate);
        components.angular_velocity_xy_cost = math::sqrt(device.math,
            next_state.angular_velocity[0] * next_state.angular_velocity[0] +
            next_state.angular_velocity[1] * next_state.angular_velocity[1]
        );

        T body_z_body[3] = {0, 0, 1};
        T body_z_world[3];
        rotate_vector_by_quaternion<DEVICE, T>(next_state.orientation, body_z_body, body_z_world);
        T gravity_norm = math::sqrt(device.math,
            parameters.dynamics.gravity[0] * parameters.dynamics.gravity[0] +
            parameters.dynamics.gravity[1] * parameters.dynamics.gravity[1] +
            parameters.dynamics.gravity[2] * parameters.dynamics.gravity[2]
        );
        T specific_force_world[3];
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            specific_force_world[dim_i] = next_state.linear_acceleration[dim_i] - parameters.dynamics.gravity[dim_i];
        }
        T thrust_projection = 0;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            thrust_projection += specific_force_world[dim_i] * body_z_world[dim_i];
        }
        components.actual_thrust_g = thrust_projection / gravity_norm;
        components.thrust_g_cost = math::abs(device.math, components.actual_thrust_g - state.target_thrust_g);

        T d_action_sq = 0;
        T action_saturation_sq = 0;
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T action_value = get(action, 0, action_i);
            T diff = action_value - state.last_action[action_i];
            d_action_sq += diff * diff;
            T saturation = math::max(device.math, math::abs(device.math, action_value) - (T)0.95, (T)0);
            action_saturation_sq += saturation * saturation;
        }
        components.d_action_cost = d_action_sq;
        components.action_saturation_cost = action_saturation_sq;

        components.weighted_cost =
              reward_parameters.tilt * components.tilt_cost
            + reward_parameters.yaw_rate * components.yaw_rate_cost
            + reward_parameters.angular_velocity_xy * components.angular_velocity_xy_cost
            + reward_parameters.thrust_g * components.thrust_g_cost
            + reward_parameters.d_action * components.d_action_cost
            + reward_parameters.action_saturation * components.action_saturation_cost;
        components.scaled_weighted_cost = reward_parameters.scale * components.weighted_cost;
        components.reward = -components.scaled_weighted_cost + reward_parameters.constant;
        components.reward = (components.reward > 0 || !reward_parameters.non_negative) ? components.reward : 0;
    }

    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename ACTION_SPEC, typename STATE, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const Multirotor<SPEC>& env, const PARAMETERS& parameters, const AttitudeSetpointTrackingSquared<T>& reward_parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, const STATE& next_state, RNG& rng){
        typename AttitudeSetpointTrackingSquared<T>::Components components;
        reward_components(device, env, parameters, reward_parameters, state, action, next_state, components, rng);
        return components.reward;
    }

    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void log_reward(DEVICE& device, const Multirotor<SPEC>& env, const PARAMETERS& parameters, const AttitudeSetpointTrackingSquared<T>& reward_parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, const STATE& next_state, RNG& rng, typename DEVICE::index_t cadence = 1){
        typename AttitudeSetpointTrackingSquared<T>::Components components;
        reward_components(device, env, parameters, reward_parameters, state, action, next_state, components, rng);
        add_scalar(device, device.logger, "reward/tilt_cost", components.tilt_cost, cadence);
        add_scalar(device, device.logger, "reward/yaw_rate_cost", components.yaw_rate_cost, cadence);
        add_scalar(device, device.logger, "reward/angular_velocity_xy_cost", components.angular_velocity_xy_cost, cadence);
        add_scalar(device, device.logger, "reward/thrust_g_cost", components.thrust_g_cost, cadence);
        add_scalar(device, device.logger, "reward/actual_thrust_g", components.actual_thrust_g, cadence);
        add_scalar(device, device.logger, "reward/d_action_cost", components.d_action_cost, cadence);
        add_scalar(device, device.logger, "reward/action_saturation_cost", components.action_saturation_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/tilt", reward_parameters.tilt * components.tilt_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/yaw_rate", reward_parameters.yaw_rate * components.yaw_rate_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/angular_velocity_xy", reward_parameters.angular_velocity_xy * components.angular_velocity_xy_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/thrust_g", reward_parameters.thrust_g * components.thrust_g_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/d_action", reward_parameters.d_action * components.d_action_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/action_saturation", reward_parameters.action_saturation * components.action_saturation_cost, cadence);
        add_scalar(device, device.logger, "reward/weighted_cost", components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward/scaled_weighted_cost", components.scaled_weighted_cost, cadence);
        add_scalar(device, device.logger, "reward/reward", components.reward, cadence);
    }

    template<typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto name(DEVICE& device, const AttitudeSetpointTrackingSquared<T>& reward_parameters){
        return "attitude_setpoint_tracking_squared";
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename T>
    std::string json(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::l2f::parameters::reward_functions::AttitudeSetpointTrackingSquared<T>& parameters){
        std::string json_string = "{";
        json_string += "\"non_negative\": " + std::string(parameters.non_negative ? "true" : "false") + ", ";
        json_string += "\"scale\": " + std::to_string(parameters.scale) + ", ";
        json_string += "\"constant\": " + std::to_string(parameters.constant) + ", ";
        json_string += "\"tilt\": " + std::to_string(parameters.tilt) + ", ";
        json_string += "\"yaw_rate\": " + std::to_string(parameters.yaw_rate) + ", ";
        json_string += "\"angular_velocity_xy\": " + std::to_string(parameters.angular_velocity_xy) + ", ";
        json_string += "\"thrust_g\": " + std::to_string(parameters.thrust_g) + ", ";
        json_string += "\"d_action\": " + std::to_string(parameters.d_action) + ", ";
        json_string += "\"action_saturation\": " + std::to_string(parameters.action_saturation);
        json_string += "}";
        return json_string;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
