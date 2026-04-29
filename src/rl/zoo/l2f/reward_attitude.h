#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_REWARD_ATTITUDE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_REWARD_ATTITUDE_H

#include <rl_tools/rl/environments/l2f/multirotor.h>
#include <rl_tools/utils/generic/typing.h>
#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters::reward_functions{
    template<typename T>
    struct AttitudeSquared{
        bool non_negative;
        T scale;
        T constant;
        T termination_penalty;
        T tilt;
        T angular_velocity;
        T action;
        T linear_velocity;
        struct Components{
            T tilt_cost;
            T angular_vel_cost;
            T action_cost;
            T linear_vel_cost;
            T weighted_cost;
            T scaled_weighted_cost;
            T reward;
        };
    };

    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reward_components(DEVICE& device, const Multirotor<SPEC>& env, const PARAMETERS& parameters, const AttitudeSquared<T>& reward_parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, const STATE& next_state, typename AttitudeSquared<T>::Components& components, RNG& rng){
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        // Linear-in-angle tilt cost: sqrt(2*(q_x^2+q_y^2)) ≈ theta/sqrt(2) near
        // identity, so the gradient does not vanish at small offsets. Range
        // [0, sqrt(2)] (sqrt(2) at 90deg tilt).
        const T* q = state.orientation;
        T tilt_quadratic = 2*(q[1]*q[1] + q[2]*q[2]);
        components.tilt_cost = math::sqrt(device.math, tilt_quadratic);
        components.angular_vel_cost = math::sqrt(device.math,
            state.angular_velocity[0]*state.angular_velocity[0] +
            state.angular_velocity[1]*state.angular_velocity[1] +
            state.angular_velocity[2]*state.angular_velocity[2]);
        T action_diff_sq = 0;
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T action_throttle_relative = (get(action, 0, action_i) + (T)1.0)/(T)2.0;
            T diff = action_throttle_relative - parameters.dynamics.hovering_throttle_relative;
            action_diff_sq += diff * diff;
        }
        components.action_cost = action_diff_sq;
        components.linear_vel_cost = math::sqrt(device.math,
            state.linear_velocity[0]*state.linear_velocity[0] +
            state.linear_velocity[1]*state.linear_velocity[1] +
            state.linear_velocity[2]*state.linear_velocity[2]);
        components.weighted_cost =
              reward_parameters.tilt * components.tilt_cost
            + reward_parameters.angular_velocity * components.angular_vel_cost
            + reward_parameters.action * components.action_cost
            + reward_parameters.linear_velocity * components.linear_vel_cost;
        components.scaled_weighted_cost = reward_parameters.scale * components.weighted_cost;
        // No termination_penalty branch: this header must be includable before
        // l2f/operations_generic.h declares rl_tools::terminated, so we avoid
        // that dependency. termination_penalty is unused for this target.
        components.reward = -components.scaled_weighted_cost + reward_parameters.constant;
        components.reward = (components.reward > 0 || !reward_parameters.non_negative) ? components.reward : 0;
    }

    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename ACTION_SPEC, typename STATE, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const Multirotor<SPEC>& env, const PARAMETERS& parameters, const AttitudeSquared<T>& reward_parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, const STATE& next_state, RNG& rng){
        typename AttitudeSquared<T>::Components components;
        reward_components(device, env, parameters, reward_parameters, state, action, next_state, components, rng);
        return components.reward;
    }

    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void log_reward(DEVICE& device, const Multirotor<SPEC>& env, const PARAMETERS& parameters, const AttitudeSquared<T>& reward_parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, const STATE& next_state, RNG& rng, typename DEVICE::index_t cadence = 1){
        typename AttitudeSquared<T>::Components components;
        reward_components(device, env, parameters, reward_parameters, state, action, next_state, components, rng);
        add_scalar(device, device.logger, "reward/tilt_cost",        components.tilt_cost,        cadence);
        add_scalar(device, device.logger, "reward/angular_vel_cost", components.angular_vel_cost, cadence);
        add_scalar(device, device.logger, "reward/action_cost",      components.action_cost,      cadence);
        add_scalar(device, device.logger, "reward/linear_vel_cost",  components.linear_vel_cost,  cadence);
        add_scalar(device, device.logger, "reward_weighted/tilt",        reward_parameters.tilt             * components.tilt_cost,        cadence);
        add_scalar(device, device.logger, "reward_weighted/angular_vel", reward_parameters.angular_velocity * components.angular_vel_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/action",      reward_parameters.action           * components.action_cost,      cadence);
        add_scalar(device, device.logger, "reward_weighted/linear_vel",  reward_parameters.linear_velocity  * components.linear_vel_cost,  cadence);
        add_scalar(device, device.logger, "reward/weighted_cost",        components.weighted_cost,        cadence);
        add_scalar(device, device.logger, "reward/scaled_weighted_cost", components.scaled_weighted_cost, cadence);
        add_scalar(device, device.logger, "reward/reward",               components.reward,               cadence);
    }

    template<typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto name(DEVICE& device, const AttitudeSquared<T>& reward_parameters){
        return "attitude_squared";
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename T>
    std::string json(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::l2f::parameters::reward_functions::AttitudeSquared<T>& parameters){
        std::string json_string = "{";
        json_string += "\"non_negative\": " + std::string(parameters.non_negative ? "true" : "false") + ", ";
        json_string += "\"scale\": " + std::to_string(parameters.scale) + ", ";
        json_string += "\"constant\": " + std::to_string(parameters.constant) + ", ";
        json_string += "\"termination_penalty\": " + std::to_string(parameters.termination_penalty) + ", ";
        json_string += "\"tilt\": " + std::to_string(parameters.tilt) + ", ";
        json_string += "\"angular_velocity\": " + std::to_string(parameters.angular_velocity) + ", ";
        json_string += "\"action\": " + std::to_string(parameters.action) + ", ";
        json_string += "\"linear_velocity\": " + std::to_string(parameters.linear_velocity);
        json_string += "}";
        return json_string;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
