#include "../../../../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_ABSOLUTE_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_ABSOLUTE_H

#include "../../multirotor.h"
#include "../../../../../utils/generic/typing.h"
#include "../../../../../utils/generic/vector_operations.h"

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    struct Absolute{
        bool non_negative;
        T scale;
        T constant;
        T termination_penalty;
        T position;
        T orientation;
        T linear_velocity;
        T angular_velocity;
        T linear_acceleration;
        T angular_acceleration;
        T action_baseline;
        T action;
    };
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::Absolute<T>& params, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action,  const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        T orientation_cost = 1 - state.orientation[0] * state.orientation[0]; //math::abs(device.math, 2 * math::acos(device.math, quaternion_w));
        T position_cost = math::abs(device.math, state.position[0]) + math::abs(device.math, state.position[1]) + math::abs(device.math, state.position[2]);
        T linear_vel_cost = math::abs(device.math, state.linear_velocity[0]) + math::abs(device.math, state.linear_velocity[1]) + math::abs(device.math, state.linear_velocity[2]);
        T angular_vel_cost = math::abs(device.math, state.angular_velocity[0]) + math::abs(device.math, state.angular_velocity[1]) + math::abs(device.math, state.angular_velocity[2]);
        T linear_acc[3];
        T angular_acc[3];
        utils::vector_operations::sub<DEVICE, T, 3>(next_state.linear_velocity, state.linear_velocity, linear_acc);
        T linear_acc_cost = (math::abs(device.math, linear_acc[0]) + math::abs(device.math, linear_acc[1]) + math::abs(device.math, linear_acc[2])) / (env.parameters.integration.dt * env.parameters.integration.dt);
        utils::vector_operations::sub<DEVICE, T, 3>(next_state.angular_velocity, state.angular_velocity, angular_acc);
        T angular_acc_cost = (math::abs(device.math, angular_acc[0]) + math::abs(device.math, angular_acc[1]) + math::abs(device.math, angular_acc[2])) / (env.parameters.integration.dt * env.parameters.integration.dt);

        T action_diff[ACTION_DIM];
//        utils::vector_operations::sub<DEVICE, T, ACTION_DIM>(action, utils::vector_operations::mean<DEVICE, T, ACTION_DIM>(action), action_diff);
        for(TI i = 0; i < ACTION_DIM; i++){
            action_diff[i] = get(action, 0, i) - params.action_baseline;
        }
//        utils::vector_operations::sub<DEVICE, T, ACTION_DIM>(action, params.action_baseline, action_diff);
        T action_cost = utils::vector_operations::norm<DEVICE, T, ACTION_DIM>(action_diff);
        action_cost *= action_cost;
        T weighted_cost = params.position * position_cost + params.orientation * orientation_cost + params.linear_velocity * linear_vel_cost + params.angular_velocity * angular_vel_cost + params.linear_acceleration * linear_acc_cost + params.angular_acceleration * angular_acc_cost + params.action * action_cost;
        bool terminated_flag = terminated(device, env, next_state, rng);
        T scaled_weighted_cost = params.scale * weighted_cost;

        T r;

        if(terminated_flag){
            r = params.termination_penalty;
        }
        else{
            r = -scaled_weighted_cost + params.constant;
            r = (r > 0 || !params.non_negative) ? r : 0;
        }

        constexpr TI cadence = 991;
        {
            add_scalar(device, device.logger, "reward/orientation_cost", orientation_cost, cadence);
            add_scalar(device, device.logger, "reward/position_cost", position_cost, cadence);
            add_scalar(device, device.logger, "reward/linear_vel_cost", linear_vel_cost, cadence);
            add_scalar(device, device.logger, "reward/angular_vel_cost", angular_vel_cost, cadence);
            add_scalar(device, device.logger, "reward/linear_acc_cost", linear_acc_cost, cadence);
            add_scalar(device, device.logger, "reward/angular_acc_cost", angular_acc_cost, cadence);
            add_scalar(device, device.logger, "reward/action_cost", action_cost, cadence);
            add_scalar(device, device.logger, "reward/pre_exp", -weighted_cost, cadence);

            add_scalar(device, device.logger, "reward_weighted/orientation_cost", params.orientation * orientation_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/position_cost", params.position * position_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/linear_vel_cost", params.linear_velocity * linear_vel_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/angular_vel_cost", params.angular_velocity * angular_vel_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/linear_acc_cost", params.linear_acceleration * linear_acc_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/angular_acc_cost", params.angular_acceleration * angular_acc_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/action_cost", params.action * action_cost, cadence);
            // log share of the weighted abs cost
            add_scalar(device, device.logger, "reward_share/orientation", params.orientation * orientation_cost / weighted_cost, cadence);
            add_scalar(device, device.logger, "reward_share/position", params.position * position_cost / weighted_cost, cadence);
            add_scalar(device, device.logger, "reward_share/linear_vel", params.linear_velocity * linear_vel_cost / weighted_cost, cadence);
            add_scalar(device, device.logger, "reward_share/angular_vel", params.angular_velocity * angular_vel_cost / weighted_cost, cadence);
            add_scalar(device, device.logger, "reward_share/linear_acc", params.linear_acceleration * linear_acc_cost / weighted_cost, cadence);
            add_scalar(device, device.logger, "reward_share/angular_acc", params.angular_acceleration * angular_acc_cost / weighted_cost, cadence);
            add_scalar(device, device.logger, "reward_share/action", params.action * action_cost / weighted_cost, cadence);
            add_scalar(device, device.logger, "reward_share/const", r/params.constant, cadence);

            add_scalar(device, device.logger, "reward/weighted_cost", weighted_cost, cadence);
            add_scalar(device, device.logger, "reward/scaled_weighted_cost", scaled_weighted_cost, cadence);
            add_scalar(device, device.logger, "reward/reward", r, cadence);
            add_scalar(device, device.logger, "reward/reward_zero", r == 0, cadence);
        }

        return r;
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif
