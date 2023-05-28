#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_SQUARED_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_SQUARED_H

#include "../../multirotor.h"
#include <backprop_tools/utils/generic/typing.h>
#include <backprop_tools/utils/generic/vector_operations.h>

namespace backprop_tools::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    struct Squared{
        bool non_negative;
        T scale;
        T constant;
        T position;
        T orientation;
        T linear_velocity;
        T angular_velocity;
        T linear_acceleration;
        T angular_acceleration;
        T action_baseline;
        T action;
    };
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::Squared<T>& params, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action,  const typename rl::environments::Multirotor<SPEC>::State& next_state) {
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
//        T q_sq = state.state[3] * state.state[3] + state.state[4] * state.state[4] + state.state[5] * state.state[5] + state.state[6] * state.state[6];
        T orientation_cost = 1 - state.state[3] * state.state[3]; //math::abs(typename DEVICE::SPEC::MATH(), 2 * math::acos(typename DEVICE::SPEC::MATH(), quaternion_w));
        T position_cost = utils::vector_operations::norm<DEVICE, T, 3>(state.state);
        position_cost *= position_cost;
        T linear_vel_cost = utils::vector_operations::norm<DEVICE, T, 3>(&state.state[3+4]);
        linear_vel_cost *= linear_vel_cost;
        T angular_vel_cost = utils::vector_operations::norm<DEVICE, T, 3>(&state.state[3+4+3]);
        angular_vel_cost *= angular_vel_cost;
        T linear_acc[3];
        utils::vector_operations::sub<DEVICE, T, 3>(&next_state.state[7], &state.state[7], linear_acc);
        T linear_acc_cost = utils::vector_operations::norm<DEVICE, T, 3>(linear_acc) / env.parameters.integration.dt;
        linear_acc_cost *= linear_acc_cost;
        T angular_acc[3];
        utils::vector_operations::sub<DEVICE, T, 3>(&next_state.state[7+3], &state.state[7+3], angular_acc);
        T angular_acc_cost = utils::vector_operations::norm<DEVICE, T, 3>(angular_acc) / env.parameters.integration.dt;
        angular_acc_cost *= angular_acc_cost;

        T action_diff[ACTION_DIM];
//        utils::vector_operations::sub<DEVICE, T, ACTION_DIM>(action, utils::vector_operations::mean<DEVICE, T, ACTION_DIM>(action), action_diff);
        for(TI i = 0; i < ACTION_DIM; i++){
            action_diff[i] = get(action, 0, i) - params.action_baseline;
        }
//        utils::vector_operations::sub<DEVICE, T, ACTION_DIM>(action, params.action_baseline, action_diff);
        T action_cost = utils::vector_operations::norm<DEVICE, T, ACTION_DIM>(action_diff);
        action_cost *= action_cost;
        T weighted_abs_cost = params.position * position_cost + params.orientation * orientation_cost + params.linear_velocity * linear_vel_cost + params.angular_velocity * angular_vel_cost + params.linear_acceleration * linear_acc_cost + params.angular_acceleration * angular_acc_cost + params.action * action_cost;
        constexpr TI cadence = 991;
        {
            add_scalar(device, device.logger, "reward/orientation_cost", orientation_cost, cadence);
            add_scalar(device, device.logger, "reward/position_cost", position_cost, cadence);
            add_scalar(device, device.logger, "reward/linear_vel_cost", linear_vel_cost, cadence);
            add_scalar(device, device.logger, "reward/angular_vel_cost", angular_vel_cost, cadence);
            add_scalar(device, device.logger, "reward/linear_acc_cost", linear_acc_cost, cadence);
            add_scalar(device, device.logger, "reward/angular_acc_cost", angular_acc_cost, cadence);
            add_scalar(device, device.logger, "reward/action_cost", action_cost, cadence);
            add_scalar(device, device.logger, "reward/pre_exp", -weighted_abs_cost, cadence);

            add_scalar(device, device.logger, "reward_weighted/orientation_cost", params.orientation * orientation_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/position_cost", params.position * position_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/linear_vel_cost", params.linear_velocity * linear_vel_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/angular_vel_cost", params.angular_velocity * angular_vel_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/linear_acc_cost", params.linear_acceleration * linear_acc_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/angular_acc_cost", params.angular_acceleration * angular_acc_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/action_cost", params.action * action_cost, cadence);
            // log share of the weighted abs cost
            add_scalar(device, device.logger, "reward_share/orientation", params.orientation * orientation_cost / weighted_abs_cost, cadence);
            add_scalar(device, device.logger, "reward_share/position", params.position * position_cost / weighted_abs_cost, cadence);
            add_scalar(device, device.logger, "reward_share/linear_vel", params.linear_velocity * linear_vel_cost / weighted_abs_cost, cadence);
            add_scalar(device, device.logger, "reward_share/angular_vel", params.angular_velocity * angular_vel_cost / weighted_abs_cost, cadence);
            add_scalar(device, device.logger, "reward_share/linear_acc", params.linear_acceleration * linear_acc_cost / weighted_abs_cost, cadence);
            add_scalar(device, device.logger, "reward_share/angular_acc", params.angular_acceleration * angular_acc_cost / weighted_abs_cost, cadence);
            add_scalar(device, device.logger, "reward_share/action", params.action * action_cost / weighted_abs_cost, cadence);
        }
        add_scalar(device, device.logger, "reward/weighted_abs_cost", weighted_abs_cost, cadence);
        T scaled_weighted_abs_cost = params.scale * weighted_abs_cost;
        add_scalar(device, device.logger, "reward/scaled_weighted_abs_cost", scaled_weighted_abs_cost, cadence);

        T reward = -scaled_weighted_abs_cost + params.constant;
        return reward > 0 || !params.non_negative ? reward : 0;
    }
}

#endif
