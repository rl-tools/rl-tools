#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_ABS_EXP_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_ABS_EXP_H

#include "../../multirotor.h"
#include <backprop_tools/utils/generic/typing.h>
#include <backprop_tools/utils/generic/vector_operations.h>

namespace backprop_tools::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    struct AbsExp{
        T scale;
        T scale_inner;
        T position;
        T orientation;
        T linear_velocity;
        T angular_velocity;
        T linear_acceleration;
        T angular_acceleration;
        T action_baseline;
        T action;
    };
    template<typename DEVICE, typename SPEC, typename T, typename ACTION_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::AbsExp<T>& params, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Multirotor<SPEC>::State& next_state) {
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == ACTION_DIM);

//        printf("state reward: %f %f %f %f %f %f %f %f %f %f %f %f %f\n", state.state[0], state.state[1], state.state[2], state.state[3], state.state[4], state.state[5], state.state[6], state.state[7], state.state[8], state.state[9], state.state[10], state.state[11], state.state[12]);

        T quaternion_w = state.state[3];
        T orientation_cost = math::abs(typename DEVICE::SPEC::MATH(), 2 * math::acos(typename DEVICE::SPEC::MATH(), quaternion_w));
        T position_cost = utils::vector_operations::norm<DEVICE, T, 3>(state.state);
        T linear_vel_cost = utils::vector_operations::norm<DEVICE, T, 3>(&state.state[3+4]);
        T angular_vel_cost = utils::vector_operations::norm<DEVICE, T, 3>(&state.state[3+4+3]);
        T linear_acc[3];
        T angular_acc[3];
        utils::vector_operations::sub<DEVICE, T, 3>(&next_state.state[7], &state.state[7], linear_acc);
        T linear_acc_cost = utils::vector_operations::norm<DEVICE, T, 3>(linear_acc) / env.parameters.integration.dt;
        utils::vector_operations::sub<DEVICE, T, 3>(&next_state.state[7+3], &state.state[7+3], angular_acc);
        T angular_acc_cost = utils::vector_operations::norm<DEVICE, T, 3>(angular_acc) / env.parameters.integration.dt;

        T action_diff[ACTION_DIM];
//        utils::vector_operations::sub<DEVICE, T, ACTION_DIM>(action, utils::vector_operations::mean<DEVICE, T, ACTION_DIM>(action), action_diff);
        for(TI i = 0; i < ACTION_DIM; i++){
            action_diff[i] = get(action, 0, i) - params.action_baseline;
        }
//        utils::vector_operations::sub<DEVICE, T, ACTION_DIM>(action, params.action_baseline, action_diff);
        T action_cost = utils::vector_operations::norm<DEVICE, T, ACTION_DIM>(action_diff);
        T weighted_abs_cost = params.position * position_cost + params.orientation * orientation_cost + params.linear_velocity * linear_vel_cost + params.angular_velocity * angular_vel_cost + params.linear_acceleration * linear_acc_cost + params.angular_acceleration * angular_acc_cost + params.action * action_cost;
        T r = math::exp(typename DEVICE::SPEC::MATH(), -params.scale_inner*weighted_abs_cost);
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

        return r * params.scale;
    }
}

#endif
