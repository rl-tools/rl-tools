#include "../../../../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_SQ_EXP_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_SQ_EXP_H

#include "../../multirotor.h"
#include "../../../../../utils/generic/typing.h"
#include "../../../../../utils/generic/vector_operations.h"

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    struct SqExp{
        T additive_constant;
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
    template<typename T, typename TI, TI T_N_MODES>
    struct SqExpMultiModal{
        static constexpr TI N_MODES = T_N_MODES;
        SqExp<T> modes[N_MODES];
    };
    template<typename DEVICE, typename SPEC, typename T, typename T_STATE, typename TI_STATE, typename LATENT_STATE, typename ACTION_SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::SqExp<T>& params, const rl::environments::multirotor::StateBase<T_STATE, TI_STATE>& state, const Matrix<ACTION_SPEC>& action, const rl::environments::multirotor::StateBase<T_STATE, TI_STATE>& next_state, RNG& rng, bool log_components = true) {
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == ACTION_DIM);

//        printf("state reward: %f %f %f %f %f %f %f %f %f %f %f %f %f\n", state.state[0], state.state[1], state.state[2], state.state[3], state.state[4], state.state[5], state.state[6], state.state[7], state.state[8], state.state[9], state.state[10], state.state[11], state.state[12]);

        T orientation_cost = 1 - state.orientation[0] * state.orientation[0]; //math::sq(device.math, 2 * math::acos(device.math, quaternion_w));
//        T orientation_cost = math::sq(device.math, 2 * math::acos(device.math, quaternion_w));
        T position_cost = state.position[0] * state.position[0] + state.position[1] * state.position[1] + state.position[2] * state.position[2];
        T linear_vel_cost = state.linear_velocity[0] * state.linear_velocity[0] + state.linear_velocity[1] * state.linear_velocity[1] + state.linear_velocity[2] * state.linear_velocity[2];
        T angular_vel_cost = state.angular_velocity[0] * state.angular_velocity[0] + state.angular_velocity[1] * state.angular_velocity[1] + state.angular_velocity[2] * state.angular_velocity[2];
        T linear_acc[3];
        T angular_acc[3];
        utils::vector_operations::sub<DEVICE, T, 3>(next_state.linear_velocity, state.linear_velocity, linear_acc);
        T linear_acc_cost = (linear_acc[0] * linear_acc[0] + linear_acc[1] * linear_acc[1] + linear_acc[2] * linear_acc[2]) / (env.parameters.integration.dt * env.parameters.integration.dt);
        utils::vector_operations::sub<DEVICE, T, 3>(next_state.angular_velocity, state.angular_velocity, angular_acc);
        T angular_acc_cost = (angular_acc[0] * angular_acc[0] + angular_acc[1] * angular_acc[1] + angular_acc[2] * angular_acc[2]) / (env.parameters.integration.dt * env.parameters.integration.dt);

        T action_diff[ACTION_DIM];
//        utils::vector_operations::sub<DEVICE, T, ACTION_DIM>(action, utils::vector_operations::mean<DEVICE, T, ACTION_DIM>(action), action_diff);
        for(TI i = 0; i < ACTION_DIM; i++){
            action_diff[i] = get(action, 0, i) - params.action_baseline;
        }
//        utils::vector_operations::sub<DEVICE, T, ACTION_DIM>(action, params.action_baseline, action_diff);
        T action_cost = utils::vector_operations::norm<DEVICE, T, ACTION_DIM>(action_diff);
        T weighted_sq_cost = params.position * position_cost + params.orientation * orientation_cost + params.linear_velocity * linear_vel_cost + params.angular_velocity * angular_vel_cost + params.linear_acceleration * linear_acc_cost + params.angular_acceleration * angular_acc_cost + params.action * action_cost;
        T sq_exp = math::exp(device.math, -params.scale_inner*weighted_sq_cost);
        T r = sq_exp * params.scale + params.additive_constant;
        constexpr TI cadence = 991;
        if(log_components){
            add_scalar(device, device.logger, "reward/orientation_cost", orientation_cost, cadence);
            add_scalar(device, device.logger, "reward/position_cost", position_cost, cadence);
            add_scalar(device, device.logger, "reward/linear_vel_cost", linear_vel_cost, cadence);
            add_scalar(device, device.logger, "reward/angular_vel_cost", angular_vel_cost, cadence);
            add_scalar(device, device.logger, "reward/linear_acc_cost", linear_acc_cost, cadence);
            add_scalar(device, device.logger, "reward/angular_acc_cost", angular_acc_cost, cadence);
            add_scalar(device, device.logger, "reward/action_cost", action_cost, cadence);
            add_scalar(device, device.logger, "reward/pre_exp", -weighted_sq_cost, cadence);

            add_scalar(device, device.logger, "reward_weighted/orientation_cost", params.orientation * orientation_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/position_cost", params.position * position_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/linear_vel_cost", params.linear_velocity * linear_vel_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/angular_vel_cost", params.angular_velocity * angular_vel_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/linear_acc_cost", params.linear_acceleration * linear_acc_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/angular_acc_cost", params.angular_acceleration * angular_acc_cost, cadence);
            add_scalar(device, device.logger, "reward_weighted/action_cost", params.action * action_cost, cadence);
            // log share of the weighted sq cost
            add_scalar(device, device.logger, "reward_share/orientation", params.orientation * orientation_cost / weighted_sq_cost, cadence);
            add_scalar(device, device.logger, "reward_share/position", params.position * position_cost / weighted_sq_cost, cadence);
            add_scalar(device, device.logger, "reward_share/linear_vel", params.linear_velocity * linear_vel_cost / weighted_sq_cost, cadence);
            add_scalar(device, device.logger, "reward_share/angular_vel", params.angular_velocity * angular_vel_cost / weighted_sq_cost, cadence);
            add_scalar(device, device.logger, "reward_share/linear_acc", params.linear_acceleration * linear_acc_cost / weighted_sq_cost, cadence);
            add_scalar(device, device.logger, "reward_share/angular_acc", params.angular_acceleration * angular_acc_cost / weighted_sq_cost, cadence);
            add_scalar(device, device.logger, "reward_share/action", params.action * action_cost / weighted_sq_cost, cadence);

            add_scalar(device, device.logger, "reward/weighted_sq_cost", weighted_sq_cost, cadence);
            add_scalar(device, device.logger, "reward/sq_exp", sq_exp, cadence);
            add_scalar(device, device.logger, "reward/reward", r, cadence);
        }

        return r;
    }
    template<typename DEVICE, typename SPEC, typename T, typename ACTION_SPEC, typename TI, TI N_MODES, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::SqExpMultiModal<T, TI, N_MODES>& params, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        T acc = 0;
        for(TI mode_i=0; mode_i < N_MODES; mode_i++){
            T output = reward(device, env, params.modes[mode_i], state, action, next_state, rng, mode_i==0);
            if(mode_i == 0){
                add_scalar(device, device.logger, "reward_mode/0", output, 991);
            }
            else{
                if(mode_i == 1){
                    add_scalar(device, device.logger, "reward_mode/1", output, 991);
                }
                else{
                    if(mode_i == 2){
                        add_scalar(device, device.logger, "reward_mode/2", output, 991);
                    }
                }
            }
            acc += output;
        }
        return acc;
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif
