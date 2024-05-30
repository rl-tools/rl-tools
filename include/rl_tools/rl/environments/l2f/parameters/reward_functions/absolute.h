#ifndef LEARNING_TO_FLY_IN_SECONDS_SIMULATOR_PARAMETERS_REWARD_FUNCTIONS_ABSOLUTE_H
#define LEARNING_TO_FLY_IN_SECONDS_SIMULATOR_PARAMETERS_REWARD_FUNCTIONS_ABSOLUTE_H

#include "../../multirotor.h"
#include <rl_tools/utils/generic/typing.h>
#include <rl_tools/utils/generic/vector_operations.h>

namespace rl_tools::rl::environments::multirotor::parameters::reward_functions{
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
        T action;
        struct Components{
            T orientation_cost;
            T position_cost;
            T linear_vel_cost;
            T angular_vel_cost;
            T linear_acc_cost;
            T angular_acc_cost;
            T action_cost;
            T weighted_cost;
            T scaled_weighted_cost;
            T reward;
        };
    };
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename rl::environments::multirotor::parameters::reward_functions::Absolute<T>::Components reward_components(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::Absolute<T>& params, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action,  const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng){
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        typename rl::environments::multirotor::parameters::reward_functions::Absolute<T>::Components components;
//        components.orientation_cost = 1 - state.orientation[0] * state.orientation[0]; //math::abs(device.math, 2 * math::acos(device.math, quaternion_w));
        components.orientation_cost = math::abs(device.math, state.orientation[3]);
        components.position_cost = 0;
        components.position_cost += math::abs(device.math, state.position[0]);
        components.position_cost += math::abs(device.math, state.position[1]);
        components.position_cost += math::abs(device.math, state.position[2]);
        components.linear_vel_cost = 0;
        components.linear_vel_cost += math::abs(device.math, state.linear_velocity[0]);
        components.linear_vel_cost += math::abs(device.math, state.linear_velocity[1]);
        components.linear_vel_cost += math::abs(device.math, state.linear_velocity[2]);
        components.angular_vel_cost = 0;
        components.angular_vel_cost += math::abs(device.math, state.angular_velocity[0]);
        components.angular_vel_cost += math::abs(device.math, state.angular_velocity[1]);
        components.angular_vel_cost += math::abs(device.math, state.angular_velocity[2]);
        T linear_acc[3];
        T angular_acc[3];
        utils::vector_operations::sub<DEVICE, T, 3>(next_state.linear_velocity, state.linear_velocity, linear_acc);
        components.linear_acc_cost = 0;
        components.linear_acc_cost += math::abs(device.math, linear_acc[0]);
        components.linear_acc_cost += math::abs(device.math, linear_acc[1]);
        components.linear_acc_cost += math::abs(device.math, linear_acc[2]);
        components.linear_acc_cost /= env.parameters.integration.dt;

        utils::vector_operations::sub<DEVICE, T, 3>(next_state.angular_velocity, state.angular_velocity, angular_acc);
        components.angular_acc_cost = 0;
        components.angular_acc_cost += math::abs(device.math, angular_acc[0]);
        components.angular_acc_cost += math::abs(device.math, angular_acc[1]);
        components.angular_acc_cost += math::abs(device.math, angular_acc[2]);
        components.angular_acc_cost /= env.parameters.integration.dt;

        components.action_cost = 0;
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T action_value = (get(action, 0, action_i) + T(1))/2;
            T action_diff = math::abs(device.math, action_value - env.parameters.dynamics.hovering_throttle_relative);
            components.action_cost += action_diff;
        }
        components.weighted_cost = params.position * components.position_cost + params.orientation * components.orientation_cost + params.linear_velocity * components.linear_vel_cost + params.angular_velocity * components.angular_vel_cost + params.linear_acceleration * components.linear_acc_cost + params.angular_acceleration * components.angular_acc_cost + params.action * components.action_cost;
        bool terminated_flag = terminated(device, env, next_state, rng);
        components.scaled_weighted_cost = params.scale * components.weighted_cost;

        if(terminated_flag){
            components.reward = params.termination_penalty;
        }
        else{
            components.reward = -components.scaled_weighted_cost + params.constant;
            components.reward = (components.reward > 0 || !params.non_negative) ? components.reward : 0;
        }

        return components;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void log_reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::Absolute<T>& params, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action,  const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        constexpr typename SPEC::TI cadence = 1;
        auto components = reward_components(device, env, params, state, action, next_state, rng);
        add_scalar(device, device.logger, "reward/orientation_cost", components.orientation_cost, cadence);
        add_scalar(device, device.logger, "reward/position_cost",    components.position_cost, cadence);
        add_scalar(device, device.logger, "reward/linear_vel_cost",  components.linear_vel_cost, cadence);
        add_scalar(device, device.logger, "reward/angular_vel_cost", components.angular_vel_cost, cadence);
        add_scalar(device, device.logger, "reward/linear_acc_cost",  components.linear_acc_cost, cadence);
        add_scalar(device, device.logger, "reward/angular_acc_cost", components.angular_acc_cost, cadence);
        add_scalar(device, device.logger, "reward/action_cost",      components.action_cost, cadence);
        add_scalar(device, device.logger, "reward/pre_exp",         -components.weighted_cost, cadence);

        add_scalar(device, device.logger, "reward_weighted/orientation_cost", params.orientation          * components.orientation_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/position_cost",    params.position             * components.position_cost,    cadence);
        add_scalar(device, device.logger, "reward_weighted/linear_vel_cost",  params.linear_velocity      * components.linear_vel_cost,  cadence);
        add_scalar(device, device.logger, "reward_weighted/angular_vel_cost", params.angular_velocity     * components.angular_vel_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/linear_acc_cost" , params.linear_acceleration  * components.linear_acc_cost,  cadence);
        add_scalar(device, device.logger, "reward_weighted/angular_acc_cost", params.angular_acceleration * components.angular_acc_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/action_cost",      params.action               * components.action_cost,      cadence);
        // log share of the weighted abs cost
        add_scalar(device, device.logger, "reward_share/orientation", params.orientation          * components.orientation_cost / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/position",    params.position             * components.position_cost    / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/linear_vel",  params.linear_velocity      * components.linear_vel_cost  / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/angular_vel", params.angular_velocity     * components.angular_vel_cost / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/linear_acc",  params.linear_acceleration  * components.linear_acc_cost  / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/angular_acc", params.angular_acceleration * components.angular_acc_cost / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/action",      params.action               * components.action_cost      / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/const",       components.reward/params.constant, cadence);

        add_scalar(device, device.logger, "reward/weighted_cost",        components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward/scaled_weighted_cost", components.scaled_weighted_cost, cadence);
        add_scalar(device, device.logger, "reward/reward",               components.reward, cadence);
        add_scalar(device, device.logger, "reward/reward_zero",          components.reward == 0, cadence);
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::Absolute<T>& params, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action,  const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng){
        auto components = reward_components(device, env, params, state, action, next_state, rng);
        return components.reward;
    }
    template<typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto name(DEVICE& device, const rl::environments::multirotor::parameters::reward_functions::Absolute<T>& params){
        return "absolute";
    }
}

#endif
