#include "../../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_REWARD_FUNCTIONS_SQUARED_SQUARED_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_REWARD_FUNCTIONS_SQUARED_SQUARED_H

#include <rl_tools/utils/generic/typing.h>
#include <rl_tools/utils/generic/vector_operations.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters::reward_functions{
    template<typename T, unsigned T_ACTION_DIM=4>
    struct Squared{
        static constexpr unsigned ACTION_DIM = T_ACTION_DIM;
        bool non_negative;
        T scale;
        T constant;
        T termination_penalty;
        T position;
        T position_clip;
        T orientation;
        T linear_velocity;
        T angular_velocity;
        T linear_acceleration;
        T angular_acceleration;
        T action[ACTION_DIM];
        T d_action[ACTION_DIM];
        T position_error_integral;
        struct Components{
            T orientation_cost;
            T position_cost;
            T linear_vel_cost;
            T angular_vel_cost;
            T linear_acc_cost;
            T angular_acc_cost;
            T action_cost;
            T weighted_action_cost;
            T action_costs[ACTION_DIM];
            T weighted_action_costs[ACTION_DIM];
            T d_action_cost;
            T weighted_d_action_cost;
            T d_action_costs[ACTION_DIM];
            T weighted_d_action_costs[ACTION_DIM];
            T position_error_integral_cost;
            T weighted_cost;
            T scaled_weighted_cost;
            T reward;
        };
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
