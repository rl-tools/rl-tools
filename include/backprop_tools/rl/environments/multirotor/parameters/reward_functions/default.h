#include "../../../../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_H

#include "../../multirotor.h"
#include "abs_exp.h"
#include "sq_exp.h"
#include "squared.h"
#include "absolute.h"
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE (0.334)
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    constexpr AbsExp<T> reward_263 = {
        10,
        1,
        10,
        10,
        0,
        0,
        0,
        0,
        BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE,
        1.0/2 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };
    template<typename T>
    constexpr AbsExp<T> reward_old_but_gold = {
            10, // scale
            1, // scale inner
            1, // position
            5, // orientation
            0.5, // linear velocity
            0.005, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> reward_old_but_gold_1 = {
            1, // scale
            1, // scale inner
            1, // position
            0, // orientation
            0.5, // linear velocity
            0.5, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            1 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> reward_old_but_gold_2 = {
            1, // scale
            0.1, // scale inner
            0, // position
            5, // orientation
            0.5, // linear velocity
            0.5, // angular velocity
            0.5, // linear acceleration
            0.05, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> reward_old_but_gold_3 = {
            1, // scale
            1, // scale inner
            10, // position
            0, // orientation
            0, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> reward_old_but_gold_4 = {
            10, // scale
            1, // scale inner
            4, // position
            5, // orientation
            0.5, // linear velocity
            0.005, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> position_only = {
            10, // scale
            1, // scale inner
            4, // position
            0, // orientation
            0, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };
    template<typename T>
    constexpr AbsExp<T> position_action_only = {
            10, // scale
            1, // scale inner
            4, // position
            0, // orientation
            0, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            2 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };
    template<typename T>
    constexpr AbsExp<T> position_action_only_2 = {
            10, // scale
            1, // scale inner
            4, // position
            0, // orientation
            0, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            0, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> position_action_only_3 = {
            10, // scale
            1, // scale inner
            2, // position
            0, // orientation
            0.1, // linear velocity
            0.02, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0.1 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> abs_exp_position_only = {
            1, // scale
            20, // scale inner
            1, // position
            0, // orientation
            0, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> abs_exp_position_orientation_lin_vel = {
            1, // scale
            1, // scale inner
            1, // position
            0.01, // orientation
            0.3, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr SqExp<T> sq_exp_position_action_only_2 = {
            0, // additive_constant
            1, // scale
            10, // scale inner
            1, // position
            0, // orientation
            0, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr SqExp<T> sq_exp_position_action_only_3 = {
            0, // additive_constant
            1, // scale
            5, // scale inner
            1, // position
            0, // orientation
            0, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            1/250.0*0.02, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T, typename TI>
    constexpr SqExpMultiModal<T, TI, 2> sq_exp_reward_mm = {
            SqExp<T>{
                    0, // additive_constant
                    1, // scale
                    5, // scale inner
                    2, // position
                    0, // orientation
                    0, // linear velocity
                    0, // angular velocity
                    0, // linear acceleration
                    0, // angular acceleration
                    BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
                    1 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
            },
            SqExp<T>{
                    0, // additive_constant
                    1, // scale
                    5, // scale inner
                    1, // position
                    0, // orientation
                    0, // linear velocity
                    0, // angular velocity
                    0, // linear acceleration
                    0, // angular acceleration
                    BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
                    0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
            }
    };

    template<typename T, typename TI>
    constexpr AbsExpMultiModal<T, TI, 3> reward_mm = {
        // + position
        AbsExp<T>{
            20, // scale
            1, // scale inner
            5, // position
            1, // orientation
            5, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            0.01, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
        },
        // + linear velocity
        AbsExp<T>{
            10, // scale
            1, // scale inner
            0, // position
            1, // orientation
            5, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            0.01, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
        },
        AbsExp<T>{
            1, // scale
            1, // scale inner
            0, // position
            1, // orientation
            0, // linear velocity
            0, // angular velocity
            0, // linear acceleration
            0.01, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
        }
    };

    template<typename T>
    constexpr AbsExp<T> reward_1 = {
            10, // scale
            0.1, // scale inner
            10, // position
            10, // orientation
            0, // linear velocity
            0.1, // angular velocity
            1, // linear acceleration
            0.5, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> reward_dr = {
        10,
        1, // scale inner
        1,
        5,
        0.5,
        0.005,
        0,
        0,
        BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE,
        1.0/2 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> reward_angular_velocity = {
        1,
        1, // scale inner
        0,
        0,
        0,
        0.01,
        0,
        0,
        BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE,
        1.0/2.0
    };

    template<typename T>
    constexpr Squared<T> reward_squared_1 = {
        false, // non-negative
        0.01, // scale
        300, // constant
        0, // termination penalty
        1, // position
        1, // orientation
        0.1, // linear_velocity
        0.01, // angular_velocity
        0.01, // linear_acceleration
        0.01, // angular_acceleration
        BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE
        0, // action
    };

    template<typename T>
    constexpr Squared<T> reward_squared_2 = {
            false, // non-negative
            0.001, // scale
            1, // constant
            0, // termination penalty
            10, // position
            10, // orientation
            0.1, // linear_velocity
            0.1, // angular_velocity
            0.1, // linear_acceleration
            0.001, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            20, // action
    };

    template<typename T>
    constexpr Squared<T> reward_squared_3 = {
            false, // non-negative
            0.1, // scale
            1, // constant
            0, // termination penalty
            10, // position
            10, // orientation
            0, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0, // action
    };
    template<typename T>
    constexpr Squared<T> reward_squared_4 = {
            false, // non-negative
            1, // scale
            0, // constant
            -100, // termination penalty
            1, // position
            0, // orientation
            0, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0, // action
    };
    template<typename T>
    constexpr Squared<T> reward_squared_5 = {
            false, // non-negative
            0.01, // scale
            0, // constant
            -10000, // termination penalty
            1, // position
            1, // orientation
            1, // linear_velocity
            0.01, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            1, // action
    };
    template<typename T>
    constexpr Squared<T> reward_squared_position_only = {
            false, // non-negative
            1, // scale
            2, // constant
            0, // termination penalty
            5, // position
            5, // orientation
            0.01, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE
            0.01, // action
    };
    template<typename T>
    constexpr Squared<T> reward_squared_position_only_torque = {
            false, // non-negative
            0.5, // scale
            2, // constant
            0, // termination penalty
            5, // position
            5, // orientation
            0.01, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE
            0.01, // action
    };
    template<typename T>
    constexpr Squared<T> reward_squared_position_only_torque_curriculum_target = {
            false, // non-negative
            0.5, // scale
            2, // constant
            0, // termination penalty
            40, // position
            5, // orientation
            1.00, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE
            1.00, // action
    };
    template<typename T>
    constexpr Squared<T> reward_squared_fast_learning = {
            false, // non-negative
            1, // scale
            0.2, // constant
            0, // termination penalty
            1, // position
            0, // orientation
            0, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE
            0.0, // action
    };
    template<typename T>
    constexpr Squared<T> reward_squared_fast_learning_negative = {
            false, // non-negative
            0.1, // scale
            0, // constant
            0, // termination penalty
            1, // position
            0, // orientation
            0, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE
            0.0, // action
    };
    template<typename T>
    constexpr Absolute<T> reward_absolute_fast_learning = {
            false, // non-negative
            1, // scale
            1, // constant
            0, // termination penalty
            1, // position
            0, // orientation
            0, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE
            0.0, // action
    };
    template<typename T>
    constexpr Absolute<T> reward_absolute_fast_learning_2 = {
            false, // non-negative
            1, // scale
            2, // constant
            0, // termination penalty
            1, // position
            0.1, // orientation
            0.1, // linear_velocity
            0.01, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE
            0.0, // action
    };
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
