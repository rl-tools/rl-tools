#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_H

#include "../../multirotor.h"
#include "abs_exp.h"
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE (-0.26)
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

    template<typename T, typename TI>
    constexpr AbsExpMultiModal<T, TI, 2> reward_mm = {
        AbsExp<T>{
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
        },
        AbsExp<T>{
            10, // scale
            1, // scale inner
            10, // position
            2, // orientation
            0.5, // linear velocity
            0.5, // angular velocity
            0, // linear acceleration
            0, // angular acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            1 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
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
    constexpr Squared<T> reward_squraed_1 = {
        false, // non-negative
        0.01, // scale
        300, // constant
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
    constexpr Squared<T> reward_squraed_2 = {
            false, // non-negative
            0.001, // scale
            1, // constant
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
    constexpr Squared<T> reward_squraed_3 = {
            false, // non-negative
            0.1, // scale
            1, // constant
            10, // position
            10, // orientation
            0, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_ACTION_BASELINE, // action baseline
            0, // action
    };
}
#endif
