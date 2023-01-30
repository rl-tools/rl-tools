#ifndef LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_H
#define LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_H

#include "../../multirotor.h"
#include "abs_exp.h"
namespace layer_in_c::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    constexpr AbsExp<T> reward_263 = {
        10,
        10,
        10,
        0,
        0,
        -1,
        1.0/2 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> reward_dr = {
        10,
        1,
        5,
        0.5,
        0.005,
        -1,
        1.0/2 // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
    };

    template<typename T>
    constexpr AbsExp<T> reward_angular_velocity = {
        1,
        0,
        0,
        0,
        0.01,
        0,
        1.0/2.0
    };
}
#endif
