#ifndef LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_DEFAULT_H
#define LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_DEFAULT_H

#include "../../multirotor.h"

namespace layer_in_c::rl::environments::multirotor::parameters::init{
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename Parameters<T, TI, ACTION_DIM, REWARD_FUNCTION>::MDP::Initialization all_around = {
            0.1,   // guidance
            0.3, // position
            1,   // orientation
            1,   // linear velocity
            10   // angular velocity
    };
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename Parameters<T, TI, ACTION_DIM, REWARD_FUNCTION>::MDP::Initialization simple = {
            0,   // guidance
            0,   // position
            0,   // orientation
            0,   // linear velocity
            0    // angular velocity
    };
}

#endif
