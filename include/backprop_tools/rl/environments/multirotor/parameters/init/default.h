#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_DEFAULT_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_DEFAULT_H

#include "../../multirotor.h"

namespace backprop_tools::rl::environments::multirotor::parameters::init{
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename ParametersBase<T, TI, ACTION_DIM, REWARD_FUNCTION>::MDP::Initialization all_around = {
            0.1, // guidance
            0.3, // position
            1,   // orientation
            1,   // linear velocity
            10,  // angular velocity
            true,// relative rpm
            -1,  // min rpm
            +1,  // max rpm
    };
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename ParametersBase<T, TI, ACTION_DIM, REWARD_FUNCTION>::MDP::Initialization orientation_all_around = {
            0.1, // guidance
            0,   // position
            3.14,   // orientation
            0,   // linear velocity
            0,   // angular velocity
            true,// relative rpm
            0,  // min rpm
            0,  // max rpm
    };
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename ParametersBase<T, TI, ACTION_DIM, REWARD_FUNCTION>::MDP::Initialization orientation_small_angle = {
            0.1, // guidance
            0,   // position
            10.0/180.0*3.14,   // orientation
            0,   // linear velocity
            0,   // angular velocity
            true,// relative rpm
            0,  // min rpm
            0,  // max rpm
    };
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename ParametersBase<T, TI, ACTION_DIM, REWARD_FUNCTION>::MDP::Initialization orientation_big_angle = {
            0.1, // guidance
            0,   // position
            20.0/180.0 * 3.14,   // orientation
            0,   // linear velocity
            0,   // angular velocity
            true,// relative rpm
            0,  // min rpm
            0,  // max rpm
    };
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename ParametersBase<T, TI, ACTION_DIM, REWARD_FUNCTION>::MDP::Initialization orientation_bigger_angle = {
            0.1, // guidance
            0,   // position
            45.0/180.0 * 3.14,   // orientation
            0,   // linear velocity
            0,   // angular velocity
            true,// relative rpm
            0,  // min rpm
            0,  // max rpm
    };
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename ParametersBase<T, TI, ACTION_DIM, REWARD_FUNCTION>::MDP::Initialization all_around_simplified = {
            0.1, // guidance
            0.3, // position
            0,   // orientation
            1,   // linear velocity
            10,  // angular velocity
            true,// relative rpm
            -1,  // min rpm
            +1,  // max rpm
    };
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename ParametersBase<T, TI, ACTION_DIM, REWARD_FUNCTION>::MDP::Initialization simple = {
            0,   // guidance
            0,   // position
            0,   // orientation
            0,   // linear velocity
            0,   // angular velocity
            true,// relative rpm
            0,   // min rpm
            0,   // max rpm
    };
}

#endif
