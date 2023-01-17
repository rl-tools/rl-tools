

#include "../../multirotor.h"
namespace layer_in_c::rl::environments::multirotor::parameters::reward_functions{
    template<typename T, typename TI, TI ACTION_DIM>
    typename Parameters<T, TI, ACTION_DIM>::MDP::Reward reward_1 = {
            10,
            10,
            1,
            0,
            1
    };

    template<typename T, typename TI, TI ACTION_DIM>
    typename Parameters<T, TI, ACTION_DIM>::Initialization default_init_parameters = {
            2,
            1,
            0.5 * math::PI<T> * 2
    };
    template<typename T, typename TI, TI ACTION_DIM>
    typename Parameters<T, TI, ACTION_DIM>::Initialization simple_init_parameters = {
            0,
            0,
            0
    };

}