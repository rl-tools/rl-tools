#include "../../multirotor.h"

#include <layer_in_c/math/operations_generic.h>

namespace layer_in_c::rl::environments::multirotor::parameters::termination{
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename layer_in_c::rl::environments::multirotor::Parameters<T, TI, 4, REWARD_FUNCTION>::MDP::Termination classic = {
        true,           // enable
//        2.0944,            // angle
        0.6,            // position
        1000,         // linear velocity
        1000 // angular velocity
    };
}
