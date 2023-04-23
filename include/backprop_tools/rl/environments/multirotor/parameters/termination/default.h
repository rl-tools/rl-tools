#include "../../multirotor.h"

#include <backprop_tools/math/operations_generic.h>

namespace backprop_tools::rl::environments::multirotor::parameters::termination{
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename backprop_tools::rl::environments::multirotor::Parameters<T, TI, 4, REWARD_FUNCTION>::MDP::Termination classic = {
        true,           // enable
//        2.0944,            // angle
        0.6,            // position
        1000,         // linear velocity
        1000 // angular velocity
    };
}
