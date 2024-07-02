
#include "../../multirotor.h"

#include <rl_tools/math/operations_generic.h>

namespace rl_tools::rl::environments::multirotor::parameters::termination{
    template<typename SPEC>
    constexpr typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::MDP::Termination fast_learning = {
        true,           // enable
        0.6,            // position
        1000,         // linear velocity
        1000, // angular velocity
        10000, // position integral
        50000, // orientation integral
    };
}