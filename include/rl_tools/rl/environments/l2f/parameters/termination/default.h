
#include "../../multirotor.h"

#include <rl_tools/math/operations_generic.h>

namespace rl_tools::rl::environments::multirotor::parameters::termination{
    template<typename SPEC>
    constexpr typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::MDP::Termination fast_learning = {
        true,           // enable
        2.0,            // position
        10,         // linear velocity
        10, // angular velocity
        1, // position integral
        50, // orientation integral
    };
}