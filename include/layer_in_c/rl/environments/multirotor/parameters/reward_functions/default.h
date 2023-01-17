

#include "../../multirotor.h"
#include "abs_exp.h"
namespace layer_in_c::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    AbsExp<T> reward_1 = {
            10,
            10,
            1,
            0,
            1
    };
}