#include "../../multirotor.h"
#include "abs_exp.h"
#include "sq_exp.h"
#include "squared.h"
#include "absolute.h"
namespace rl_tools::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    constexpr Squared<T> squared = {
            false, // non-negative
            1.0, // scale
            20, // constant
            0, // termination penalty
            20, // position
            2.5, // orientation
            0.5, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            0.5, // action
    };
}
