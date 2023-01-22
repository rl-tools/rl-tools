

#include "../../multirotor.h"
#include "abs_exp.h"
namespace layer_in_c::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    AbsExp<T> reward_1 = [](){
        AbsExp<T> reward_function;
        reward_function.scale = 10;
        reward_function.position = 1;
        reward_function.orientation = 5;
        reward_function.linear_velocity = 0;
        reward_function.angular_velocity = 0;
        reward_function.action_baseline = 0;
        reward_function.action = 0;
        return reward_function;
    }();
}