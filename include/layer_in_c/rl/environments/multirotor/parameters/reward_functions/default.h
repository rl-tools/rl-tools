

#include "../../multirotor.h"
#include "abs_exp.h"
namespace layer_in_c::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    AbsExp<T> reward_263 = [](){
        AbsExp<T> reward_function;
        reward_function.scale = 10;
        reward_function.position = 10;
        reward_function.orientation = 10;
        reward_function.linear_velocity = 0;
        reward_function.angular_velocity = 0;
        reward_function.action_baseline = -1;
        reward_function.action = 1.0/2; // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
        return reward_function;
    }();

    template<typename T>
    AbsExp<T> reward_dr = [](){
        AbsExp<T> reward_function;
        reward_function.scale = 10;
        reward_function.position = 1;
        reward_function.orientation = 5;
        reward_function.linear_velocity = 0.5;
        reward_function.angular_velocity = 0.005;
        reward_function.action_baseline = -1;
        reward_function.action = 1.0/2; // divide by to because actions are transformed from -1 -> 1 to 0 to 2 by the baseline => norm will be 2x
        return reward_function;
    }();

    template<typename T>
    AbsExp<T> reward_angular_velocity = [](){
        AbsExp<T> reward_function;
        reward_function.scale = 1;
        reward_function.position = 0;
        reward_function.orientation = 0;
        reward_function.linear_velocity = 0;
        reward_function.angular_velocity = 0.01;
        reward_function.action_baseline = 0;
        reward_function.action = 1.0/2.0;
        return reward_function;
    }();
}