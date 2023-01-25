

#include <layer_in_c/rl/environments/multirotor/parameters/reward_functions/abs_exp.h>
#include <layer_in_c/rl/environments/multirotor/parameters/dynamics/crazy_flie.h>
#include <layer_in_c/rl/environments/multirotor/parameters/init/default.h>


namespace parameters_0{
    template <typename T>
    using REWARD_FUNCTION = layer_in_c::rl::environments::multirotor::parameters::reward_functions::AbsExp<T>;

    template <typename T>
    REWARD_FUNCTION<T> reward_function = [](){
        REWARD_FUNCTION<T> f;
        f.scale = 10;
        f.position = 1;
        f.orientation = 5;
        f.linear_velocity = 0;
        f.angular_velocity = 0;
        f.action_baseline = 0;
        f.action = 0;
        return f;
    }();

    template<typename T, typename TI>
    const layer_in_c::rl::environments::multirotor::Parameters<T, TI, 4, REWARD_FUNCTION<T>> parameters = {
            layer_in_c::rl::environments::multirotor::parameters::dynamics::crazy_flie<T, TI, REWARD_FUNCTION<T>>,
            {0.01}, // integration dt
            {
                    layer_in_c::rl::environments::multirotor::parameters::init::simple<T, TI, 4, REWARD_FUNCTION<T>>,
                    typename layer_in_c::rl::environments::multirotor::Parameters<T, TI, 4, REWARD_FUNCTION<T>>::MDP::Termination({true, 2, 50/3.6, 5*2*layer_in_c::math::PI<T>}),
                    reward_function<T>
            }
    };
}
