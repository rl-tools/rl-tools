

#include <layer_in_c/rl/environments/multirotor/parameters/reward_functions/abs_exp.h>
#include <layer_in_c/rl/environments/multirotor/parameters/reward_functions/squared.h>
#include <layer_in_c/rl/environments/multirotor/parameters/dynamics/crazy_flie.h>
#include <layer_in_c/rl/environments/multirotor/parameters/init/default.h>
#include <layer_in_c/rl/environments/multirotor/parameters/termination/default.h>


namespace parameters_0{


    template <typename T>
    auto reward_function = layer_in_c::rl::environments::multirotor::parameters::reward_functions::reward_dr<T>;

    template <typename T>
    using REWARD_FUNCTION = decltype(reward_function<T>);

    template<typename T, typename TI>
    const layer_in_c::rl::environments::multirotor::Parameters<T, TI, 4, REWARD_FUNCTION<T>> parameters = {
            layer_in_c::rl::environments::multirotor::parameters::dynamics::crazy_flie<T, TI, REWARD_FUNCTION<T>>,
            {0.01}, // integration dt
            {
                    layer_in_c::rl::environments::multirotor::parameters::init::all_around<T, TI, 4, REWARD_FUNCTION<T>>,
                    layer_in_c::rl::environments::multirotor::parameters::termination::classic<T, TI, 4, REWARD_FUNCTION<T>>,
                    reward_function<T>,
            }
    };
}
