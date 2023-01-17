#include "../multirotor.h"

#include "dynamics/mrs.h"
#include "init/default.h"
#include "reward_functions/default.h"
namespace layer_in_c::rl::environments::multirotor::parameters {
    template<typename T, typename TI, TI N>
    rl::environments::multirotor::Parameters<T, TI, 4> default_parameters = {
            rl::environments::multirotor::parameters::dynamics::mrs<T, TI>,
            {0.02}, // integration dt
            {
                    rl::environments::multirotor::parameters::init::simple<T, TI, 4>,
                    rl::environments::multirotor::parameters::reward_functions::reward_1<T, TI, 4>
            }
    };

}