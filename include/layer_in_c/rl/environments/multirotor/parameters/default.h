#include "../multirotor.h"

#include "dynamics/mrs.h"
#include "init/default.h"
#include "reward_functions/default.h"
namespace layer_in_c::rl::environments::multirotor::parameters {
    namespace default_internal{
        template <typename T>
        const auto reward_function = rl::environments::multirotor::parameters::reward_functions::reward_1<T>;
        template <typename T>
        using REWARD_FUNCTION = decltype(reward_function<T>);
    }
    template<typename T, typename TI>
    const rl::environments::multirotor::Parameters<T, TI, 4, default_internal::REWARD_FUNCTION<T>> default_parameters = {
            rl::environments::multirotor::parameters::dynamics::mrs<T, TI, default_internal::REWARD_FUNCTION<T>>,
            {0.02}, // integration dt
            {
                    rl::environments::multirotor::parameters::init::simple<T, TI, 4, default_internal::REWARD_FUNCTION<T>>,
                    default_internal::reward_function<T>
            }
    };

}