#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/abs_exp.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/squared.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/default.h>
#include <backprop_tools/rl/environments/multirotor/parameters/dynamics/crazy_flie.h>
#include <backprop_tools/rl/environments/multirotor/parameters/init/default.h>
#include <backprop_tools/rl/environments/multirotor/parameters/termination/default.h>

namespace parameters{
    template<typename T, typename TI>
    struct environment{
        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_dr<T>;
        using REWARD_FUNCTION_CONST = typename backprop_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
        using REWARD_FUNCTION = typename backprop_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

        static constexpr backprop_tools::rl::environments::multirotor::Parameters<T, TI, 4, REWARD_FUNCTION> parameters = {
                backprop_tools::rl::environments::multirotor::parameters::dynamics::crazy_flie<T, TI, REWARD_FUNCTION>,
                {0.01}, // integration dt
                {
                        backprop_tools::rl::environments::multirotor::parameters::init::all_around<T, TI, 4, REWARD_FUNCTION>,
                        backprop_tools::rl::environments::multirotor::parameters::termination::classic<T, TI, 4, REWARD_FUNCTION>,
                        reward_function,
                }
        };

        using PARAMETERS = typename backprop_tools::utils::typing::remove_cv_t<decltype(parameters)>;

        using ENVIRONMENT_SPEC = bpt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, bpt::rl::environments::multirotor::StaticParameters>;
        using ENVIRONMENT = bpt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };


}
