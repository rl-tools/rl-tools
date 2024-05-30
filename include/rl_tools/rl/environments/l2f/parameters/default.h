#include "../multirotor.h"

#include "dynamics/mrs.h"
#include "init/default.h"
#include "reward_functions/default.h"
#include "termination/default.h"
namespace rl_tools::rl::environments::multirotor::parameters {
    namespace default_internal{
        template <typename T>
        const auto reward_function = rl::environments::multirotor::parameters::reward_functions::reward_dr<T>;
        template <typename T>
        using REWARD_FUNCTION = decltype(reward_function<T>);
    }
    template <typename T>
    static constexpr auto reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque<T>;
    template <typename T>
    using REWARD_FUNCTION_CONST = typename rl_tools::utils::typing::remove_cv_t<decltype(reward_function<T>)>;
    template <typename T>
    using REWARD_FUNCTION = typename rl_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST<T>>::type;
    template <typename T, typename TI>
    using PARAMETERS_TYPE = rl_tools::rl::environments::multirotor::ParametersBase<T, TI, (TI)4, REWARD_FUNCTION<T>>;
    template<typename T, typename TI>
    const typename PARAMETERS_TYPE<T, TI>::Dynamics dynamics_parameters = rl::environments::multirotor::parameters::dynamics::mrs<T, TI, REWARD_FUNCTION<T>>;
    template<typename T, typename TI>
    const typename PARAMETERS_TYPE<T, TI>::Integration integration = {0.02};
    template<typename T, typename TI>
    const typename PARAMETERS_TYPE<T, TI>::MDP mdp = {
//                        rl_tools::rl::environments::multirotor::parameters::init::all_around_orientation_only<T, TI, 4, REWARD_FUNCTION>,
            rl_tools::rl::environments::multirotor::parameters::init::all_around_2<T, TI, 4, REWARD_FUNCTION<T>>,
//                        rl_tools::rl::environments::multirotor::parameters::init::orientation_all_around<T, TI, 4, REWARD_FUNCTION>,
//                        rl_tools::rl::environments::multirotor::parameters::init::simple<T, TI, 4, REWARD_FUNCTION>,
            reward_function<T>,
            {   // Observation noise
                    0.001, // position
                    0.001, // orientation
                    0.002, // linear_velocity
                    0.002, // angular_velocity
            },
            {   // Action noise
                    0, // std of additive gaussian noise onto the normalized action (-1, 1)
            },
            rl_tools::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION<T>>
    };
    template<typename T, typename TI>
    const PARAMETERS_TYPE<T, TI> default_parameters = {
            dynamics_parameters<T, TI>,
            integration<T, TI>,
            mdp<T, TI>
    };

}