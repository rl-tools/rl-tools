#include "../../multirotor.h"

#define RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_POSITION (0.5)
#define RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_LINEAR_VELOCITY (0.5)
#define RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_ANGULAR_VELOCITY (1)

namespace rl_tools::rl::environments::multirotor::parameters::init{
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization init_90_deg = {
            0.1, // guidance
            0.2, // position
            1.5707963267948966,   // orientation
            1,   // linear velocity
            1,  // angular velocity
            true,// relative rpm
            -1,  // min rpm
            +1,  // max rpm
    };
}
