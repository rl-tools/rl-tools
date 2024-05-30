#include "../../multirotor.h"

#define RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_POSITION (0.5)
#define RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_LINEAR_VELOCITY (0.5)
#define RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_ANGULAR_VELOCITY (1)

namespace rl_tools::rl::environments::multirotor::parameters::init{
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization all_around = {
            0.1, // guidance
            0.3, // position
            3.14,   // orientation
            1,   // linear velocity
            10,  // angular velocity
            true,// relative rpm
            -1,  // min rpm
            +1,  // max rpm
    };
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization all_around_2 = {
            0.1, // guidance
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_POSITION,   // position
            3.14,   // orientation
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_LINEAR_VELOCITY,   // linear velocity
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_ANGULAR_VELOCITY,   // angular velocity
            true,// relative rpm
            -0,  // min rpm
            +0,  // max rpm
    };
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization orientation_all_around = {
            0.1, // guidance
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_POSITION,   // position
            3.14,   // orientation
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_LINEAR_VELOCITY,   // linear velocity
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_ANGULAR_VELOCITY,   // angular velocity
            true,// relative rpm
            0,  // min rpm
            0,  // max rpm
    };
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization orientation_small_angle = {
            0.1, // guidance
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_POSITION,   // position
            10.0/180.0*3.14,   // orientation
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_LINEAR_VELOCITY,   // linear velocity
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_ANGULAR_VELOCITY,   // angular velocity
            true,// relative rpm
            0,  // min rpm
            0,  // max rpm
    };
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization orientation_big_angle = {
            0.1, // guidance
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_POSITION,   // position
            20.0/180.0 * 3.14,   // orientation
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_LINEAR_VELOCITY,   // linear velocity
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_ANGULAR_VELOCITY,   // angular velocity
            true,// relative rpm
            0,  // min rpm
            0,  // max rpm
    };
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization orientation_bigger_angle = {
            0.1, // guidance
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_POSITION,   // position
            45.0/180.0 * 3.14,   // orientation
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_LINEAR_VELOCITY,   // linear velocity
            RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_ANGULAR_VELOCITY,   // angular velocity
            true,// relative rpm
            0,  // min rpm
            0,  // max rpm
    };
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization orientation_biggest_angle = {
            0.0, // guidance
            0.0, //RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_POSITION,   // position
            45.0/180.0 * 3.14,   // orientation
            0.0,//RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_LINEAR_VELOCITY,   // linear velocity
            0.0,//RL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_INIT_ANGULAR_VELOCITY,   // angular velocity
            true,// relative rpm
            0,  // min rpm
            0,  // max rpm
    };
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization all_around_simplified = {
            0.1, // guidance
            0.3, // position
            0,   // orientation
            1,   // linear velocity
            10,  // angular velocity
            true,// relative rpm
            -1,  // min rpm
            +1,  // max rpm
    };
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization simple = {
            0,   // guidance
            0,   // position
            0,   // orientation
            0,   // linear velocity
            0,   // angular velocity
            true,// relative rpm
            0,   // min rpm
            0,   // max rpm
    };
    template<typename SPEC>
    constexpr typename ParametersBase<SPEC>::MDP::Initialization all_positions = {
            0.5, // guidance
            0.3, // position
            0,   // orientation
            0,   // linear velocity
            0,  // angular velocity
            true,// relative rpm
            0,  // min rpm
            0,  // max rpm
    };
}
