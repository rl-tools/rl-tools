#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_BIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_BIG_H

#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/parameters/reward_functions/squared.h>
#include <rl_tools/rl/environments/l2f/parameters/reward_functions/default.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/crazyflie.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/race.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/x500_sim.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/x500_real.h>
#include <rl_tools/rl/environments/l2f/parameters/init/default.h>
#include <rl_tools/rl/environments/l2f/parameters/termination/default.h>


#include "environment.h"
#include <rl_tools/utils/generic/typing.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f{
    namespace rlt = rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename T, typename TI>
    struct ENVIRONMENT_BIG_FACTORY{
        using ENVIRONMENT_FACTORY = ENVIRONMENT_FACTORY<DEVICE, T, TI>;
        using PARAMETERS_SPEC = typename ENVIRONMENT_FACTORY::PARAMETERS_SPEC;
        using PARAMETERS_TYPE = typename ENVIRONMENT_FACTORY::PARAMETERS_TYPE;

        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            ENVIRONMENT_FACTORY::init,
            ENVIRONMENT_FACTORY::reward_function,
            ENVIRONMENT_FACTORY::observation_noise,
            ENVIRONMENT_FACTORY::action_noise,
            ENVIRONMENT_FACTORY::termination
        };
        static constexpr TI SIMULATION_FREQUENCY = 100;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {
            1.0/((T)SIMULATION_FREQUENCY) // integration dt
        };
        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                ENVIRONMENT_FACTORY::dynamics,
                integration,
                mdp,
                ENVIRONMENT_FACTORY::domain_randomization
            },
            ENVIRONMENT_FACTORY::disturbances
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI ACTION_HISTORY_LENGTH = 16;
            static constexpr TI EPISODE_STEP_LIMIT = 5 * SIMULATION_FREQUENCY;
            static constexpr TI CLOSED_FORM = false;
            using STATE_BASE = StateBase<T, TI>;
            using STATE_TYPE = StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<T, TI, STATE_BASE>>;
            using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                            observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                                    observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                                            observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = typename ENVIRONMENT_FACTORY::ENVIRONMENT_STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif