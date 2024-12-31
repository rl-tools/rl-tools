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

#include <rl_tools/rl/environments/l2f/persist_code.h>


#include "environment.h"
#include <rl_tools/utils/generic/typing.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f{
    namespace rlt = rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename T, typename TI>
    struct ENVIRONMENT_BIG_FACTORY{
        using ENVIRONMENT_FACTORY_BASE = ENVIRONMENT_FACTORY<DEVICE, T, TI>;
        using PARAMETERS_SPEC = typename ENVIRONMENT_FACTORY_BASE::PARAMETERS_SPEC;
        using PARAMETERS_TYPE = typename ENVIRONMENT_FACTORY_BASE::PARAMETERS_TYPE;

        using REWARD_FUNCTION = rl_tools::rl::environments::l2f::parameters::reward_functions::Squared<T>;
        static constexpr REWARD_FUNCTION reward_function = {
                false, // non-negative
                00.10, // scale
                01.10, // constant
                00.00, // termination penalty
                10.00, // position
                02.50, // orientation
                01.00, // linear_velocity
                00.00, // angular_velocity
                00.00, // linear_acceleration
                00.00, // angular_acceleration
                02.00, // action
        };
//        static constexpr auto reward_function = [](){
//            auto reward_function = ENVIRONMENT_FACTORY::reward_function;
////            reward_function.constant = 0;
//            reward_function.scale = 0.09;
//            reward_function.position = 20;
//            reward_function.orientation = 0.05;
//            reward_function.linear_velocity = 2.0;
//            reward_function.action = 0.0;
////            reward_function.termination_penalty = 0;
//            return reward_function;
//        }();

        static constexpr auto termination = [](){
            auto termination = ENVIRONMENT_FACTORY_BASE::termination;
            termination.linear_velocity_threshold = 2;
            return termination;
        }();

        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            ENVIRONMENT_FACTORY_BASE::init,
//            [](){
//                auto base = rl_tools::rl::environments::l2f::parameters::init::init_0_deg<PARAMETERS_SPEC>;
//                base.max_linear_velocity = 0.0;
//                base.max_angular_velocity = 0.0;
//                return base;
//            }(),
            reward_function,
            ENVIRONMENT_FACTORY_BASE::observation_noise,
            ENVIRONMENT_FACTORY_BASE::action_noise,
            termination
        };
        static constexpr TI SIMULATION_FREQUENCY = 100;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {
            1.0/((T)SIMULATION_FREQUENCY) // integration dt
        };
        static constexpr PARAMETERS_TYPE nominal_parameters(const typename PARAMETERS_TYPE::Dynamics& dynamics)
        {
            return {
                {
                    dynamics,
                    integration,
                    mdp,
                    ENVIRONMENT_FACTORY_BASE::domain_randomization
                },
    //            ENVIRONMENT_FACTORY::disturbances
                typename PARAMETERS_TYPE::Disturbances{
                    typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0}, //{0, 0.027 * 9.81 / 3}, // random_force;
                    typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} //{0, 0.027 * 9.81 / 10000} // random_torque;
                }
            };
        }

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
            using OBSERVATION_TYPE_PRIVILEGED = typename ENVIRONMENT_FACTORY_BASE::ENVIRONMENT_STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters(ENVIRONMENT_FACTORY_BASE::dynamics);
            static constexpr TI N_DYNAMICS_VALUES = 2;
            static constexpr typename PARAMETERS_TYPE::Dynamics DYNAMICS_VALUES[N_DYNAMICS_VALUES] = {
                rl_tools::rl::environments::l2f::parameters::dynamics::registry<rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::x500_real, PARAMETERS_SPEC>,
                rl_tools::rl::environments::l2f::parameters::dynamics::registry<rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>
            };
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif