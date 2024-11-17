#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DEFAULT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_DEFAULT_H

#include "../multirotor.h"

#include "dynamics/mrs.h"
#include "init/default.h"
#include "reward_functions/default.h"
#include "termination/default.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters {
    struct DEFAULT_CONFIG{
        static constexpr bool ZERO_ORIENTATION_INIT = true;
    };
    template<typename T, typename TI, typename CONFIG = DEFAULT_CONFIG>
    struct DefaultParameters{
        constexpr static auto MODEL = rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie;

        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;
        static constexpr auto reward_function = rl_tools::rl::environments::l2f::parameters::reward_functions::squared<T>;
//        static constexpr auto reward_function = rl_tools::rl::environments::l2f::parameters::reward_functions::squared_no_orientation<T>;
        using REWARD_FUNCTION_CONST = typename rl_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
        using REWARD_FUNCTION = typename rl_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

        using PARAMETERS_SPEC = rl_tools::rl::environments::l2f::ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION, rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY, MODEL>;
        using PARAMETERS_TYPE = rl_tools::rl::environments::l2f::ParametersDisturbances<T, TI, rl_tools::rl::environments::l2f::ParametersBase<PARAMETERS_SPEC>>;

        static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = rl_tools::rl::environments::l2f::parameters::dynamics::registry<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {
            0.01 // integration dt
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = CONFIG::ZERO_ORIENTATION_INIT ? rl_tools::rl::environments::l2f::parameters::init::init_0_deg<PARAMETERS_SPEC> : rl_tools::rl::environments::l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP::ObservationNoise observation_noise = {
            0.0, // position
            0.00, // orientation
            0.0, // linear_velocity
            0.0, // angular_velocity
            0.0, // imu acceleration
        };
//        static constexpr typename PARAMETERS_TYPE::MDP::ObservationNoise observation_noise = {
//                0.0, // position
//                0.0, // orientation
//                0.0, // linear_velocity
//                0.0, // angular_velocity
//                0.0, // imu acceleration
//        };
        static constexpr typename PARAMETERS_TYPE::MDP::ActionNoise action_noise = {
            0, // std of additive gaussian noise onto the normalized action (-1, 1)
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = rl_tools::rl::environments::l2f::parameters::termination::fast_learning<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            init,
            reward_function,
            observation_noise,
            action_noise,
            termination
        };
        static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
            0.0, // thrust_to_weight_min;
            0.0, // thrust_to_weight_max;
            0.0, // thrust_to_weight_by_torque_to_inertia_min;
            0.0, // thrust_to_weight_by_torque_to_inertia_max;
            0.02, // mass_min;
            0.035, // mass_max;
            0.0, // mass_size_deviation;
            0.0, // motor_time_constant;
            0.0 // rotor_torque_constant;
        };
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0}, // random_force;
            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_torque;
        };
        static constexpr PARAMETERS_TYPE parameters = {
            {
                dynamics,
                integration,
                mdp,
                domain_randomization
            },
            disturbances
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI ACTION_HISTORY_LENGTH = 16;
            static constexpr TI CLOSED_FORM = false;
            using STATE_BASE = StateBase<T, TI>;
            using STATE_TYPE = StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<T, TI, STATE_BASE>>;
            using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                            observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                                    observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                                            observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = observation::Position<observation::PositionSpecificationPrivileged<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                            observation::LinearVelocity<observation::LinearVelocitySpecificationPrivileged<T, TI,
                                    observation::AngularVelocity<observation::AngularVelocitySpecificationPrivileged<T, TI,
                                            observation::RandomForce<observation::RandomForceSpecification<T, TI,
                                                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>
                                            >
                                            >
                                    >>
                            >>
                    >>
            >>;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = parameters;
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif