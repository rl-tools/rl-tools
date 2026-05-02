#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_ATTITUDE_SETPOINT_CTBR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_ATTITUDE_SETPOINT_CTBR_H

#include "environment_attitude_setpoint.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f{
    namespace rlt = rl_tools;
    using namespace rl_tools::rl::environments::l2f;

    template <typename DEVICE, typename TYPE_POLICY, typename TI>
    struct ENVIRONMENT_ATTITUDE_SETPOINT_CTBR_FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        using BASE = ENVIRONMENT_ATTITUDE_SETPOINT_FACTORY<DEVICE, TYPE_POLICY, TI>;
        using REWARD_FUNCTION = rl_tools::rl::environments::l2f::parameters::reward_functions::AttitudeSetpointTrackingCTBRSquared<T>;
        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, BASE::ENVIRONMENT_FACTORY_BASE::EPISODE_STEP_LIMIT_OUTER, REWARD_FUNCTION>;
        struct DOMAIN_RANDOMIZATION_OPTIONS{
            static constexpr bool THRUST_TO_WEIGHT = false;
            static constexpr bool MASS = false;
            static constexpr bool TORQUE_TO_INERTIA = false;
            static constexpr bool MASS_SIZE_DEVIATION = false;
            static constexpr bool ROTOR_TORQUE_CONSTANT = false;
            static constexpr bool DISTURBANCE_FORCE = false;
            static constexpr bool ROTOR_TIME_CONSTANT = false;
        };
        using PARAMETERS_TYPE = ParametersAttitudeSetpoint<ParametersAttitudeSetpointSpecification<T, TI,
            ParametersCTBRController<ParametersSpecification<T, TI,
            ParametersTrajectory<ParametersTrajectorySpecification<T, TI, BASE::ENVIRONMENT_FACTORY_BASE::TRAJECTORY_LENGTH, BASE::ENVIRONMENT_FACTORY_BASE::TRAJECTORY_DT,
            ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS,
            ParametersDisturbances<ParametersSpecification<T, TI,
            ParametersIMU<ParametersSpecification<T, TI,
            ParametersBase<PARAMETERS_SPEC>>>>>>>>>>>>>;

        static constexpr typename PARAMETERS_TYPE::CTBRController ctbr_controller = {
            (T)0,
            (T)1,
            {(T)10, (T)10, (T)10},
            {216, 216, 96},
            {5.76, 5.76, 1.6},
            {(T)0.0030, (T)0.0030, (T)0.0012}
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
                0.0,
                0,
                (T)0.17453292519943295,
                0,
                (T)0.5,
                true,
                (T)(2 * BASE::dynamics.hovering_throttle_relative - 1),
                (T)(2 * BASE::dynamics.hovering_throttle_relative - 1),
        };
        static constexpr REWARD_FUNCTION reward_function = {
                false,
                01.00,
                03.00,
                04.00,
                00.50,
                02.50,
                00.00
        };
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            init,
            reward_function,
            BASE::observation_noise,
            BASE::ENVIRONMENT_FACTORY_BASE::action_noise,
            BASE::termination
        };
        static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        };
        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                {
                    {
                        {
                            {
                                {
                                    BASE::dynamics,
                                    BASE::integration,
                                    mdp
                                },
                                BASE::imu
                            },
                            BASE::ENVIRONMENT_FACTORY_BASE::disturbances
                        },
                        domain_randomization
                    },
                    BASE::trajectory,
                    BASE::trajectory_parameters
                },
                ctbr_controller
            },
            BASE::attitude_setpoint_sampling
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr auto ACTION_INTERFACE = parameters::ActionInterface::CTBR;
            static constexpr TI N_SUBSTEPS = BASE::ENVIRONMENT_STATIC_PARAMETERS::N_SUBSTEPS;
            static constexpr TI ACTION_HISTORY_LENGTH = 16;
            static constexpr TI EPISODE_STEP_LIMIT = BASE::ENVIRONMENT_STATIC_PARAMETERS::EPISODE_STEP_LIMIT;
            static constexpr TI CLOSED_FORM = BASE::ENVIRONMENT_STATIC_PARAMETERS::CLOSED_FORM;
            static_assert(CLOSED_FORM == false);

            using STATE_BASE_INNER = StateBase<StateSpecification<T, TI>>;
            using STATE_BASE_LA = StateLastAction<StateSpecification<T, TI, STATE_BASE_INNER>>;
            using STATE_BASE_LAA = StateLinearAcceleration<StateSpecification<T, TI, STATE_BASE_LA>>;
            using STATE_BASE_GB = StateGyroBias<StateGyroBiasSpecification<T, TI, STATE_BASE_LAA>>;
            using STATE_BASE_MAHONY = StateMahony<StateMahonySpecification<T, TI, STATE_BASE_GB>>;
            using STATE_BASE = StateCTBRController<StateSpecification<T, TI, STATE_BASE_MAHONY>>;
            using STATE_WITH_RANDOM_FORCE = StateRandomForce<StateSpecification<T, TI, STATE_BASE>>;
            using STATE_WITH_ROTORS = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, STATE_WITH_RANDOM_FORCE>>;
            using STATE_WITH_TRAJECTORY = StateTrajectory<StateSpecification<T, TI, STATE_WITH_ROTORS>>;
            using STATE_TYPE = StateAttitudeSetpoint<StateAttitudeSetpointSpecification<T, TI, STATE_WITH_TRAJECTORY>>;

            using OBSERVATION_TYPE = observation::AttitudeSetpoint<observation::AttitudeSetpointSpecification<T, TI,
                    observation::OrientationMahonyWorldZ<observation::OrientationMahonyWorldZSpecification<T, TI, observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
            static constexpr T STATE_LIMIT_POSITION_X = BASE::ENVIRONMENT_STATIC_PARAMETERS::STATE_LIMIT_POSITION_X;
            static constexpr T STATE_LIMIT_POSITION_Y = BASE::ENVIRONMENT_STATIC_PARAMETERS::STATE_LIMIT_POSITION_Y;
            static constexpr T STATE_LIMIT_POSITION_Z = BASE::ENVIRONMENT_STATIC_PARAMETERS::STATE_LIMIT_POSITION_Z;
            static constexpr T STATE_LIMIT_VELOCITY_X = BASE::ENVIRONMENT_STATIC_PARAMETERS::STATE_LIMIT_VELOCITY_X;
            static constexpr T STATE_LIMIT_VELOCITY_Y = BASE::ENVIRONMENT_STATIC_PARAMETERS::STATE_LIMIT_VELOCITY_Y;
            static constexpr T STATE_LIMIT_VELOCITY_Z = BASE::ENVIRONMENT_STATIC_PARAMETERS::STATE_LIMIT_VELOCITY_Z;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = BASE::ENVIRONMENT_STATIC_PARAMETERS::STATE_LIMIT_ANGULAR_VELOCITY_X;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = BASE::ENVIRONMENT_STATIC_PARAMETERS::STATE_LIMIT_ANGULAR_VELOCITY_Y;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = BASE::ENVIRONMENT_STATIC_PARAMETERS::STATE_LIMIT_ANGULAR_VELOCITY_Z;
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
