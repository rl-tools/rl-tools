#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_ATTITUDE_SETPOINT_MINIMAL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_ATTITUDE_SETPOINT_MINIMAL_H

#include "environment_attitude_setpoint.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f{
    namespace rlt = rl_tools;
    using namespace rl_tools::rl::environments::l2f;

    template <typename DEVICE, typename TYPE_POLICY, typename TI>
    struct ENVIRONMENT_ATTITUDE_SETPOINT_MINIMAL_FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        using BASE = ENVIRONMENT_ATTITUDE_SETPOINT_FACTORY<DEVICE, TYPE_POLICY, TI>;
        using PARAMETERS_TYPE = typename BASE::PARAMETERS_TYPE;

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr auto ACTION_INTERFACE = parameters::ActionInterface::DIRECT_MOTOR;
            static constexpr TI N_SUBSTEPS = BASE::ENVIRONMENT_STATIC_PARAMETERS::N_SUBSTEPS;
            static constexpr TI ACTION_HISTORY_LENGTH = 0;
            static constexpr TI EPISODE_STEP_LIMIT = BASE::ENVIRONMENT_STATIC_PARAMETERS::EPISODE_STEP_LIMIT;
            static constexpr TI CLOSED_FORM = BASE::ENVIRONMENT_STATIC_PARAMETERS::CLOSED_FORM;

            using STATE_BASE_INNER = StateBase<StateSpecification<T, TI>>;
            using STATE_BASE_LA = StateLastAction<StateSpecification<T, TI, STATE_BASE_INNER>>;
            using STATE_BASE_LAA = StateLinearAcceleration<StateSpecification<T, TI, STATE_BASE_LA>>;
            using STATE_BASE_GB = StateIMU<T, TI, STATE_BASE_LAA>;
            using STATE_BASE = StateMahony<StateMahonySpecification<T, TI, STATE_BASE_GB>>;
            using STATE_WITH_RANDOM_FORCE = StateRandomForce<StateSpecification<T, TI, STATE_BASE>>;
            using STATE_WITH_ROTORS = StateRotors<StateRotorsSpecification<T, TI, CLOSED_FORM, STATE_WITH_RANDOM_FORCE>>;
            using STATE_WITH_TRAJECTORY = StateTrajectory<StateSpecification<T, TI, STATE_WITH_ROTORS>>;
            using STATE_TYPE = StateAttitudeSetpoint<StateAttitudeSetpointSpecification<T, TI, STATE_WITH_TRAJECTORY>>;

            using OBSERVATION_TYPE = observation::AttitudeSetpoint<observation::AttitudeSetpointSpecification<T, TI,
                    observation::OrientationMahonyWorldZ<observation::OrientationMahonyWorldZSpecification<T, TI>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = BASE::nominal_parameters;
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
