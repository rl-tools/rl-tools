#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_CTBR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_CTBR_H

#include "environment_tiny.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f{
    namespace rlt = rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename TYPE_POLICY, typename TI>
    struct ENVIRONMENT_CTBR_FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        using BASE = ENVIRONMENT_TINY_FACTORY<DEVICE, TYPE_POLICY, TI>;
        static constexpr TI CONTROL_FREQUENCY = 10;
        static constexpr TI N_SUBSTEPS = 10;
        static constexpr TI EPISODE_LENGTH_S = 5;
        static constexpr TI EPISODE_STEP_LIMIT = EPISODE_LENGTH_S * CONTROL_FREQUENCY;
        static constexpr TI TRAJECTORY_LENGTH = EPISODE_STEP_LIMIT;
        static constexpr TI TRAJECTORY_DT = 1000000 / CONTROL_FREQUENCY;
        using REWARD_FUNCTION = typename BASE::REWARD_FUNCTION;
        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
        using PARAMETERS_TYPE = ParametersCTBRController<ParametersSpecification<T, TI,
            ParametersTrajectory<ParametersTrajectorySpecification<T, TI, TRAJECTORY_LENGTH, TRAJECTORY_DT,
            ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DefaultParametersDomainRandomizationOptions,
            ParametersDisturbances<ParametersSpecification<T, TI,
            ParametersBase<PARAMETERS_SPEC>>>>>>>>>;

        static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = BASE::dynamics;
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
                BASE::init.guidance,
                BASE::init.max_position,
                BASE::init.max_angle,
                BASE::init.max_linear_velocity,
                BASE::init.max_angular_velocity,
                true,
                (T)(2 * BASE::dynamics.hovering_throttle_relative - 1),
                (T)(2 * BASE::dynamics.hovering_throttle_relative - 1),
        };
        static constexpr REWARD_FUNCTION reward_function = {
                false,
                00.10,
                02.00,
                -10.00,
                10.00,
                00.00,
                02.50,
                03.00,
                00.10,
                00.00,
                00.00,
                {(T)0.00, (T)0.00, (T)0.00, (T)0.0000},
                {(T)0.0, (T)10.0, (T)10.0, (T)10.0},
                00.00
        };
        static constexpr typename PARAMETERS_TYPE::CTBRController ctbr_controller = {
            (T)0,
            (T)1,
            {(T)4, (T)4, (T)2},
            {(T)216, (T)216, (T)96},
            {(T)5.76, (T)5.76, (T)1.6},
            {(T)0.0030, (T)0.0030, (T)0.0012}
        };
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            init,
            reward_function,
            BASE::ENVIRONMENT_FACTORY_BASE::observation_noise,
            BASE::ENVIRONMENT_FACTORY_BASE::action_noise,
            BASE::ENVIRONMENT_FACTORY_BASE::termination
        };
        static constexpr typename PARAMETERS_TYPE::Integration integration = {
            (T)1 / (T)CONTROL_FREQUENCY
        };
        static constexpr typename PARAMETERS_TYPE::Trajectory trajectory = {};
        static constexpr typename PARAMETERS_TYPE::TrajectoryParameters trajectory_parameters = BASE::trajectory_parameters;
        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                {
                    {
                        {
                            dynamics,
                            integration,
                            mdp
                        },
                        BASE::ENVIRONMENT_FACTORY_BASE::disturbances
                    },
                    BASE::ENVIRONMENT_FACTORY_BASE::domain_randomization
                },
                trajectory,
                trajectory_parameters
            },
            ctbr_controller
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr auto ACTION_INTERFACE = parameters::ActionInterface::CTBR;
            static constexpr TI N_SUBSTEPS = ENVIRONMENT_CTBR_FACTORY::N_SUBSTEPS;
            static constexpr TI ACTION_HISTORY_LENGTH = BASE::ENVIRONMENT_STATIC_PARAMETERS::ACTION_HISTORY_LENGTH;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT_CTBR_FACTORY::EPISODE_STEP_LIMIT;
            static constexpr TI CLOSED_FORM = BASE::ENVIRONMENT_STATIC_PARAMETERS::CLOSED_FORM;
            static constexpr TI TRAJECTORY_TRACKING_LOOKAHEAD_STEPS = BASE::ENVIRONMENT_STATIC_PARAMETERS::TRAJECTORY_TRACKING_LOOKAHEAD_STEPS;
            static constexpr TI TRAJECTORY_TRACKING_LOOKAHEAD_INTERVAL = BASE::ENVIRONMENT_STATIC_PARAMETERS::TRAJECTORY_TRACKING_LOOKAHEAD_INTERVAL;
            using STATE_BASE_INNER = StateBase<StateSpecification<T, TI>>;
            using STATE_BASE_LAST_ACTION = StateLastAction<StateSpecification<T, TI, STATE_BASE_INNER>>;
            using STATE_BASE_CTBR = StateCTBRController<StateSpecification<T, TI, STATE_BASE_LAST_ACTION>>;
            using STATE_WITH_RANDOM_FORCE = StateRandomForce<StateSpecification<T, TI, STATE_BASE_CTBR>>;
            using STATE_WITH_ROTORS = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, STATE_WITH_RANDOM_FORCE>>;
            using STATE_TYPE = StateTrajectory<StateSpecification<T, TI, STATE_WITH_ROTORS>>;
            using OBSERVATION_TYPE = typename BASE::ENVIRONMENT_STATIC_PARAMETERS::OBSERVATION_TYPE;
            using OBSERVATION_TYPE_PRIVILEGED = typename BASE::ENVIRONMENT_STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = BASE::ENVIRONMENT_STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE;
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
