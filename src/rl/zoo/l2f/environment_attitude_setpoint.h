#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_ATTITUDE_SETPOINT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_ATTITUDE_SETPOINT_H

#include "setpoint_attitude.h"
#include "reward_attitude_setpoint_tracking.h"
#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include "setpoint_attitude_cpu.h"
#include <rl_tools/rl/environments/l2f/operations_multitask_generic.h>
#include <rl_tools/rl/environments/l2f/parameters/reward_functions/squared/operations_generic.h>
#include <rl_tools/rl/environments/l2f/parameters/reward_functions/default.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/crazyflie.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/arpl.h>
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
    template <typename DEVICE, typename TYPE_POLICY, typename TI>
    struct ENVIRONMENT_ATTITUDE_SETPOINT_FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        static constexpr TI SIMULATION_FREQUENCY = 100;
        static constexpr TI EPISODE_LENGTH_S = 5;
        using ENVIRONMENT_FACTORY_BASE = ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI, EPISODE_LENGTH_S, SIMULATION_FREQUENCY>;
        using REWARD_FUNCTION = rl_tools::rl::environments::l2f::parameters::reward_functions::AttitudeSetpointTrackingSquared<T>;
        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, ENVIRONMENT_FACTORY_BASE::EPISODE_STEP_LIMIT_OUTER, REWARD_FUNCTION>;
        using PARAMETERS_TYPE = ParametersAttitudeSetpoint<ParametersAttitudeSetpointSpecification<T, TI,
            ParametersTrajectory<ParametersTrajectorySpecification<T, TI, ENVIRONMENT_FACTORY_BASE::TRAJECTORY_LENGTH, ENVIRONMENT_FACTORY_BASE::TRAJECTORY_DT,
            ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DefaultParametersDomainRandomizationOptions,
            ParametersDisturbances<ParametersSpecification<T, TI,
            ParametersIMU<ParametersSpecification<T, TI,
            ParametersBase<PARAMETERS_SPEC>>>>>>>>>>>;

        static constexpr auto MODEL = rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie;
        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;
        static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = rl_tools::rl::environments::l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
        static constexpr typename ParametersBase<PARAMETERS_SPEC>::MDP::Initialization init = {
                0.0,
                0,
                1.5707963267948966 * 60.0/90.0,
                0,
                1,
                true,
                -1,
                +1,
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
                true,
                200,
                1.5707963267948966,
                100,
                35,
                10000,
                50000,
        };
        static constexpr REWARD_FUNCTION reward_function = {
                false,
                01.00,
                02.00,
                04.00,
                00.35,
                00.05,
                01.50,
                00.05,
                01.00,
        };
        static constexpr typename PARAMETERS_TYPE::MDP::ObservationNoise observation_noise = {
            0,
            0,
            0,
            0.005,
            0.05,
        };
        static constexpr typename PARAMETERS_TYPE::IMU imu = {
            {0.02, 60.0, 0.005}
        };
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            init,
            reward_function,
            observation_noise,
            ENVIRONMENT_FACTORY_BASE::action_noise,
            termination
        };
        static constexpr typename PARAMETERS_TYPE::Integration integration = {
            (T)1/(T)SIMULATION_FREQUENCY
        };

        static constexpr decltype(ENVIRONMENT_FACTORY_BASE::trajectory) trajectory = {};
        static constexpr typename PARAMETERS_TYPE::TrajectoryParameters trajectory_parameters = {
            rl::environments::l2f::parameters::trajectories::Type::LISSAJOUS,
            {
                rl::environments::l2f::parameters::trajectories::lissajous::Parameters<T>{
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0
                }
            }
        };
        static constexpr typename PARAMETERS_TYPE::AttitudeSetpointSampling attitude_setpoint_sampling = {
            (T)0.5235987755982988,
            (T)2.0,
            (T)0.4,
            (T)1.6,
            (TI)25,
            (TI)100,
        };

        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                {
                    {
                        {
                            {
                                dynamics,
                                integration,
                                mdp
                            },
                            imu
                        },
                        ENVIRONMENT_FACTORY_BASE::disturbances
                    },
                    ENVIRONMENT_FACTORY_BASE::domain_randomization
                },
                trajectory,
                trajectory_parameters
            },
            attitude_setpoint_sampling
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 32;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT_FACTORY_BASE::EPISODE_STEP_LIMIT_OUTER;
            static constexpr TI CLOSED_FORM = false;
            using STATE_BASE_INNER = StateBase<StateSpecification<T, TI>>;
            using STATE_BASE_LA = StateLastAction<StateSpecification<T, TI, STATE_BASE_INNER>>;
            using STATE_BASE_LAA = StateLinearAcceleration<StateSpecification<T, TI, STATE_BASE_LA>>;
            using STATE_BASE_GB = StateGyroBias<StateGyroBiasSpecification<T, TI, STATE_BASE_LAA>>;
            using STATE_BASE = StateMahony<StateMahonySpecification<T, TI, STATE_BASE_GB>>;
            using STATE_WITH_RANDOM_FORCE = StateRandomForce<StateSpecification<T, TI, STATE_BASE>>;
            using STATE_WITH_ROTORS = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, STATE_WITH_RANDOM_FORCE>>;
            using STATE_WITH_TRAJECTORY = StateTrajectory<StateSpecification<T, TI, STATE_WITH_ROTORS>>;
            using STATE_TYPE = StateAttitudeSetpoint<StateAttitudeSetpointSpecification<T, TI, STATE_WITH_TRAJECTORY>>;
            using OBSERVATION_TYPE = observation::AttitudeSetpoint<observation::AttitudeSetpointSpecification<T, TI,
                    observation::OrientationMahonyWorldZ<observation::OrientationMahonyWorldZSpecification<T, TI,
                    observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                    observation::LinearAccelerationBodyFrame<observation::LinearAccelerationBodyFrameSpecification<T, TI,
                    observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = observation::AttitudeSetpoint<observation::AttitudeSetpointSpecificationPrivileged<T, TI,
                    typename ENVIRONMENT_FACTORY_BASE::ENVIRONMENT_STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED>>;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
            static constexpr T STATE_LIMIT_POSITION = 100000;
            static constexpr T STATE_LIMIT_VELOCITY = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
