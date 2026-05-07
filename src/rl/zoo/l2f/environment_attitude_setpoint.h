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
            ParametersTrajectory<ParametersTrajectorySpecification<T, TI, ENVIRONMENT_FACTORY_BASE::TRAJECTORY_LENGTH, ENVIRONMENT_FACTORY_BASE::TRAJECTORY_DT,
            ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS,
            ParametersDisturbances<ParametersSpecification<T, TI,
            ParametersIMU<ParametersSpecification<T, TI,
            ParametersBase<PARAMETERS_SPEC>>>>>>>>>>>;

        static constexpr auto MODEL = rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie;
        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;
        // static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = rl_tools::rl::environments::l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = [](){
            auto p = rl_tools::rl::environments::l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
            // p.rotor_time_constants_rising[0] = 0.072;
            // p.rotor_time_constants_rising[1] = 0.072;
            // p.rotor_time_constants_rising[2] = 0.072;
            // p.rotor_time_constants_rising[3] = 0.072;
            // p.rotor_time_constants_falling[0] = 0.072;
            // p.rotor_time_constants_falling[1] = 0.072;
            // p.rotor_time_constants_falling[2] = 0.072;
            // p.rotor_time_constants_falling[3] = 0.072;
            // p.mass = 0.025;
            // p.rotor_thrust_coefficients[0][0] = 0;
            // p.rotor_thrust_coefficients[1][0] = 0;
            // p.rotor_thrust_coefficients[2][0] = 0;
            // p.rotor_thrust_coefficients[3][0] = 0;
            // p.rotor_thrust_coefficients[0][1] = 0;
            // p.rotor_thrust_coefficients[1][1] = 0;
            // p.rotor_thrust_coefficients[2][1] = 0;
            // p.rotor_thrust_coefficients[3][1] = 0;
            // p.rotor_thrust_coefficients[0][2] = 0.1302;
            // p.rotor_thrust_coefficients[1][2] = 0.1302;
            // p.rotor_thrust_coefficients[2][2] = 0.1302;
            // p.rotor_thrust_coefficients[3][2] = 0.1302;
            // // Recompute hover throttle for the overridden mass + thrust curve
            // // (registry value was for the original curve and mass).
            // // hover_rpm = sqrt(m*g/(4*c2)) with action_limit [0, 1] -> hovering_throttle_relative = hover_rpm
            // p.hovering_throttle_relative = 0.6864;
            return p;
        }();
        static constexpr typename ParametersBase<PARAMETERS_SPEC>::MDP::Initialization init = {
                0.0, // guidance probability
                0,   // max initial position
                1.5707963267948966 * 0.0/90.0, // max initial attitude error, 60 deg
                0,   // max initial linear velocity
                1,   // max initial angular velocity
                true, // initialize rotor speeds relative to action limits
                -1,  // min relative rotor command
                +1,  // max relative rotor command
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
                true,  // enabled
                20000,   // position runaway threshold
                10,    // attitude threshold, effectively disabled for this target
                10000,   // linear velocity runaway threshold
                35,    // angular velocity safety threshold
                10000, // unused pose-integral guard
                50000, // unused attitude-integral guard
        };
        static constexpr REWARD_FUNCTION reward_function = {
                false, // non_negative
                01.00, // scale
                01.00, // constant
                00.40, // tilt
                00.05, // yaw_rate
                00.05, // angular_velocity_xy
                00.25, // thrust_g
                00.01, // d_action
                00.00, // action_saturation
        };
        static constexpr typename PARAMETERS_TYPE::MDP::ObservationNoise observation_noise = {
            0,     // position
            0,     // orientation
            0,     // linear velocity
            0.005, // angular velocity, rad/s
            0.05,  // accelerometer specific force, m/s^2
        };
        static constexpr typename PARAMETERS_TYPE::IMU imu = {
            // {0.02, 60.0, 0.005} // gyro bias: init half-range, OU tau, OU steady-state sigma
            {0, 0, 0}
        };
        static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
            0, // min thrust-to-weight after randomized thrust-curve scaling
            0, // max thrust-to-weight after randomized thrust-curve scaling
            0,      // torque-to-inertia disabled
            0,
            0,      // mass randomization disabled
            0,
            0,      // mass-size deviation disabled
            0,      // rotor rising time-constant randomization disabled
            0,
            0,      // rotor falling time-constant randomization disabled
            0,
            0,      // rotor torque-constant randomization disabled
            0,
            0,      // orientation offset randomization disabled
            0       // disturbance-force randomization disabled
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
                    0, // A
                    0, // B
                    0, // C
                    1, // a
                    1, // b
                    1, // c
                    1, // interval
                    0  // ramp duration
                }
            }
        };
        static constexpr typename PARAMETERS_TYPE::AttitudeSetpointSampling attitude_setpoint_sampling = {
            (T)0.5235987755982988, // max tilt angle, 30 deg
            (T)2.0,                // max yaw rate, rad/s
            (T)0.4,                // min thrust command, g
            (T)1.4,                // max thrust command, g; below min randomized thrust-to-weight
            (TI)25,                // min hold time, 0.25 s at 100 Hz
            (TI)100,               // max hold time, 1.0 s at 100 Hz
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
                    domain_randomization
                },
                trajectory,
                trajectory_parameters
            },
            attitude_setpoint_sampling
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr auto ACTION_INTERFACE = parameters::ActionInterface::DIRECT_MOTOR;
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 32;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT_FACTORY_BASE::EPISODE_STEP_LIMIT_OUTER;
            static constexpr TI CLOSED_FORM = false;
            // Innermost-first: physical state, last action, finite-difference acceleration,
            // gyro bias, Mahony reduced attitude, random force, rotors/history, trajectory,
            // then the current setpoint command.
            using STATE_BASE_INNER = StateBase<StateSpecification<T, TI>>;
            using STATE_BASE_LA = StateLastAction<StateSpecification<T, TI, STATE_BASE_INNER>>;
            using STATE_BASE_LAA = StateLinearAcceleration<StateSpecification<T, TI, STATE_BASE_LA>>;
            using STATE_BASE_GB = StateGyroBias<StateGyroBiasSpecification<T, TI, STATE_BASE_LAA>>;
            using STATE_BASE = StateMahony<StateMahonySpecification<T, TI, STATE_BASE_GB>>;
            using STATE_WITH_RANDOM_FORCE = StateRandomForce<StateSpecification<T, TI, STATE_BASE>>;
            using STATE_WITH_ROTORS = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, STATE_WITH_RANDOM_FORCE>>;
            using STATE_WITH_TRAJECTORY = StateTrajectory<StateSpecification<T, TI, STATE_WITH_ROTORS>>;
            using STATE_TYPE = StateAttitudeSetpoint<StateAttitudeSetpointSpecification<T, TI, STATE_WITH_TRAJECTORY>>;
            // Actor observation: command plus onboard-estimable attitude, gyro,
            // accelerometer, and motor-action history. No yaw or yaw-bias estimate.
            using OBSERVATION_TYPE = observation::AttitudeSetpoint<observation::AttitudeSetpointSpecification<T, TI,
                    observation::OrientationWorldZ<observation::OrientationWorldZSpecification<T, TI,
                    observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                    observation::LinearAccelerationBodyFrame<observation::LinearAccelerationBodyFrameSpecification<T, TI,
                    observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = observation::AttitudeSetpoint<observation::AttitudeSetpointSpecificationPrivileged<T, TI,
                    typename ENVIRONMENT_FACTORY_BASE::ENVIRONMENT_STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED>>;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
            static constexpr T STATE_LIMIT_POSITION_X = 100000;
            static constexpr T STATE_LIMIT_POSITION_Y = 100000;
            static constexpr T STATE_LIMIT_POSITION_Z = 100000;
            static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
            static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
            static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
