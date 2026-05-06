#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_TINY_ATTITUDE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_TINY_ATTITUDE_H

#include "reward_attitude.h"
#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
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
    struct ENVIRONMENT_TINY_ATTITUDE_FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        static constexpr TI SIMULATION_FREQUENCY = 100;
        static constexpr TI EPISODE_LENGTH_S = 5;
        using ENVIRONMENT_FACTORY_BASE = ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI, EPISODE_LENGTH_S, SIMULATION_FREQUENCY>;
        using REWARD_FUNCTION = rl_tools::rl::environments::l2f::parameters::reward_functions::AttitudeSquared<T>;
        // Rebuild the PARAMETERS wrapping with our AttitudeSquared reward. All other
        // nested types (Dynamics, Integration, Termination, ...) resolve to
        // reward-function-independent concrete types, so the values from the base
        // factory remain assignable into this spec.
        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, ENVIRONMENT_FACTORY_BASE::EPISODE_STEP_LIMIT_OUTER, REWARD_FUNCTION>;
        using PARAMETERS_TYPE = ParametersTrajectory<ParametersTrajectorySpecification<T, TI, ENVIRONMENT_FACTORY_BASE::TRAJECTORY_LENGTH, ENVIRONMENT_FACTORY_BASE::TRAJECTORY_DT, ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DefaultParametersDomainRandomizationOptions, ParametersDisturbances<ParametersSpecification<T, TI, ParametersIMU<ParametersSpecification<T, TI, ParametersBase<PARAMETERS_SPEC>>>>>>>>>;

        static constexpr auto MODEL = rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie;
        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;
        static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = [](){
            auto p = rl_tools::rl::environments::l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
            // p.rotor_time_constants_rising[0] = 0.072;
            // // p.rotor_time_constants_rising[1] = 0.072;
            // // p.rotor_time_constants_rising[2] = 0.072;
            // // p.rotor_time_constants_rising[3] = 0.072;
            // // p.rotor_time_constants_falling[0] = 0.072;
            // // p.rotor_time_constants_falling[1] = 0.072;
            // // p.rotor_time_constants_falling[2] = 0.072;
            // // p.rotor_time_constants_falling[3] = 0.072;
            // // p.mass = 0.025;
            // // p.rotor_thrust_coefficients[0][0] = 0;
            // // p.rotor_thrust_coefficients[1][0] = 0;
            // // p.rotor_thrust_coefficients[2][0] = 0;
            // // p.rotor_thrust_coefficients[3][0] = 0;
            // // p.rotor_thrust_coefficients[0][1] = 0;
            // // p.rotor_thrust_coefficients[1][1] = 0;
            // // p.rotor_thrust_coefficients[2][1] = 0;
            // // p.rotor_thrust_coefficients[3][1] = 0;
            // // p.rotor_thrust_coefficients[0][2] = 0.1302;
            // // p.rotor_thrust_coefficients[1][2] = 0.1302;
            // // p.rotor_thrust_coefficients[2][2] = 0.1302;
            // // p.rotor_thrust_coefficients[3][2] = 0.1302;
            // // Recompute hover throttle for the overridden mass + thrust curve
            // // (registry value was for the original curve and mass).
            // // hover_rpm = sqrt(m*g/(4*c2)) with action_limit [0, 1] -> hovering_throttle_relative = hover_rpm
            // p.hovering_throttle_relative = 0.6864;
            return p;
        }();

        static constexpr typename ParametersBase<PARAMETERS_SPEC>::MDP::Initialization init = {
                0.0, // guidance
                0,   // position
                1.5707963267948966 * 60.0/90.0,   // orientation (~60 deg)
                0,   // linear velocity
                1,   // angular velocity
                true,// relative rpm
                -1,  // min rpm
                +1,  // max rpm
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
                true,  // enabled
                10,    // position (now actually fires: state is no longer clamped to 0)
                10, // angle (> pi)
                20,    // linear velocity (now actually fires: state is no longer clamped to 0)
                35,    // angular velocity (safety net)
                10000, // position integral
                50000, // orientation integral
        };
        // Tilt cost is 1 - R[2][2] — proper roll/pitch penalty with direct gradient
        // at any tilt, unlike Squared's orientation_cost which only penalizes yaw.
        static constexpr REWARD_FUNCTION reward_function = {
                true,  // non-negative: clip reward at 0 so dying can never beat surviving badly
                01.00, // scale
                01.00, // constant: alive bonus per step (must exceed worst-case weighted_cost to keep early-episode reward positive)
                00.00, // termination penalty (unused, see reward_attitude.h)
                02.00, // tilt
                00.15, // angular_velocity
                00.15, // d_action — penalize action change between steps (smoothness)
                00.15, // linear_acceleration — penalize jerky motion (gravity-subtracted)
        };
        // Realistic MEMS IMU noise (e.g. ICM-20602 at 50Hz BW) + gyro bias OU process.
        // The Mahony filter has to deal with these in real life, so we train against them.
        static constexpr typename PARAMETERS_TYPE::MDP::ObservationNoise observation_noise = {
            0,      // position
            0,      // orientation (privileged-critic only)
            0,      // linear_velocity
            0.005,  // angular_velocity (rad/s) — ~0.3 deg/s gyro white noise
            0.05,   // imu_acceleration (m/s^2) — ~0.5% g accel white noise
        };
        static constexpr typename PARAMETERS_TYPE::IMU imu = {
            {0.02, 60.0, 0.005} // gyro_bias: turn-on up to ~1 deg/s, OU tau 60s, steady-state sigma 0.3 deg/s
        };
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            init,
            reward_function,
            observation_noise,
            ENVIRONMENT_FACTORY_BASE::action_noise,
            termination
        };
        static constexpr typename PARAMETERS_TYPE::Integration integration = {
            1.0/((T)SIMULATION_FREQUENCY) // integration dt
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
                    0  // ramp_duration
                }
            }
        };

        static constexpr PARAMETERS_TYPE nominal_parameters = {
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
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr auto ACTION_INTERFACE = parameters::ActionInterface::DIRECT_MOTOR;
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 32;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT_FACTORY_BASE::EPISODE_STEP_LIMIT_OUTER;
            static constexpr TI CLOSED_FORM = false;
            using STATE_BASE_INNER = StateBase<StateSpecification<T, TI>>;
            using STATE_BASE_LA = StateLastAction<StateSpecification<T, TI, STATE_BASE_INNER>>;
            using STATE_BASE_LAA = StateLinearAcceleration<StateSpecification<T, TI, STATE_BASE_LA>>;
            using STATE_BASE_GB = StateGyroBias<StateGyroBiasSpecification<T, TI, STATE_BASE_LAA>>;
            // Active Mahony filter (default KP=1, KI=0.3). Sim-to-real: train with the same
            // filter that runs on the real drone so the policy is robust to its lag/error.
            using STATE_BASE = StateMahony<StateMahonySpecification<T, TI, STATE_BASE_GB>>;
            using STATE_TYPE = StateTrajectory<StateSpecification<T, TI, StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>>>;
            using OBSERVATION_TYPE = observation::OrientationMahonyWorldZ<observation::OrientationMahonyWorldZSpecification<T, TI,
                    observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                            observation::LinearAccelerationBodyFrame<observation::LinearAccelerationBodyFrameSpecification<T, TI,
                                    observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = typename ENVIRONMENT_FACTORY_BASE::ENVIRONMENT_STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
            static constexpr T STATE_LIMIT_POSITION_X = 0;
            static constexpr T STATE_LIMIT_POSITION_Y = 0;
            static constexpr T STATE_LIMIT_POSITION_Z = 0;
            static constexpr T STATE_LIMIT_VELOCITY_X = 10000;
            static constexpr T STATE_LIMIT_VELOCITY_Y = 10000;
            static constexpr T STATE_LIMIT_VELOCITY_Z = 10000;
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
