#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_BIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_BIG_H

#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic.h>
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
    template <typename DEVICE, typename T, typename TI, typename OPTIONS>
    struct ENVIRONMENT_BIG_FACTORY{
        using ENVIRONMENT_FACTORY_BASE = ENVIRONMENT_FACTORY<DEVICE, T, TI>;
        using PARAMETERS_SPEC = typename ENVIRONMENT_FACTORY_BASE::PARAMETERS_SPEC;
        using PARAMETERS_TYPE = typename ENVIRONMENT_FACTORY_BASE::PARAMETERS_TYPE;

        using REWARD_FUNCTION = rl_tools::rl::environments::l2f::parameters::reward_functions::Squared<T>;
        static constexpr REWARD_FUNCTION reward_function = {
                false, // non-negative
                01.00, // scale
                01.50, // constant
                00.00, // termination penalty
                01.00, // position
                00.10, // orientation
                00.00, // linear_velocity
                00.00, // angular_velocity
                00.00, // linear_acceleration
                00.00, // angular_acceleration
                00.00, // action
                01.00, // d_action
        };

        static constexpr auto termination = [](){
            auto termination = ENVIRONMENT_FACTORY_BASE::termination;
            termination.linear_velocity_threshold = 2;
            return termination;
        }();

        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            ENVIRONMENT_FACTORY_BASE::init,
            reward_function,
            { // observation_noise
                0.00,// position
                0.00, // orientation
                0.00, // linear_velocity
                0.00, // angular_velocity
                0.00, // imu acceleration
            },
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
                    {
                        dynamics,
                        integration,
                        mdp
                    }, // Base
                    typename PARAMETERS_TYPE::Disturbances{
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0}, //{0, 0.027 * 9.81 / 3}, // random_force;
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} //{0, 0.027 * 9.81 / 10000} // random_torque;
                    }
                }, // Disturbances
                ENVIRONMENT_FACTORY_BASE::domain_randomization
            }; // DomainRandomization
        }

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = OPTIONS::SEQUENTIAL_MODEL ? 1 : 16;
            static constexpr TI EPISODE_STEP_LIMIT = 5 * SIMULATION_FREQUENCY;
            static constexpr TI CLOSED_FORM = false;
            static constexpr bool RANDOMIZE_THRUST_CURVES = OPTIONS::RANDOMIZE_THRUST_CURVES;
            static constexpr bool RANDOMIZE_MOTOR_MAPPING = OPTIONS::RANDOMIZE_MOTOR_MAPPING;
            static constexpr bool OBSERVE_THRUST_CURVES = OPTIONS::RANDOMIZE_THRUST_CURVES && OPTIONS::OBSERVE_THRASH_MARKOV;
            static constexpr bool OBSERVE_MOTOR_POSITIONS = OPTIONS::RANDOMIZE_MOTOR_MAPPING && OPTIONS::OBSERVE_THRASH_MARKOV;
            static constexpr TI ANGULAR_VELOCITY_DELAY = 0; // one step at 100hz = 10ms ~ delay from IMU to input to the policy: 1.3ms time constant of the IIR in the IMU (bw ~110Hz) + synchronization delay (2ms) + (negligible SPI transfer latency due to it being interrupt-based) + 1ms sensor.c RTOS loop @ 1khz + 2ms for the RLtools loop
            using STATE_BASE = StateAngularVelocityDelay<StateAngularVelocityDelaySpecification<T, TI, ANGULAR_VELOCITY_DELAY, StateLastAction<StateSpecification<T, TI, StateBase<StateSpecification<T, TI>>>>>>; // make sure to also change the observation to the delayed one
            using STATE_TYPE_MOTOR_DELAY = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>;
            using STATE_TYPE_NO_MOTOR_DELAY = StateRandomForce<StateSpecification<T, TI, STATE_BASE>>;
            using STATE_TYPE = rl_tools::utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, STATE_TYPE_MOTOR_DELAY, STATE_TYPE_NO_MOTOR_DELAY>;
            using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                    observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                    observation::AngularVelocityDelayed<observation::AngularVelocityDelayedSpecification<T, TI, ANGULAR_VELOCITY_DELAY,
                    observation::Multiplex<observation::MultiplexSpecification<TI, OBSERVE_THRUST_CURVES, observation::ParametersThrustCurves<observation::ParametersThrustCurvesSpecification<T, TI, PARAMETERS_TYPE::N>>,
                    observation::Multiplex<observation::MultiplexSpecification<TI, OBSERVE_MOTOR_POSITIONS, observation::ParametersMotorPosition<observation::ParametersMotorPositionSpecification<T, TI, PARAMETERS_TYPE::N>>,
                    rl_tools::utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>, observation::LastComponent<TI>>
                    // observation::ParametersMass<observation::ParametersMassSpecification<T, TI
            >>>>>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters(ENVIRONMENT_FACTORY_BASE::dynamics);
            static constexpr TI N_DYNAMICS_VALUES = 1;
            static constexpr typename PARAMETERS_TYPE::Dynamics DYNAMICS_VALUES[N_DYNAMICS_VALUES] = {
                rl_tools::rl::environments::l2f::parameters::dynamics::registry<rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>
            };
            static constexpr T STATE_LIMIT_POSITION = 100000;
            static constexpr T STATE_LIMIT_VELOCITY = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::MultiTaskSpecification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::MultirotorMultiTask<ENVIRONMENT_SPEC>;
        static_assert(rl::environments::PREVENT_DEFAULT_GET_UI<ENVIRONMENT>::value);
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif