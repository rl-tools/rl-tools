#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_BIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_BIG_H

#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/persist_code.h>


#include "environment.h"
#include <rl_tools/utils/generic/typing.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f{
    namespace rlt = rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename T, typename TI, typename OPTIONS>
    struct ENVIRONMENT_BIG_FACTORY{

        static constexpr auto MODEL = rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie;
        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;

        using REWARD_FUNCTION = rl_tools::rl::environments::l2f::parameters::reward_functions::Squared<T>;
        static constexpr REWARD_FUNCTION reward_function = {
                false, // non-negative
                01.00, // scale
                00.50, // constant
                -100.00, // termination penalty
                01.00, // position
                00.10, // orientation
                00.00, // linear_velocity
                00.00, // angular_velocity
                00.00, // linear_acceleration
                00.00, // angular_acceleration
                00.00, // action
                01.00, // d_action
                00.00, // position_error_integral
        };

        struct DOMAIN_RANDOMIZATION_OPTIONS{
            static constexpr bool ON = false;
            static constexpr bool THRUST_TO_WEIGHT = ON;
            static constexpr bool MASS = ON;
            static constexpr bool TORQUE_TO_INERTIA = ON;
            static constexpr bool MASS_SIZE_DEVIATION = ON;
        };

        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION>;
        using PARAMETERS_TYPE = ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, ParametersDisturbances<ParametersSpecification<T, TI, ParametersBase<PARAMETERS_SPEC>>>>>;

        static constexpr auto dynamics = rl_tools::rl::environments::l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;

        static constexpr auto init = rl_tools::rl::environments::l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP::ActionNoise action_noise = {
            0, // std of additive gaussian noise onto the normalized action (-1, 1)
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
            true,  // enable
            1,     // position
            2,     // linear velocity
            35,    // angular velocity
            10000, // position integral
            50000, // orientation integral
        };
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            init,
            reward_function,
            { // observation_noise
                0.00,// position
                0.00, // orientation
                0.00, // linear_velocity
                0.00, // angular_velocity
                0.00, // imu acceleration
            },
            action_noise,
            termination
        };
        static constexpr TI SIMULATION_FREQUENCY = 100;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {
            1.0/((T)SIMULATION_FREQUENCY) // integration dt
        };
        static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = DOMAIN_RANDOMIZATION_OPTIONS::ON ? typename PARAMETERS_TYPE::DomainRandomization{
            1.5, // thrust_to_weight_min;
            5.0, // thrust_to_weight_max;
            0.001, // thrust_to_weight_by_torque_to_inertia_min;
            0.100, // thrust_to_weight_by_torque_to_inertia_max;
            0.02, // mass_min;
            5.00, // mass_max;
            0.1, // mass_size_deviation;
            0.0, // motor_time_constant;
            0.0, // rotor_torque_constant;
            0.0  // orientation_offset_angle_max;
        } : typename PARAMETERS_TYPE::DomainRandomization{
            0.0, // thrust_to_weight_min;
            0.0, // thrust_to_weight_max;
            0.0, // thrust_to_weight_by_torque_to_inertia_min;
            0.0, // thrust_to_weight_by_torque_to_inertia_max;
            0.0, // mass_min;
            0.0, // mass_max;
            0.0, // mass_size_deviation;
            0.0, // motor_time_constant;
            0.0, // rotor_torque_constant;
            0.0  // orientation_offset_angle_max;
        };
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0}, //{0, 0.027 * 9.81 / 3}, // random_force;
            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} //{0, 0.027 * 9.81 / 10000} // random_torque;
        };
        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                {
                    dynamics,
                    integration,
                    mdp
                }, // Base
                disturbances
            }, // Disturbances
            domain_randomization
        }; // DomainRandomization

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = OPTIONS::SEQUENTIAL_MODEL ? 1 : 16;
            static constexpr TI EPISODE_STEP_LIMIT = 5 * SIMULATION_FREQUENCY;
            static constexpr TI CLOSED_FORM = false;
            static constexpr TI ANGULAR_VELOCITY_DELAY = 0; // one step at 100hz = 10ms ~ delay from IMU to input to the policy: 1.3ms time constant of the IIR in the IMU (bw ~110Hz) + synchronization delay (2ms) + (negligible SPI transfer latency due to it being interrupt-based) + 1ms sensor.c RTOS loop @ 1khz + 2ms for the RLtools loop
            static constexpr TI ANGULAR_VELOCITY_HISTORY = ANGULAR_VELOCITY_DELAY;
            // using STATE_TYPE = StatePoseErrorIntegral<StateSpecification<T, TI, DefaultActionHistoryState<T, TI, ACTION_HISTORY_LENGTH, ANGULAR_VELOCITY_HISTORY>>>;
            using STATE_TYPE = DefaultActionHistoryState<T, TI, ACTION_HISTORY_LENGTH, ANGULAR_VELOCITY_HISTORY>;
            using OBSERVATION_TYPE_MARKOVIAN = DefaultActionHistoryObservation<T, TI, ACTION_HISTORY_LENGTH, ANGULAR_VELOCITY_DELAY,
                observation::ParametersMotorPosition<observation::ParametersMotorPositionSpecification<T, TI, PARAMETERS_TYPE::N,
                observation::ParametersThrustCurves<observation::ParametersThrustCurvesSpecification<T, TI, PARAMETERS_TYPE::N,
                observation::ParametersMass<observation::ParametersMassSpecification<T, TI,
                observation::ParametersInertia<observation::ParametersInertiaSpecification<T, TI
                // observation::PoseIntegral<observation::PoseIntegralSpecification<T, TI
            >>>>>>>>>;
            using OBSERVATION_TYPE_PO = DefaultActionHistoryObservation<T, TI, ACTION_HISTORY_LENGTH, ANGULAR_VELOCITY_DELAY>;
            using OBSERVATION_TYPE = OBSERVATION_TYPE_PO;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr PARAMETERS PARAMETER_VALUES = nominal_parameters;
            static constexpr T STATE_LIMIT_POSITION = 100000;
            static constexpr T STATE_LIMIT_VELOCITY = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif