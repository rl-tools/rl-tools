#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic.h>

// #include <rl_tools/rl/environments/l2f/persist_code.h>


#include <rl_tools/utils/generic/typing.h>

namespace builder{
    using namespace rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename T, typename TI, typename OPTIONS>
    struct ENVIRONMENT_FACTORY{

        static constexpr auto MODEL = parameters::dynamics::REGISTRY::crazyflie;
        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;

        using REWARD_FUNCTION = parameters::reward_functions::Squared<T>;
        static constexpr REWARD_FUNCTION reward_function = {
                false, // non-negative
                01.00, // scale
                01.50, // constant
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
            static constexpr bool ENABLED = false;
            static constexpr bool THRUST_TO_WEIGHT = ENABLED;
            static constexpr bool MASS = ENABLED;
            static constexpr bool THRUST_TO_WEIGHT_TO_TORQUE_TO_INERTIA = ENABLED;
            static constexpr bool MASS_SIZE_DEVIATION = ENABLED;
            static constexpr bool ROTOR_TORQUE_CONSTANT = ENABLED;
            static constexpr bool DISTURBANCE_FORCE = ENABLED;
        };

        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION>;
        using PARAMETERS_TYPE = ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, ParametersDisturbances<ParametersSpecification<T, TI, ParametersBase<PARAMETERS_SPEC>>>>>;

        static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = rl_tools::rl::environments::l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;

        static constexpr TI SIMULATION_FREQUENCY = 100;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {
            1.0/((T)SIMULATION_FREQUENCY) // integration dt
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = rl_tools::rl::environments::l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP::ObservationNoise observation_noise = {
            0, // position
            0, // orientation
            0, // linear_velocity
            0, // angular_velocity
            0, // imu acceleration
        };
        static constexpr typename PARAMETERS_TYPE::MDP::ActionNoise action_noise = {
            0, // std of additive gaussian noise onto the normalized action (-1, 1)
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
            true,  // enable
            1,     // position
            2,    // linear velocity
            35,    // angular velocity
            10000, // position integral
            50000, // orientation integral
        };
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            init,
            reward_function,
            observation_noise,
            action_noise,
            termination
        };
        static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
            0, // thrust_to_weight_min;
            0, // thrust_to_weight_max;
            0, // thrust_to_weight_by_torque_to_inertia_min;
            0, // thrust_to_weight_by_torque_to_inertia_max;
            0, // mass_min;
            0, // mass_max;
            0, // mass_size_deviation;
            0, // motor_time_constant_rising_min;
            0, // motor_time_constant_rising_max;
            0, // motor_time_constant_falling_min;
            0, // motor_time_constant_falling_max;
            0, // rotor_torque_constant_min;
            0, // rotor_torque_constant_max;
            0, // orientation_offset_angle_max;
            0  // disturbance_force_max;
        };
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0}, // random_force;
            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_torque;
        };
        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                {
                    dynamics,
                    integration,
                    mdp,
                }, // Base
                disturbances
            }, // Disturbances
            domain_randomization
        }; // Domain Randomization

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = OPTIONS::SEQUENTIAL_MODEL ? 1 : 1;
            static constexpr TI EPISODE_STEP_LIMIT = 5 * SIMULATION_FREQUENCY;
            static constexpr TI CLOSED_FORM = false;
            static constexpr bool RANDOMIZE_THRUST_CURVES = OPTIONS::RANDOMIZE_THRUST_CURVES;
            static constexpr bool RANDOMIZE_MOTOR_MAPPING = OPTIONS::RANDOMIZE_MOTOR_MAPPING;
            static constexpr bool OBSERVE_THRUST_CURVES = OPTIONS::RANDOMIZE_THRUST_CURVES && OPTIONS::OBSERVE_THRASH_MARKOV;
            static constexpr bool OBSERVE_MOTOR_POSITIONS = OPTIONS::RANDOMIZE_MOTOR_MAPPING && OPTIONS::OBSERVE_THRASH_MARKOV;
            static constexpr TI ANGULAR_VELOCITY_DELAY = 0; // one step at 100hz = 10ms ~ delay from IMU to input to the policy: 1.3ms time constant of the IIR in the IMU (bw ~110Hz) + synchronization delay (2ms) + (negligible SPI transfer latency due to it being interrupt-based) + 1ms sensor.c RTOS loop @ 1khz + 2ms for the RLtools loop
            using STATE_BASE = StateAngularVelocityDelay<StateAngularVelocityDelaySpecification<T, TI, ANGULAR_VELOCITY_DELAY, StateLastAction<StateSpecification<T, TI, StateBase<StateSpecification<T, TI>>>>>>;
            using STATE_TYPE_MOTOR_DELAY = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>;
            using STATE_TYPE_NO_MOTOR_DELAY = StateRandomForce<StateSpecification<T, TI, STATE_BASE>>;
            using STATE_TYPE = rl_tools::utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, STATE_TYPE_MOTOR_DELAY, STATE_TYPE_NO_MOTOR_DELAY>;
            using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                    observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                    observation::AngularVelocityDelayed<observation::AngularVelocityDelayedSpecification<T, TI, ANGULAR_VELOCITY_DELAY,
                    observation::Multiplex<observation::MultiplexSpecification<TI, OBSERVE_THRUST_CURVES, observation::ParametersThrustCurves<observation::ParametersThrustCurvesSpecification<T, TI, PARAMETERS_TYPE::N>>,
                    observation::Multiplex<observation::MultiplexSpecification<TI, OBSERVE_MOTOR_POSITIONS, observation::ParametersMotorPosition<observation::ParametersMotorPositionSpecification<T, TI, PARAMETERS_TYPE::N>>,
                    // observation::RandomForce<observation::RandomForceSpecification<T, TI,
                    observation::ActionHistory<observation::ActionHistorySpecification<T, TI, 1, // one-step action history to Markovify the d_action regularization
                    utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>, observation::LastComponent<TI>>
                    // observation::ParametersMass<observation::ParametersMassSpecification<T, TI
            >>>>>>>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
            static constexpr TI N_DYNAMICS_VALUES = 1;
            static constexpr typename PARAMETERS_TYPE::Dynamics DYNAMICS_VALUES[N_DYNAMICS_VALUES] = {
                rl_tools::rl::environments::l2f::parameters::dynamics::registry<parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>
            };
            static constexpr T STATE_LIMIT_POSITION = 100000;
            static constexpr T STATE_LIMIT_VELOCITY = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
        };

        using ENVIRONMENT_SPEC = Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl::environments::Multirotor<ENVIRONMENT_SPEC>;
        // static_assert(rl::environments::PREVENT_DEFAULT_GET_UI<ENVIRONMENT>::value);
    };
}
