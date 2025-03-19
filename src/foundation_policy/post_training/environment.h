#include "../pre_training/environment.h"

namespace builder{
    using namespace rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename T, typename TI, typename OPTIONS>
    struct ENVIRONMENT_FACTORY_POST_TRAINING{

        using BASE_ENV = ENVIRONMENT_FACTORY<DEVICE, T, TI, OPTIONS>;

        using PARAMETERS_TYPE = typename BASE_ENV::PARAMETERS_TYPE;
        static constexpr PARAMETERS_TYPE nominal_parameters = [](){
            PARAMETERS_TYPE params = {
                {
                    {
                        BASE_ENV::dynamics,
                        BASE_ENV::integration,
                        BASE_ENV::mdp,
                    }, // Base
                    BASE_ENV::disturbances
                }, // Disturbances
                BASE_ENV::domain_randomization
            }; // Domain Randomization
            if constexpr(OPTIONS::OBSERVATION_NOISE){
                params.mdp.observation_noise = {
                    0.00, // position
                    0.00, // orientation
                    0.00, // linear_velocity
                    0.00, // angular_velocity
                    0.00 // imu acceleration
                };
            }
            return params;
        }();

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI EPISODE_STEP_LIMIT = 5 * BASE_ENV::SIMULATION_FREQUENCY;
            static constexpr TI CLOSED_FORM = false;
            static constexpr bool RANDOMIZE_THRUST_CURVES = OPTIONS::RANDOMIZE_THRUST_CURVES;
            static constexpr bool RANDOMIZE_MOTOR_MAPPING = OPTIONS::RANDOMIZE_MOTOR_MAPPING;
            static constexpr bool OBSERVE_THRUST_CURVES = OPTIONS::RANDOMIZE_THRUST_CURVES && OPTIONS::OBSERVE_THRASH_MARKOV;
            static constexpr bool OBSERVE_MOTOR_POSITIONS = OPTIONS::RANDOMIZE_MOTOR_MAPPING && OPTIONS::OBSERVE_THRASH_MARKOV;
            static_assert(OPTIONS::ACTION_HISTORY_LENGTH >= 1);
            static constexpr TI ANGULAR_VELOCITY_DELAY = 0; // one step at 100hz = 10ms ~ delay from IMU to input to the policy: 1.3ms time constant of the IIR in the IMU (bw ~110Hz) + synchronization delay (2ms) + (negligible SPI transfer latency due to it being interrupt-based) + 1ms sensor.c RTOS loop @ 1khz + 2ms for the RLtools loop
            using STATE_BASE = StateAngularVelocityDelay<StateAngularVelocityDelaySpecification<T, TI, ANGULAR_VELOCITY_DELAY, StateLastAction<StateSpecification<T, TI, StateBase<StateSpecification<T, TI>>>>>>;
            using STATE_TYPE_MOTOR_DELAY = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, OPTIONS::ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>;
            using STATE_TYPE_NO_MOTOR_DELAY = StateRandomForce<StateSpecification<T, TI, STATE_BASE>>;
            using STATE_TYPE = rl_tools::utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, STATE_TYPE_MOTOR_DELAY, STATE_TYPE_NO_MOTOR_DELAY>;
            static_assert(!OPTIONS::ACTION_HISTORY || OPTIONS::MOTOR_DELAY, "Action history implies motor delay");
            using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                    observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                    observation::AngularVelocityDelayed<observation::AngularVelocityDelayedSpecification<T, TI, ANGULAR_VELOCITY_DELAY,
                    observation::Multiplex<observation::MultiplexSpecification<TI, OBSERVE_THRUST_CURVES, observation::ParametersThrustCurves<observation::ParametersThrustCurvesSpecification<T, TI, BASE_ENV::PARAMETERS_TYPE::N>>,
                    observation::Multiplex<observation::MultiplexSpecification<TI, OBSERVE_MOTOR_POSITIONS, observation::ParametersMotorPosition<observation::ParametersMotorPositionSpecification<T, TI, BASE_ENV::PARAMETERS_TYPE::N>>,
                    observation::ActionHistory<observation::ActionHistorySpecification<T, TI, OPTIONS::ACTION_HISTORY ? OPTIONS::ACTION_HISTORY_LENGTH : 1, // one-step action history to Markovify the d_action regularization
                    rl_tools::utils::typing::conditional_t<!OPTIONS::ACTION_HISTORY, observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>, observation::LastComponent<TI>>
                    // one-step action history to Markovify the d_action regularization
                    // observation::ParametersMass<observation::ParametersMassSpecification<T, TI
            >>>>>>>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = typename BASE_ENV::PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
            static constexpr TI N_DYNAMICS_VALUES = 1;
            static constexpr typename BASE_ENV::PARAMETERS_TYPE::Dynamics DYNAMICS_VALUES[N_DYNAMICS_VALUES] = {
                rl_tools::rl::environments::l2f::parameters::dynamics::registry<rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie, typename BASE_ENV::PARAMETERS_SPEC>
            };
            static constexpr T STATE_LIMIT_POSITION = 100000;
            static constexpr T STATE_LIMIT_VELOCITY = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
        // static_assert(rl::environments::PREVENT_DEFAULT_GET_UI<ENVIRONMENT>::value);
    };
}
