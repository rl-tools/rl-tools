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

        using BASE_ENV = rl_tools::rl::environments::Multirotor<Specification<T, TI>>;

        static constexpr auto MODEL = parameters::dynamics::REGISTRY::crazyflie;
        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;

        using REWARD_FUNCTION = parameters::reward_functions::Squared<T>;
        struct DOMAIN_RANDOMIZATION_OPTIONS{
            static constexpr bool ENABLED = false;
            static constexpr bool THRUST_TO_WEIGHT = ENABLED;
            static constexpr bool MASS = ENABLED;
            static constexpr bool TORQUE_TO_INERTIA = ENABLED;
            static constexpr bool MASS_SIZE_DEVIATION = ENABLED;
            static constexpr bool ROTOR_TORQUE_CONSTANT = ENABLED;
            static constexpr bool DISTURBANCE_FORCE = ENABLED;
            static constexpr bool ROTOR_TIME_CONSTANT = ENABLED;
        };

        struct TRAJECTORY_OPTIONS{
            static constexpr bool LANGEVIN = true;
        };
        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION>;
        using PARAMETERS_TYPE = ParametersTrajectory<ParametersTrajectorySpecification<T, TI, TRAJECTORY_OPTIONS, ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, ParametersDisturbances<ParametersSpecification<T, TI, ParametersBase<PARAMETERS_SPEC>>>>>>>;

        static constexpr TI SIMULATION_FREQUENCY = 100;

        static constexpr auto BASE_PARAMS = BASE_ENV::SPEC::PARAMETER_VALUES;

        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                {
                    {
                        BASE_PARAMS.dynamics,
                        BASE_PARAMS.integration,
                        BASE_PARAMS.mdp
                    }, // Base
                    BASE_PARAMS.disturbances
                }, // Disturbances
                BASE_PARAMS.domain_randomization
            }, // DomainRandomization
            BASE_PARAMS.trajectory // Trajectory
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 1;
            static constexpr TI EPISODE_STEP_LIMIT = 5 * SIMULATION_FREQUENCY;
            static constexpr TI CLOSED_FORM = false;
            static constexpr TI ANGULAR_VELOCITY_DELAY = 0; // one step at 100hz = 10ms ~ delay from IMU to input to the policy: 1.3ms time constant of the IIR in the IMU (bw ~110Hz) + synchronization delay (2ms) + (negligible SPI transfer latency due to it being interrupt-based) + 1ms sensor.c RTOS loop @ 1khz + 2ms for the RLtools loop
            using STATE_BASE = StateAngularVelocityDelay<StateAngularVelocityDelaySpecification<T, TI, ANGULAR_VELOCITY_DELAY, StateLastAction<StateSpecification<T, TI, StateBase<StateSpecification<T, TI>>>>>>;
            using STATE_TYPE_MOTOR_DELAY = StateTrajectory<StateSpecification<T, TI, StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>>>;
            using STATE_TYPE_NO_MOTOR_DELAY = StateRandomForce<StateSpecification<T, TI, STATE_BASE>>;
            using STATE_TYPE = rl_tools::utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, STATE_TYPE_MOTOR_DELAY, STATE_TYPE_NO_MOTOR_DELAY>;
            using OBSERVATION_TYPE = observation::TrajectoryTrackingPosition<observation::PositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                    observation::TrajectoryTrackingLinearVelocity<observation::LinearVelocitySpecification<T, TI,
                    observation::AngularVelocityDelayed<observation::AngularVelocityDelayedSpecification<T, TI, ANGULAR_VELOCITY_DELAY,
                    // observation::RandomForce<observation::RandomForceSpecification<T, TI,
                    observation::ActionHistory<observation::ActionHistorySpecification<T, TI, 1, // one-step action history to Markovify the d_action regularization
                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>>
            >>>>>>>>>;
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
